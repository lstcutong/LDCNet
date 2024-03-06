import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn.pool import voxel_grid
from torch_scatter import segment_csr
import pointops


class PointBatchNorm(nn.Module):
    """
    Batch Normalization for Point Clouds data in shape of [B*N, C], [B*N, L, C]
    """

    def __init__(self, embed_channels):
        super().__init__()
        self.norm = nn.BatchNorm1d(embed_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() == 3:
            return (
                self.norm(input.transpose(1, 2).contiguous())
                .transpose(1, 2)
                .contiguous()
            )
        elif input.dim() == 2:
            return self.norm(input)
        else:
            raise NotImplementedError


def sampling_offset_helper(o, stride):
    n_o, count = [o[0].item() // stride], o[0].item() // stride
    for i in range(1, o.shape[0]):
        count += (o[i].item() - o[i - 1].item()) // stride
        n_o.append(count)
    n_o = torch.cuda.IntTensor(n_o)
    return n_o


def voxelize_helper(coord, offset, voxel_size, start):
    batch = pointops.offset2batch(offset)
    start = (
        segment_csr(
            coord,
            torch.cat([batch.new_zeros(1), torch.cumsum(batch.bincount(), dim=0)]),
            reduce="min",
        )
        if start is None
        else start
    )
    cluster = voxel_grid(
        pos=coord - start[batch], size=voxel_size, batch=batch, start=0
    )
    unique, cluster, counts = torch.unique(
        cluster, sorted=True, return_inverse=True, return_counts=True
    )
    _, sorted_cluster_indices = torch.sort(cluster)
    idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])

    return batch, sorted_cluster_indices, idx_ptr


import torch
import torch.nn as nn
import pointops
from ..builder import MODELS

class UnaryBlockPXO(nn.Module):
    expansion = 1
    def __init__(self, inc, outc):
        super(UnaryBlockPXO, self).__init__()
        self.linear = nn.Linear(inc, outc, bias=False)
        self.bn = nn.BatchNorm1d(outc)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        x = self.relu(self.bn(self.linear(x)))
        return [p, x, o]

class UnaryBlockX(nn.Module):
    def __init__(self, inc, outc):
        super(UnaryBlockX, self).__init__()
        self.linear = nn.Linear(inc, outc, bias=False)
        self.bn = nn.BatchNorm1d(outc)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn(self.linear(x)))
        return x


class GlobalPooling(nn.Module):
    def __init__(self, in_planes):
        super().__init__()
        self.linear1 = nn.Sequential(
                nn.Linear(2 * in_planes, in_planes),
                nn.BatchNorm1d(in_planes),
                nn.ReLU(inplace=True),
            )
        self.linear2 = nn.Sequential(
            nn.Linear(in_planes, in_planes), nn.ReLU(inplace=True)
        )
    
    def forward(self, pxo):
        _, x, o = pxo  # (n, 3), (n, c), (b)
        x_tmp = []
        for i in range(o.shape[0]):
            if i == 0:
                s_i, e_i, cnt = 0, o[0], o[0]
            else:
                s_i, e_i, cnt = o[i - 1], o[i], o[i] - o[i - 1]
            x_b = x[s_i:e_i, :]
            x_b = torch.cat(
                (x_b, self.linear2(x_b.sum(0, True) / cnt).repeat(cnt, 1)), 1
            )
            x_tmp.append(x_b)
        x = torch.cat(x_tmp, 0)
        x = self.linear1(x)
        return x

class TransitionUp(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(out_planes, out_planes),
            nn.BatchNorm1d(out_planes),
            nn.ReLU(inplace=True),
        )
        self.linear2 = nn.Sequential(
            nn.Linear(in_planes, out_planes),
            nn.BatchNorm1d(out_planes),
            nn.ReLU(inplace=True),
        )

    def forward(self, pxo1, pxo2):
        p1, x1, o1 = pxo1
        p2, x2, o2 = pxo2
        x = self.linear1(x1) + pointops.interpolation(
            p2, p1, self.linear2(x2), o2, o1
        )
        return x


class TransitionDown(nn.Module):
    def __init__(self, in_planes, out_planes, sampler, pooling):
        super().__init__()
        if pooling["type"] == "grid-pool":
            self.linear = nn.Linear(in_planes, out_planes, bias=False)
            self.sampling_pooling_together = True
            self.sampler = None
            self.pooling = MODELS.build(pooling)
        else:
            self.linear = nn.Linear(3 + in_planes, out_planes, bias=False)
            self.sampling_pooling_together = False
            self.sampler = MODELS.build(sampler)
            self.pooling = MODELS.build(pooling)

        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        
        if self.sampling_pooling_together:
            n_p, x, n_o = self.pooling(dict(
                source_p = p,
                source_f = x,
                source_o = o
            ))
            x = self.relu(
                self.bn(self.linear(x))
            )
        else:
            n_p, n_o, _ = self.sampler(
                dict(p=p, o=o)
            )
            x = self.pooling(dict(
                source_p = p,
                target_p = n_p,
                source_f = x,
                source_o = o,
                target_o = n_o,
                with_xyz = True,
            ))
            x = self.relu(
                self.bn(self.linear(x))
            )
            # (m, c, nsample)
        #x = self.pool(x).squeeze(-1)  # (m, c)
        p, o = n_p, n_o

        return [p, x, o]