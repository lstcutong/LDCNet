import pointops
import torch
import torch.nn as nn
from pointpd.models.builder import MODELS
from torch_geometric.nn.pool import voxel_grid
from torch_scatter import segment_csr
from pointpd.models.modules.block_helper import * 

# furthest point sampling
@MODELS.register_module('fps')
class FPSampling(nn.Module):
    def __init__(self, stride, **kwargs):
        super().__init__()
        self.stride = stride

    def forward(self, input_dict):
        '''
        return: n_p [n // stride, 3], n_o [B], idx
        '''
        p, o = input_dict["p"], input_dict["o"]
        n_o = sampling_offset_helper(o, self.stride)
        idx = pointops.farthest_point_sampling(p, o, n_o)
        n_p = p[idx.long(), :]
        return n_p, n_o, idx


# random sampling
@MODELS.register_module('rs')
class RSampling(nn.Module):
    def __init__(self, stride, **kwargs):
        super().__init__()
        self.stride = stride

    def forward(self, input_dict):
        '''
        return: n_p [n // stride, 3], n_o [B], idx
        '''
        p, o = input_dict["p"], input_dict["o"]
        n_o = sampling_offset_helper(o, self.stride)
        n, b, n_max = p.shape[0], o.shape[0], o[0]

        idx = torch.cuda.IntTensor(n_o[b-1].item()).zero_()
        for i in range(0, b):
            if i==0:
                idx[0: n_o[0]] = torch.randperm(o[0])[:n_o[0]].int().to(p.device)
            else:
                idx[n_o[i-1]:n_o[i]] = torch.randperm(o[i] - o[i-1])[:n_o[i] - n_o[i-1]].int().to(p.device) + o[i - 1]
        n_p = p[idx.long(), :]
        return n_p, n_o, idx


# voxel sampling
@MODELS.register_module('vs')
class VSampling(nn.Module):
    def __init__(self, voxel_size, **kwargs):
        super().__init__()
        self.voxel_size = voxel_size

    def forward(self, input_dict, start=None):
        '''
        return: return: n_p [?, 3], n_o [B], idx
        '''
        coord, offset = input_dict["p"], input_dict["o"]
        batch, sorted_cluster_indices, idx_ptr = voxelize_helper(coord, offset, self.voxel_size, start)

        n_p = segment_csr(coord[sorted_cluster_indices], idx_ptr, reduce="mean")
        batch = batch[idx_ptr[:-1]]
        n_o = pointops.batch2offset(batch)
        return n_p, n_o, (sorted_cluster_indices, idx_ptr)




