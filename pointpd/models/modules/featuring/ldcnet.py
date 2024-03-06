import torch
import torch.nn as nn
import pointops
import einops
from pointpd.models.builder import MODELS
from ..block_helper import *

torch.nn.LayerNorm


def queryandgroup_global(xyz, feat, offset, use_xyz=True):
    idx = []
    base_offset = 0
    for i in range(len(offset)):
        if i==0:
            base_offset = offset[0]
            idx.append(torch.arange(0,offset[0]).unsqueeze(0).repeat((offset[0],1)))
        else:
            current_offset = offset[i] - offset[i-1]
            if current_offset != base_offset:
                raise ValueError
            idx.append(torch.arange(offset[i-1],offset[i]).unsqueeze(0).repeat((current_offset,1)))
    idx = torch.cat(idx, dim=0) # n, base_offset
    n, c = xyz.shape[0],  feat.shape[1]
    grouped_xyz = xyz[idx.view(-1).long(), :].view(n, base_offset, 3) # (n, base_offset, 3)
    grouped_xyz -= xyz.unsqueeze(1)

    grouped_feat = feat[idx.view(-1).long(), :].view(n, base_offset, c) # (n, base_offset, c)
    if use_xyz:
        return torch.cat((grouped_xyz, grouped_feat), -1) # (m, nsample, 3+c)
    else:
        return grouped_feat


def calculate_relation(feat1, feat2, relation_type="-"):
    '''
    input: feat1: (n,c1), feat2: (n, c2)
    output: n, c2
    '''
    n1, c1 = feat1.shape
    n2, c2 = feat2.shape
    assert n1 == n2
    if relation_type == "-":
        rel = torch.sub
    elif relation_type == "+":
        rel = torch.add
    elif relation_type == "*":
        rel = torch.mul
    else:
        raise NotImplementedError
    
    if c1 == c2:
        return rel(feat1, feat2)
    else:
        return rel(feat1.unsqueeze(1), feat2.unsqueeze(2)).sum(2)
    

def get_keypoints_furthest(pxo, num_keypoints):
    p, x, o = pxo
    idx = pointops.farthest_point_sampling(p, o, torch.cuda.IntTensor([num_keypoints * i for i in range(1, len(o) + 1)]))
    kp = p[idx.long(), :]
    kx = x[idx.long(), :]
    ko = torch.cuda.IntTensor([num_keypoints * i for i in range(1, len(o) + 1)])
    return kp, kx, ko


class LayerNorm1d(nn.BatchNorm1d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(input.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()


class PointTransformerLayerGlobal(nn.Module):
    def __init__(self, in_planes, out_planes, share_planes=8):
        super().__init__()
        self.mid_planes = mid_planes = out_planes // 1
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.linear_q = nn.Linear(in_planes, mid_planes)
        self.linear_k = nn.Linear(in_planes, mid_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)
        self.linear_p = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3), nn.ReLU(inplace=True), nn.Linear(3, out_planes))
        self.linear_w = nn.Sequential(nn.BatchNorm1d(mid_planes), nn.ReLU(inplace=True),
                                    nn.Linear(mid_planes, mid_planes // share_planes),
                                    nn.BatchNorm1d(mid_planes // share_planes), nn.ReLU(inplace=True),
                                    nn.Linear(out_planes // share_planes, out_planes // share_planes))
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, pxo) -> torch.Tensor:
        p, x, o = pxo  # (n, 3), (n, c), (b)
        x_q, x_k, x_v = self.linear_q(x), self.linear_k(x), self.linear_v(x)  # (n, c)
        x_k = queryandgroup_global(p, x_k, o, use_xyz=True)  # (n, nsample, 3+c)
        x_v = queryandgroup_global(p, x_v, o, use_xyz=False)  # (n, nsample, c)
        p_r, x_k = x_k[:, :, 0:3], x_k[:, :, 3:]
        for i, layer in enumerate(self.linear_p): p_r = layer(p_r.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i == 1 else layer(p_r)    # (n, nsample, c)
        w = x_k - x_q.unsqueeze(1) + p_r.view(p_r.shape[0], p_r.shape[1], self.out_planes // self.mid_planes, self.mid_planes).sum(2)  # (n, nsample, c)
        for i, layer in enumerate(self.linear_w): w = layer(w.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i % 3 == 0 else layer(w)
        w = self.softmax(w)  # (n, nsample, c)
        n, nsample, c = x_v.shape; s = self.share_planes
        x = ((x_v + p_r).view(n, nsample, s, c // s) * w.unsqueeze(2)).sum(1).view(n, c)
        return x


class LDCReasoning(nn.Module):
    def __init__(self, in_planes, planes, share_planes=8):
        super().__init__()
        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.transformer = PointTransformerLayerGlobal(planes, planes, share_planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.linear3 = nn.Linear(planes, planes, bias=False)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        identity = x
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.transformer([p, x, o])))
        x = self.bn3(self.linear3(x))
        x += identity
        x = self.relu(x)
        return [p, x, o]


class LDCAugmentation(nn.Module):
    def __init__(self, c1, c2, use_pca=False):
        super().__init__()
        if c1 == c2:
            self.linear = nn.Linear(c1, c1, bias=False)
        else:
            self.linear = nn.Linear(c1 + c2, c1, bias=False)
        self.use_pca = use_pca
        self.bn = nn.BatchNorm1d(c1)
        self.relu = nn.ReLU(inplace=True)
        self.linear_w = nn.Sequential(nn.BatchNorm1d(c2), nn.ReLU(inplace=True),
                                    nn.Linear(c2, c2 // 8),
                                    nn.BatchNorm1d(c2 // 8), nn.ReLU(inplace=True),
                                    nn.Linear(c2 // 8, c2))


    def pca(self, pxo):
        pass

    def forward(self, pxo, gpxo):
        p, x, o = pxo  # (n, 3), (n, c1), (b)
        gp, gx, go = gpxo # (m, 3), (m, c2), (b)

        n, c1 = x.shape
        m, c2 = gx.shape

        idx, _ = pointops.knn_query(1, gp, go, p, o) #第三个参数是中心点, idx: n,1 dist: n,1

        aug_feat = gx[idx.long().view(-1), :] # n,c2

        if self.use_pca:
            pass

        relation = calculate_relation(x, aug_feat, relation_type="-") # n, c2
        for i, layer in enumerate(self.linear_w): relation  = layer(relation)

        soft_weight = torch.nn.functional.softmax(relation, 1)
        aug_feat = soft_weight * aug_feat

        if c1 == c2:
            x = x + aug_feat
        else:
            x = torch.cat((x, aug_feat), 1)
        x = self.bn(self.linear(x))
        #x = self.bn(x)
        x = self.relu(x)
        return [p, x, o]


class MSSCA(nn.Module):
    def __init__(self, in_planes, out_planes, spatial=[8, 16, 32]):
        super().__init__()
        self.enc = UnaryBlockPXO(in_planes, out_planes)
        self.spatial = spatial
    
    def forward(self, pxo):
        p,x,o = self.enc(pxo)
        out = [x]
        for sp in self.spatial:
            x_v, _ = pointops.knn_query_and_group(x, p, o, p, o, None, sp, False)
            x_v = x_v.mean(1)
            out.append(x_v)
        
        out = torch.cat(out, dim=1)
        return p, out, o


