import pointops
import torch
import torch.nn as nn
from pointpd.models.builder import MODELS
from pointpd.models.modules.block_helper import *
from pointpd.models.modules.grouping import KnnGrouping

@MODELS.register_module('knn-pool')
class KnnPooling(nn.Module):
    def __init__(self, nsample, dilation, reduction="max", **kwargs):
        super().__init__()
        self.group = KnnGrouping(nsample, dilation)
        assert reduction in ['sum', 'max', 'mean']
        if reduction == 'max':
            self.pool = lambda x: torch.max(x, dim=1, keepdim=False)[0]
        elif reduction == 'mean':
            self.pool = lambda x: torch.mean(x, dim=1, keepdim=False)
        elif reduction == 'sum':
            self.pool = lambda x: torch.sum(x, dim=1, keepdim=False)
        

    def forward(self, input_dict):
        '''
        return: n_p [n // stride, 3], n_o [B], idx
        '''
        feat, idx = self.group(input_dict)
        new_feat = self.pool(feat)
        return new_feat


@MODELS.register_module('grid-pool')
class GridPooling(nn.Module):
    def __init__(self, grid_size, reduction="max"):
        super().__init__()
        self.grid_size = grid_size
        self.reduction = reduction

    def forward(self, input_dict, start=None):
        coord, feat, offset = input_dict['source_p'], input_dict['source_f'], input_dict['source_o']
        batch, sorted_cluster_indices, idx_ptr = voxelize_helper(coord, offset, self.grid_size, start)

        coord = segment_csr(coord[sorted_cluster_indices], idx_ptr, reduce="mean")
        feat = segment_csr(feat[sorted_cluster_indices], idx_ptr, reduce=self.reduction)
        batch = batch[idx_ptr[:-1]]
        offset = pointops.batch2offset(batch)
        return [coord, feat, offset]
