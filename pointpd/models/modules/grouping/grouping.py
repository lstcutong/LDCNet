import pointops
import torch
import torch.nn as nn
from pointpd.models.builder import MODELS
import numpy as np


@MODELS.register_module('knn-grouper')
class KnnGrouping(nn.Module):
    def __init__(self, nsample, dilation, **kwargs):
        super().__init__()
        self.nsample = nsample
        self.dilation = dilation

    def forward(self, input_dict):
        '''
        return: grouped_feat [ntarget, nsample, c], idx [ntarget, nsample]
        '''
        return pointops.query_and_group(
            nsample = self.nsample,
            xyz = input_dict['source_p'],
            new_xyz = input_dict['target_p'],
            feat = input_dict['source_f'],
            idx=input_dict['idx'] if 'idx' in input_dict.keys() else None,
            offset=input_dict['source_o'],
            new_offset=input_dict['target_o'],
            dilation=self.dilation,
            with_feat=True,
            with_xyz=input_dict['with_xyz']
        )

@MODELS.register_module('ball-grouper')
class BallGrouping(nn.Module):
    def __init__(self, max_radius, min_radius, nsample, **kwargs):
        super().__init__()
        self.max_radius = max_radius
        self.min_radius = min_radius
        self.nsample = nsample

    def forward(self, input_dict):
        '''
        return: grouped_feat [ntarget, nsample, c], idx [ntarget, nsample]
        '''
        a, b = pointops.ball_query_and_group(
            feat = input_dict['source_f'],
            xyz = input_dict['source_p'],
            offset=input_dict['source_o'],
            new_xyz=input_dict['target_p'],
            new_offset=input_dict['target_o'],
            idx=None,
            max_radio=self.max_radius,
            min_radio=self.min_radius,
            nsample=self.nsample,
            with_xyz=input_dict['with_xyz'],
        )
        #print(a, b)
        return a,b

'''
@MODELS.register_module('ball-grouper-tp')
class BallGroupingTP(nn.Module):
    def __init__(self, max_radius, min_radius, nsample, **kwargs):
        super().__init__()
        self.max_radius = max_radius
        self.min_radius = min_radius
        self.nsample = nsample

    def forward(self, input_dict):
        
        return: grouped_feat [ntarget, nsample, c], idx [ntarget, nsample]
        
        idx, dist = tp.ball_query(
            self.max_radius,
            self.nsample,
            input_dict['source_p'],
            input_dict['target_p'],
            mode="partial_dense",
            batch_x=pointops.offset2batch(input_dict['source_o']),
            batch_y=pointops.offset2batch(input_dict['target_o']),
        )
        return pointops.grouping(idx, input_dict['source_f'], input_dict['source_p'], input_dict['target_p'], input_dict['with_xyz']), idx
'''