import torch
import torch.nn as nn
from typing import List, Type
from ..builder import MODELS
import copy

'''
@MODELS.register_module()
class BalancedSemSegHead(nn.Module):
    def __init__(self,
                 num_classes, in_channels,
                 mlps=None,
                 norm_args={'norm': 'bn1d'},
                 act_args={'act': 'relu'},
                 dropout=0.5,
                 global_feat=None, 
                 ):
        """semantic segmentation head for balanced batch representation.
        Args:
            num_classes: class num.
            in_channles: the base channel num.
        Returns:
            logits: (B, num_classes, N)
        """
        super().__init__()

        if global_feat is not None:
            self.global_feat = global_feat.split(',')
            multiplier = len(self.global_feat) + 1
        else:
            self.global_feat = None
            multiplier = 1
        in_channels *= multiplier
        
        if mlps is None:
            mlps = [in_channels, in_channels] + [num_classes]
        else:
            if not isinstance(mlps, List):
                mlps = [mlps]
            mlps = [in_channels] + mlps + [num_classes]
        heads = []
        for i in range(len(mlps) - 2):
            heads.append(create_convblock1d(mlps[i], mlps[i + 1],
                                            norm_args=norm_args,
                                            act_args=act_args))
            if dropout:
                heads.append(nn.Dropout(dropout))

        heads.append(create_convblock1d(mlps[-2], mlps[-1], act_args=None))
        self.head = nn.Sequential(*heads)

    def forward(self, input_dict):
        end_points = input_dict["feature"]

        if self.global_feat is not None: 
            global_feats = [] 
            for feat_type in self.global_feat:
                if 'max' in feat_type:
                    global_feats.append(torch.max(end_points, dim=-1, keepdim=True)[0])
                elif feat_type in ['avg', 'mean']:
                    global_feats.append(torch.mean(end_points, dim=-1, keepdim=True))
            global_feats = torch.cat(global_feats, dim=1).expand(-1, -1, end_points.shape[-1])
            end_points = torch.cat((end_points, global_feats), dim=1)

        logits = self.head(end_points)
 
        b, cl, n = logits.shape
        offset = input_dict["offset"]
        bs = 0
        true_logits = []
        for i in range(len(offset)):
            es = offset[i]
            true_logits.append(logits[i, :, 0: es - bs].permute((1, 0)))
            bs = es
        
        logits = torch.cat(true_logits).reshape((-1, cl))

        return logits
'''

@MODELS.register_module()
class OffsetSemSegHead(nn.Module):
    def __init__(self,
                 num_classes, in_channels,
                 ):
        """semantic segmentation head for offset representation.
        Args:
            num_classes: class num.
            in_channles: the base channel num.
        Returns:
            logits: (N, num_classes)
        """
        super().__init__()
        
        self.seg_head = (
            nn.Sequential(
                nn.Linear(in_channels, in_channels),
                torch.nn.BatchNorm1d(in_channels),
                nn.ReLU(inplace=True),
                nn.Linear(in_channels, num_classes),
            )
            if num_classes > 0
            else nn.Identity()
        )

    def forward(self, input_dict):
        end_points = input_dict["feature"]
        logits = self.seg_head(end_points)

        return logits