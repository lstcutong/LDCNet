"""Official implementation of PointNext
PointNeXt: Revisiting PointNet++ with Improved Training and Scaling Strategies
https://arxiv.org/abs/2206.04670
Guocheng Qian, Yuchen Li, Houwen Peng, Jinjie Mai, Hasan Abed Al Kader Hammoud, Mohamed Elhoseiny, Bernard Ghanem
"""
from typing import List, Type
import logging
import torch
import torch.nn as nn
from pointpd.models.builder import MODELS



def get_reduction_fn(reduction):
    reduction = 'mean' if reduction.lower() == 'avg' else reduction
    assert reduction in ['sum', 'max', 'mean']
    if reduction == 'max':
        pool = lambda x: torch.max(x, dim=1, keepdim=False)[0]
    elif reduction == 'mean':
        pool = lambda x: torch.mean(x, dim=1, keepdim=False)
    elif reduction == 'sum':
        pool = lambda x: torch.sum(x, dim=1, keepdim=False)
    return pool

class MLP(nn.Module):
    def __init__(self, inc, outc):
        super(MLP, self).__init__()
        self.linear = nn.Linear(inc, outc, bias=False)
        self.bn = nn.BatchNorm1d(outc)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn(self.linear(x)))
        return x


class LocalAggregation(nn.Module):
    """Local aggregation layer for a set 
    Set abstraction layer abstracts features from a larger set to a smaller set
    Local aggregation layer aggregates features from the same set
    """

    def __init__(self,
                 channels: List[int],
                 grouper,
                 reduction='max',
                 **kwargs
                 ):
        super().__init__()
        channels[0] = channels[0] + 3
        convs = []
        for i in range(len(channels) - 1):  # #layers in each blocks
            convs.append(MLP(channels[i], channels[i + 1]))
        self.convs = nn.Sequential(*convs)
        self.grouper = MODELS.build(grouper)
        self.pool = get_reduction_fn("max")

    def forward(self, pxo):
        # p: position, f: feature
        p, x, o = pxo
        # neighborhood_features
        grouped_feat, _ = self.grouper(dict(
                source_p = p,
                target_p = p,
                source_f = x,
                source_o = o,
                target_o = o,
                with_xyz = True,
            )
        )
        x = self.pool(grouped_feat)
        x= self.convs(x)
        return x

class SetAbstraction(nn.Module):
    """The modified set abstraction module in PointNet++ with residual connection support
    """
    def __init__(self,
                 in_channels, out_channels,
                 sampler, pooling,
                 layers=1,
                 use_res=False,
                 **kwargs, 
                 ):
        super().__init__()
        self.use_res = use_res
        self.out_channels = out_channels
        if self.use_res:
            self.skipconv = nn.Sequential(
                nn.Linear(in_channels + 3, out_channels),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True)
            )

        self.convs = nn.Sequential(
                nn.Linear(in_channels + 3, out_channels),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True)
        )

        if pooling["type"] == "grid-pool":
            self.sampling_pooling_together = True
            self.sampler = None
            self.pooling = MODELS.build(pooling)
        else:
            self.sampling_pooling_together = False
            self.sampler = MODELS.build(sampler)
            self.pooling = MODELS.build(pooling)
            

    def forward(self, pxo):
        p, x, o = pxo
        if self.sampling_pooling_together:
            n_p, x, n_o = self.pooling(dict(
                source_p = p,
                source_f = x,
                source_o = o
            ))
            x = self.convs(torch.cat([n_p, x], dim=1))
        else:
            n_p, n_o, idx = self.sampler(
                dict(p=p, o=o)
            )
            try:
                identity = torch.cat([n_p, x[idx.long(), :]], dim=1)
                identity = self.skipconv(identity) if self.use_res else torch.zeros((n_p.shape[0], self.out_channels)).to(x.device).float()
            except:
                identity = torch.zeros((n_p.shape[0], self.out_channels)).to(x.device).float()
            x = self.pooling(dict(
                source_p = p,
                target_p = n_p,
                source_f = x,
                source_o = o,
                target_o = n_o,
                with_xyz = True,
            ))
            x = self.convs(x)

        p = n_p
        o = n_o
        return p, x, o




@MODELS.register_module()
class InvResMLP(nn.Module):
    def __init__(self,
                 in_channels,
                 grouper,
                 expansion=1,
                 use_res=True,
                 num_posconvs=2,
                 **kwargs
                 ):
        super().__init__()
        self.use_res = use_res
        mid_channels = int(in_channels * expansion)
        self.convs = LocalAggregation([in_channels, in_channels], grouper=grouper, **kwargs)
        if num_posconvs < 1:
            channels = []
        elif num_posconvs == 1:
            channels = [in_channels, in_channels]
        else:
            channels = [in_channels, mid_channels, in_channels]
        pwconv = []
        for i in range(len(channels) - 1):
            pwconv.append(MLP(channels[i], channels[i + 1]))
                          
        self.pwconv = nn.Sequential(*pwconv)

    def forward(self, pxo):
        p, x, o = pxo
        identity = x
        x = self.convs([p, x, o])
        x = self.pwconv(x)
        if x.shape[-1] == identity.shape[-1] and self.use_res:
            x = x + identity
        return [p, x, o]

