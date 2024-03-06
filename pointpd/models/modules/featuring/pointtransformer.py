"""
Point Transformer V1 for Semantic Segmentation

Might be a bit different from the original paper

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import torch
import torch.nn as nn
import einops
import pointops
from pointpd.models.builder import MODELS

torch.nn.LayerNorm


class LayerNorm1d(nn.BatchNorm1d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return (
            super()
            .forward(input.transpose(1, 2).contiguous())
            .transpose(1, 2)
            .contiguous()
        )


class PointTransformerLayer(nn.Module):
    def __init__(self, in_planes, out_planes, grouper, share_planes=8):
        super().__init__()
        self.mid_planes = mid_planes = out_planes // 1
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.linear_q = nn.Linear(in_planes, mid_planes)
        self.linear_k = nn.Linear(in_planes, mid_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)
        self.linear_p = nn.Sequential(
            nn.Linear(3, 3),
            LayerNorm1d(3),
            nn.ReLU(inplace=True),
            nn.Linear(3, out_planes),
        )
        self.linear_w = nn.Sequential(
            LayerNorm1d(mid_planes),
            nn.ReLU(inplace=True),
            nn.Linear(mid_planes, out_planes // share_planes),
            LayerNorm1d(out_planes // share_planes),
            nn.ReLU(inplace=True),
            nn.Linear(out_planes // share_planes, out_planes // share_planes),
        )
        self.softmax = nn.Softmax(dim=1)

        self.grouper = MODELS.build(grouper)

    def forward(self, pxo) -> torch.Tensor:
        p, x, o = pxo  # (n, 3), (n, c), (b)
        x_q, x_k, x_v = self.linear_q(x), self.linear_k(x), self.linear_v(x)
        x_k, idx = self.grouper(
            dict(
                source_p = p,
                target_p = p,
                source_f = x_k,
                source_o = o,
                target_o = o,
                with_xyz = True,
            )
        )
        x_v, _ = self.grouper(
            dict(
                source_p = p,
                target_p = p,
                source_f = x_v,
                source_o = o,
                target_o = o,
                idx = idx,
                with_xyz = False,
            )
        )
        p_r, x_k = x_k[:, :, 0:3], x_k[:, :, 3:]
        p_r = self.linear_p(p_r)
        r_qk = (
            x_k
            - x_q.unsqueeze(1)
            + einops.reduce(
                p_r, "n ns (i j) -> n ns j", reduction="sum", j=self.mid_planes
            )
        )
        w = self.linear_w(r_qk)  # (n, nsample, c)
        w = self.softmax(w)
        x = torch.einsum(
            "n t s i, n t i -> n s i",
            einops.rearrange(x_v + p_r, "n ns (s i) -> n ns s i", s=self.share_planes),
            w,
        )
        x = einops.rearrange(x, "n s i -> n (s i)")
        return x


@MODELS.register_module()
class PointTransformerBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, grouper, share_planes=8, **kwargs):
        super(PointTransformerBlock, self).__init__()
        self.linear1 = nn.Linear(in_channels, in_channels, bias=False)
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.transformer = PointTransformerLayer(in_channels, in_channels, grouper, share_planes)
        self.bn2 = nn.BatchNorm1d(in_channels)
        self.linear3 = nn.Linear(in_channels, in_channels * self.expansion, bias=False)
        self.bn3 = nn.BatchNorm1d(in_channels * self.expansion)
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


