
import torch
import torch.nn as nn
from typing import List, Type
from ..builder import MODELS
from pointpd.models.modules.block_helper import TransitionUp, GlobalPooling, UnaryBlockX

@MODELS.register_module()
class BaseSegDecoder(nn.Module):
    def __init__(self, 
            dec_blocks, 
            channels,
            down=[True, True, True, True],
            global_pool=True,
            **kwargs):
        super().__init__()
        channels = channels[::-1]
        assert len(dec_blocks) == len(channels) -1 or len(dec_blocks) == len(channels)
        if len(dec_blocks) == len(channels) - 1:
            dec_blocks.append(None)

        self.in_channels = channels[0]
        self.global_pool = GlobalPooling(self.in_channels) if global_pool else None

        self.up = down[::-1]
        channels = channels[1:]
        
        self.dec = nn.ModuleList()
        for i in range(0, len(channels)):
            self.dec.append(
                self._make_dec(
                    dec_blocks[i], channels[i], self.up[i]
                )
            )
        
        self.last_block = None
        if dec_blocks[-1] is not None:
            self.last_block = []
            for _ in range(dec_blocks[-1]["repeats"]):
                self.last_block.append(
                    MODELS.build(dec_blocks[-1])
                )
            self.last_block = nn.Sequential(*self.last_block)
        
    def _make_dec(
        self, block, channel, up
    ):
        if up:
            layers = [
                TransitionUp(self.in_channels, channel)
            ]
        else:
            layers = [
                UnaryBlockX(self.in_channels, channel)
            ]

        self.in_channels = channel
        if block is not None:
            for _ in range(block["repeats"]):
                layers.append(
                    MODELS.build(block)
                )
        return nn.Sequential(*layers)

    def forward(self, input_dict):
        p = input_dict["point_set_positions"][::-1]
        x = input_dict["point_set_features"][::-1]
        o = input_dict["point_set_offsets"][::-1]

        if self.global_pool is not None:
            x[0] = self.global_pool([p[0], x[0], o[0]])
        
        for i in range(0, len(p) - 1):
            if len(self.dec[i]) > 1:
                x[i] = self.dec[i][1:]([p[i], x[i], o[i]])[1]
            if self.up[i]:
                x[i + 1] = self.dec[i][0]([p[i+1], x[i+1], o[i+1]], [p[i], x[i], o[i]])
            else:
                x[i + 1] = self.dec[i][0](x[i])

        if self.last_block is not None:
            x[-1] = self.last_block([p[-1], x[-1], o[-1]])[1]

        output_dict = {
            "feature": x[-1]
        }
        output_dict.update(input_dict)
        return output_dict