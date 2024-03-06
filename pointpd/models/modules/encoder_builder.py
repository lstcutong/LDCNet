import torch
import torch.nn as nn
from ..builder import MODELS
from .block_helper import *
from pointpd.models.modules.featuring.ldcnet import *

@MODELS.register_module()
class BasicEncoder(nn.Module):
    def __init__(self,
                    in_channels,
                    enc_blocks,
                    samplers, poolings,
                    channels=[32, 64, 128, 256, 512],
                    down=[True, True, True, True]
                ):
        super().__init__()
        self.in_channels = in_channels


        assert len(enc_blocks) == len(channels)
        assert len(samplers) == len(channels) - 1
        assert len(poolings) == len(channels) - 1
        assert len(down) == len(channels) - 1

        # first layer doesn't downsample
        down = [False] + down
        samplers = [None] + samplers
        poolings = [None] + poolings

        self.enc = nn.ModuleList()
        for i in range(len(channels)):
            self.enc.append(
                self._make_enc(enc_blocks[i], channels[i], samplers[i], poolings[i], down[i])
            )

       
    def _make_enc(self, block, channel, sampler, pooling, down):
        if down:
            layers = [
                TransitionDown(self.in_channels, channel, sampler, pooling)
            ]
        else:
            layers = [
                UnaryBlockPXO(self.in_channels, channel)
            ]
        self.in_channels = channel
        for _ in range(block["repeats"]):
            layers.append(
                MODELS.build(block)
            )
        return nn.Sequential(*layers)


    def forward(self, input_dict, **kwargs):
        p0 = input_dict["coord"]
        x0 = input_dict["feat"]
        o0 = input_dict["offset"].int()

        position_set, feature_set, offset_set = [p0], [x0], [o0]
        for i in range(len(self.enc)):
            p, x, o = self.enc[i]([position_set[-1], feature_set[-1], offset_set[-1]])

            position_set.append(p)
            feature_set.append(x)
            offset_set.append(o)

        output_dict = {
            "point_set_positions": position_set[1:],
            "point_set_features": feature_set[1:],
            "point_set_offsets": offset_set[1:]
        }

        output_dict.update(input_dict)

        return output_dict


@MODELS.register_module()
class LDCEncoder(nn.Module):
    def __init__(self,
                    in_channels,
                    enc_blocks,
                    samplers, poolings,
                    g_blocknumber,
                    channels=[32, 64, 128, 256, 512],
                    down=[True, True, True, True],
                    num_keypoints=256,
                    spatial_scales=[8, 16, 32],
                    rgb_importance=1):
        super().__init__()
        self.c = in_channels
        self.spatial_scales = spatial_scales
        self.sampling = "furthest"
        self.rgb_importance = rgb_importance
        self.g_blocknumber = g_blocknumber

        self.mssca = MSSCA(self.c, 8, self.spatial_scales)
        self.in_channels = (len(self.spatial_scales) + 1) * 8
        self.num_keypoints = num_keypoints

        assert len(enc_blocks) == len(channels)
        assert len(g_blocknumber) == len(channels)
        assert len(samplers) == len(channels) - 1
        assert len(poolings) == len(channels) - 1
        assert len(down) == len(channels) - 1

        # first layer doesn't downsample
        down = [False] + down
        samplers = [None] + samplers
        poolings = [None] + poolings

        self.enc = nn.ModuleList()
        for i in range(len(channels)):
            self.enc.append(
                self._make_enc(enc_blocks[i], channels[i], samplers[i], poolings[i], down[i])
            )
        

        self.ldc_enc = nn.ModuleList()
        g_channels = [channels[0]] + channels
        for i in range(0, len(channels)):
            self.ldc_enc.append(
                self._make_ldc_enc_(g_channels[i], g_channels[i+1], self.g_blocknumber[i])
            )

        self.ldc_aug = nn.ModuleList()
        for i in range(1, len(channels)):
            self.ldc_aug.append(
                LDCAugmentation(channels[i], channels[i])
            )

       
    def _make_enc(self, block, channel, sampler, pooling, down):
        if down:
            layers = [
                TransitionDown(self.in_channels, channel, sampler, pooling)
            ]
        else:
            layers = [
                UnaryBlockPXO(self.in_channels, channel)
            ]
        self.in_channels = channel
        for _ in range(block["repeats"]):
            layers.append(
                MODELS.build(block)
            )
        return nn.Sequential(*layers)

    def _make_ldc_enc_(self, in_channels, out_channels, g_blocknumber):
        layers = [
            UnaryBlockPXO(in_channels, out_channels)
        ]
        for _ in range(g_blocknumber):
            layers.append(
                LDCReasoning(out_channels, out_channels, 8)
            )
        return nn.Sequential(*layers)


    def forward(self, input_dict, **kwargs):
        p0 = input_dict["coord"]
        x0 = input_dict["feat"]
        o0 = input_dict["offset"].int()
        x0[:, 3:] = self.rgb_importance * x0[:, 3:]

        p0, x0, o0 = self.mssca([p0, x0, o0])
        gp0, gx0, go0 = get_keypoints_furthest([p0, x0, o0], self.num_keypoints)


        position_set, feature_set, offset_set = [p0], [x0], [o0]
        g_position_set, g_feature_set, g_offset_set = [gp0], [gx0], [go0]

        for i in range(len(self.enc)):
            p, x, o = self.enc[i]([position_set[-1], feature_set[-1], offset_set[-1]])
            gp, gx, go = self.ldc_enc[i]([g_position_set[-1], g_feature_set[-1], g_offset_set[-1]])
            p, x, o = self.ldc_aug[i]([p, x, o], [gp, gx, go])

            position_set.append(p)
            feature_set.append(x)
            offset_set.append(o)

            g_position_set.append(gp)
            g_feature_set.append(gx)
            g_offset_set.append(go)

        output_dict = {
            "point_set_positions": position_set[1:],
            "point_set_features": feature_set[1:],
            "point_set_offsets": offset_set[1:]
        }

        output_dict.update(input_dict)

        return output_dict