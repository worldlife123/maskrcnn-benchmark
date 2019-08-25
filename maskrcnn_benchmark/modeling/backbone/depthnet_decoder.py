# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn
from collections import OrderedDict

class DepthNetDecoder(nn.Module):
    """
    Module that adds FPN on top of a list of feature maps.
    The feature maps are currently supposed to be in increasing depth
    order, and must be consecutive
    """

    def __init__(
        self, in_channels_list, out_channels=1, decoder_channels_list=[32, 64, 128, 256], scales=range(4), use_skips=True, freeze_weight=False
    ):
        """
        Arguments:
            in_channels_list (list[int]): number of channels for each feature map that
                will be fed
            out_channels (int): number of channels of the FPN representation
        """

        super(DepthNetDecoder, self).__init__()

        self.out_channels = out_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = in_channels_list
        self.num_ch_dec = decoder_channels_list

        # decoder
        self.convs = OrderedDict()
        for i in range(len(self.num_ch_dec)-1, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == len(self.num_ch_dec)-1 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)
            self.add_module("upconv_{}_{}".format(i, 0), self.convs[("upconv", i, 0)])

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)
            self.add_module("upconv_{}_{}".format(i, 1), self.convs[("upconv", i, 1)])

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.out_channels)
            self.add_module("dispconv_{}".format(s), self.convs[("dispconv", s)])

        # self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        # self.outputs = {}

        # decoder
        outputs = []
        x = input_features[-1]
        for i in range(len(self.num_ch_dec)-1, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [F.interpolate(x, scale_factor=2, mode="nearest")] # upsample
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            # if i in self.scales:
                # self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))
            outputs.append(self.sigmoid(self.convs[("dispconv", i)](x)))

        return outputs # self.outputs

class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out

def build_depthnet_decoder(cfg):
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    depth_decoder = DepthNetDecoder(
        in_channels_list=[
            in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        freeze_weight=cfg.MODEL.DEPTHNET.FREEZE_WEIGHT,
    )
    return depth_decoder

