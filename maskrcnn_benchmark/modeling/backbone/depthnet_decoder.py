# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn
from collections import OrderedDict

from .resnet import BottleneckWithFixedBatchNorm

class DepthNetFPNDecoder(nn.Module):
    """
    Module that adds FPN on top of a list of feature maps.
    The feature maps are currently supposed to be in increasing depth
    order, and must be consecutive
    """

    def __init__(
        self, in_channels_list, out_channels=1, freeze_weight=False, coarse_to_fine=False,
    ):
        """
        Arguments:
            in_channels_list (list[int]): number of channels for each feature map that
                will be fed
            out_channels (int): number of channels of the FPN representation
        """

        super(DepthNetFPNDecoder, self).__init__()

        self.out_channels = out_channels
        self.upsample_mode = 'nearest'
        self.coarse_to_fine = coarse_to_fine

        self.in_channels_list = in_channels_list

        # decoder
        self.convs = OrderedDict()

        for s in range(len(in_channels_list)):
            self.convs[("dispconv", s)] = Conv3x3(self.in_channels_list[s], self.out_channels)
            # self.convs[("dispconv", s)] = BottleneckWithFixedBatchNorm(self.in_channels_list[s], self.in_channels_list[s] // 4, self.out_channels)
            # self.convs[("dispconv", s)] = BottleneckWithFixedBatchNorm(self.in_channels_list[s], self.in_channels_list[s] * 4, self.out_channels) # inverted bottleneck
            if freeze_weight:
                for p in self.convs[("dispconv", s)].parameters():
                    p.requires_grad = False
            # self.convs[("dispupconv", s)] = UpConv3x3(self.out_channels, self.in_channels_list[s])
            self.add_module("dispconv_{}".format(s), self.convs[("dispconv", s)])
            # self.add_module("dispupconv_{}".format(s), self.convs[("dispupconv", s)])
            if self.coarse_to_fine and s < len(in_channels_list)-1:
                self.convs[("refineconv", s)] = Conv3x3(self.out_channels, self.out_channels)
                # self.convs[("refineconv", s)] = BottleneckWithFixedBatchNorm(self.out_channels, self.in_channels_list[s], self.out_channels)
                if freeze_weight:
                    for p in self.convs[("refineconv", s)].parameters():
                        p.requires_grad = False
                self.add_module("refineconv_{}".format(s), self.convs[("refineconv", s)])


        # self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        # self.outputs = {}

        # decoder

        # x = input_features[-1]
        # for i in range(len(self.in_channels_list)-1, -1, -1):
        #     x = self.convs[("dispconv", i)](x)
        #     outputs["disp_output"].append(self.sigmoid(x))
        #     if i>0: 
        #         x = F.interpolate(x, scale_factor=2, mode="nearest") # upsample
        #         x += input_features[i-1]

        outputs = {"disp_output":[], "decoder_output":[]}
        # [print(f.shape) for f in input_features]
        # x = input_features[-1]
        prev_disp = None
        for i in range(len(self.in_channels_list)-1, -1, -1):
            x = input_features[i]
            disp = self.convs[("dispconv", i)](x)
            if self.coarse_to_fine and not prev_disp is None:
                prev_disp = F.interpolate(prev_disp, scale_factor=2, mode='nearest')
                disp = self.convs[("refineconv", i)](disp+prev_disp)
            prev_disp = disp
            outputs["disp_output"].append(self.sigmoid(disp))
            # outputs["decoder_output"].append(self.convs[("dispupconv", i)](x))

        # reverse so that the pyramid is upside-down just as the input
        outputs["disp_output"].reverse()
        # outputs["decoder_output"].reverse()
        return outputs # self.outputs


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

            if freeze_weight:
                for p in self.convs[("upconv", i, 0)].parameters():
                    p.requires_grad = False
                for p in self.convs[("upconv", i, 1)].parameters():
                    p.requires_grad = False

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.out_channels)
            self.add_module("dispconv_{}".format(s), self.convs[("dispconv", s)])
            for p in self.convs[("dispconv", s)].parameters():
                    p.requires_grad = False

        # self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        # self.outputs = {}

        # decoder
        outputs = {"disp_output":[], "decoder_output":[]}
        # [print(f.shape) for f in input_features]
        x = input_features[-1]
        for i in range(len(self.num_ch_dec)-1, -1, -1):
            out1 = self.convs[("upconv", i, 0)](x)
            x = [F.interpolate(out1, scale_factor=2, mode="nearest")] # upsample
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            # if i in self.scales:
                # self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))
            outputs["decoder_output"].append(out1)
            outputs["disp_output"].append(self.sigmoid(self.convs[("dispconv", i)](x)))

        # reverse so that the pyramid is upside-down just as the input
        outputs["decoder_output"].reverse()
        outputs["disp_output"].reverse()
        return outputs # self.outputs


class UpProjection(nn.Module):

    def __init__(self, num_input_features, num_output_features):
        super(UpProjection, self).__init__()

        self.conv1 = nn.Conv2d(num_input_features, num_output_features,
                               kernel_size=5, stride=1, padding=2, bias=False)      
        self.bn1 = nn.BatchNorm2d(num_output_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(num_output_features, num_output_features,          
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(num_output_features)

        self.conv2 = nn.Conv2d(num_input_features, num_output_features,             
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(num_output_features)

    def forward(self, x, size):
        x = F.interpolate(x, size=size, mode='bilinear',align_corners=True)
        x_conv1 = self.relu(self.bn1(self.conv1(x)))
        bran1 = self.bn1_2(self.conv1_2(x_conv1))
        bran2 = self.bn2(self.conv2(x))

        out = self.relu(bran1 + bran2)

        return out

class ADFF(nn.Module):
    """
    Adaptive Dense Features Fusion module
    """

    def __init__(
        self, in_channels_list, out_channels, freeze_weight=False
    ):
        """
        Arguments:
            in_channels_list (list[int]): number of channels for each feature map that
                will be fed
            out_channels (int): number of channels of the FPN representation
        """

        super(ADFF, self).__init__()

        self.out_channels = out_channels
        self.in_channels_list = in_channels_list


    def forward(self, input_features):
        pass # TODO


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
        self.normalize = nn.BatchNorm2d(int(out_channels))

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        out = self.normalize(x)
        return out

class UpConv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels):
        super(UpConv3x3, self).__init__()

        # if use_refl:
        #     self.pad = nn.ReflectionPad2d(1)
        # else:
        #     self.pad = nn.ZeroPad2d(1)
        self.conv = nn.ConvTranspose2d(int(in_channels), int(out_channels), 3, padding=1)
        self.normalize = nn.BatchNorm2d(int(out_channels))

    def forward(self, x):
        # x = self.pad(x)
        x = self.conv(x)
        out = self.normalize(x)
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

