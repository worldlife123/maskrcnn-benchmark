# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F

from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.make_layers import conv_with_kaiming_uniform, conv_with_zero_weight
from . import fpn as fpn_module
from . import depthnet_decoder as depth_module
from . import resnet
from . import monodepth2 as monodepth2_module


@registry.BACKBONES.register("R-50-C4")
@registry.BACKBONES.register("R-50-C5")
@registry.BACKBONES.register("R-101-C4")
@registry.BACKBONES.register("R-101-C5")
def build_resnet_backbone(cfg):
    body = resnet.ResNet(cfg)
    model = nn.Sequential(OrderedDict([("body", body)]))
    model.out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    return model


@registry.BACKBONES.register("R-50-FPN")
@registry.BACKBONES.register("R-101-FPN")
@registry.BACKBONES.register("R-152-FPN")
def build_resnet_fpn_backbone(cfg):
    body = resnet.ResNet(cfg)
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    fpn = fpn_module.FPN(
        in_channels_list=[
            in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelMaxPool(),
        freeze_weight=cfg.MODEL.FPN.FREEZE_WEIGHT,
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model


@registry.BACKBONES.register("R-50-FPN-RETINANET")
@registry.BACKBONES.register("R-101-FPN-RETINANET")
def build_resnet_fpn_p3p7_backbone(cfg):
    body = resnet.ResNet(cfg)
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    in_channels_p6p7 = in_channels_stage2 * 8 if cfg.MODEL.RETINANET.USE_C5 \
        else out_channels
    fpn = fpn_module.FPN(
        in_channels_list=[
            0,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelP6P7(in_channels_p6p7, out_channels),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model

@registry.BACKBONES.register("R-50-DEPTHNET")
@registry.BACKBONES.register("R-101-DEPTHNET")
@registry.BACKBONES.register("R-152-DEPTHNET")
def build_resnet_depthnet_backbone(cfg):
    model = ResnetDepthNet(cfg)
    return model

class ResnetDepthNet(nn.Module):
    def __init__(self, cfg):
        super(ResnetDepthNet, self).__init__()

        self.body = resnet.ResNet(cfg)
        in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
        self.out_channels = cfg.MODEL.DEPTHNET.DECODER_OUTPUT_CHANNELS # cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
        
        self.depth_decoder = depth_module.DepthNetDecoder(
            in_channels_list=[
                in_channels_stage2,
                in_channels_stage2 * 2,
                in_channels_stage2 * 4,
                in_channels_stage2 * 8,
            ],
            decoder_channels_list=[
                self.out_channels,
                self.out_channels,
                self.out_channels,
                self.out_channels,
            ],
            out_channels=2, # estimate both left and right disparity
            freeze_weight=cfg.MODEL.DEPTHNET.FREEZE_WEIGHT,
        )

    def forward(self, x):
        x = self.body(x)
        depth_out = self.depth_decoder(x)
        return {
            "decoder_out": depth_out["decoder_output"],
            "depth_out": depth_out["disp_output"],
        }

@registry.BACKBONES.register("R-50-DEPTHNET-FPN")
@registry.BACKBONES.register("R-101-DEPTHNET-FPN")
@registry.BACKBONES.register("R-152-DEPTHNET-FPN")
def build_resnet_depthnet_fpn_backbone(cfg):
    model = ResnetDepthNetFPN(cfg)
    return model

class ResnetDepthNetFPN(nn.Module):
    def __init__(self, cfg):
        super(ResnetDepthNetFPN, self).__init__()

        self.body = resnet.ResNet(cfg)
        in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
        self.out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
        self.fpn = fpn_module.FPN(
            in_channels_list=[
                in_channels_stage2,
                in_channels_stage2 * 2,
                in_channels_stage2 * 4,
                in_channels_stage2 * 8,
            ],
            out_channels=self.out_channels,
            conv_block=conv_with_kaiming_uniform(
                cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
            ),
            top_blocks=fpn_module.LastLevelMaxPool(),
            freeze_weight=cfg.MODEL.FPN.FREEZE_WEIGHT,
        )
        self.depth_decoder = depth_module.DepthNetFPNDecoder(
            in_channels_list=[
                self.out_channels,
                self.out_channels,
                self.out_channels,
                self.out_channels,
            ],
            out_channels=2, # estimate both left and right disparity
            freeze_weight=cfg.MODEL.DEPTHNET.FREEZE_WEIGHT,
            coarse_to_fine=cfg.MODEL.DEPTHNET.DECODER_COARSE_TO_FINE
        )

    def forward(self, x):
        x = self.body(x)
        fpn_out = self.fpn(x)
        depth_out = self.depth_decoder(fpn_out[0:4])
        return {
            "fpn_out": fpn_out,
            "depth_out": depth_out["disp_output"],
        }

@registry.BACKBONES.register("R-50-FPN-DEPTHNET-FPN")
@registry.BACKBONES.register("R-101-FPN-DEPTHNET-FPN")
@registry.BACKBONES.register("R-152-FPN-DEPTHNET-FPN")
def build_resnet_fpn_depthnet_fpn_backbone(cfg):
    model = ResnetFPNDepthNetFPN(cfg)
    return model

class ResnetFPNDepthNetFPN(nn.Module):
    def __init__(self, cfg):
        super(ResnetFPNDepthNetFPN, self).__init__()

        self.body = resnet.ResNet(cfg)
        in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
        self.out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
        self.fpn = fpn_module.FPN(
            in_channels_list=[
                in_channels_stage2,
                in_channels_stage2 * 2,
                in_channels_stage2 * 4,
                in_channels_stage2 * 8,
            ],
            out_channels=self.out_channels,
            conv_block=conv_with_kaiming_uniform(
                cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
            ),
            top_blocks=fpn_module.LastLevelMaxPool(),
            freeze_weight=cfg.MODEL.FPN.FREEZE_WEIGHT,
        )
        self.depth_decoder = depth_module.DepthNetFPNDecoder(
            in_channels_list=[
                self.out_channels,
                self.out_channels,
                self.out_channels,
                self.out_channels,
            ],
            out_channels=2, # estimate both left and right disparity
            freeze_weight=cfg.MODEL.DEPTHNET.FREEZE_WEIGHT,
            coarse_to_fine=cfg.MODEL.DEPTHNET.DECODER_COARSE_TO_FINE
        )
        self.fpn2 = fpn_module.FPN(
            in_channels_list=[
                2,
                2,
                2,
                2,
            ],
            out_channels=self.out_channels,
            conv_block=conv_with_kaiming_uniform(
                cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
            ),
            # top_blocks=fpn_module.LastLevelMaxPool(),
            freeze_weight=cfg.MODEL.FPN.FREEZE_WEIGHT,
        )

    def forward(self, x):
        x = self.body(x)
        fpn_out = self.fpn(x)
        depth_out = self.depth_decoder(fpn_out[0:4])
        fpn2_out = self.fpn2(depth_out["disp_output"])
        # return fpn_out+fpn2_out, depth_out["disp_output"]
        # add fpn2 to fpn
        # fpn_out = tuple([(f1+f2) for f1,f2 in zip(fpn_out[0:4], fpn2_out)])
        return {
            "fpn_out": fpn_out[0:4],
            "depth_out": depth_out["disp_output"],
            "fpn2_out": fpn2_out
        }

@registry.BACKBONES.register("R-50-FPN-WITH-MONODEPTH2")
@registry.BACKBONES.register("R-101-FPN-WITH-MONODEPTH2")
@registry.BACKBONES.register("R-152-FPN-WITH-MONODEPTH2")
def build_resnet_fpn_with_monodepth2_backbone(cfg):
    model = ResnetFPNWithMonodepth2(cfg)
    return model

class ResnetFPNWithMonodepth2(nn.Module):
    def __init__(self, cfg):
        super(ResnetFPNWithMonodepth2, self).__init__()

        self.body = resnet.ResNet(cfg)
        in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
        self.out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
        self.fpn = fpn_module.FPN(
            in_channels_list=[
                in_channels_stage2,
                in_channels_stage2 * 2,
                in_channels_stage2 * 4,
                in_channels_stage2 * 8,
            ],
            out_channels=self.out_channels,
            conv_block=conv_with_kaiming_uniform(
                cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
            ),
            top_blocks=fpn_module.LastLevelMaxPool(),
            freeze_weight=cfg.MODEL.FPN.FREEZE_WEIGHT,
        )
        self.monodepth_encoder = monodepth2_module.ResnetEncoder(18, False)
        self.monodepth_decoder = monodepth2_module.DepthDecoder(self.monodepth_encoder.num_ch_enc)
        self.fpn2 = fpn_module.FPN(
            in_channels_list= self.monodepth_decoder.num_ch_dec[1:5], #self.monodepth_encoder.num_ch_enc,
            out_channels=self.out_channels,
            conv_block=conv_with_kaiming_uniform(
                cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
            ),
            # top_blocks=fpn_module.LastLevelMaxPool(),
            freeze_weight=cfg.MODEL.FPN.FREEZE_WEIGHT,
        )

        enc_weights = torch.load(cfg.MODEL.DEPTHNET.MONODEPTH2.ENCODER.PRETRAINED_WEIGHT)
        dec_weights = torch.load(cfg.MODEL.DEPTHNET.MONODEPTH2.DECODER.PRETRAINED_WEIGHT)

        model_dict = self.monodepth_encoder.state_dict()
        self.monodepth_encoder.load_state_dict({k: v for k, v in enc_weights.items() if k in model_dict})
        self.monodepth_decoder.load_state_dict(dec_weights)

        if cfg.MODEL.DEPTHNET.MONODEPTH2.ENCODER.FREEZE_WEIGHT:
            for p in self.monodepth_encoder.parameters():
                p.requires_grad = False
        if cfg.MODEL.DEPTHNET.MONODEPTH2.DECODER.FREEZE_WEIGHT:
            for p in self.monodepth_decoder.parameters():
                p.requires_grad = False

    def forward(self, x):
        # resnet_out = self.body(x)
        fpn_out = self.fpn(self.body(x))
        # x = self.monodepth_encoder(x)
        depth_out = self.monodepth_decoder(self.monodepth_encoder(x))
        # fpn2_out = self.fpn2(encoder_out)
        disp_feat = [F.interpolate(depth_out[("dispfeat", i)], size=(fpn_out[i-1].shape[2], fpn_out[i-1].shape[3]), mode="nearest") for i in range(1,5)]
        fpn2_out = self.fpn2(disp_feat)
        # return fpn_out+fpn2_out, depth_out["disp_output"]
        # add fpn2 to fpn
        # fpn_out = tuple([(f1+f2) for f1,f2 in zip(fpn_out[0:4], fpn2_out)])
        return {
            "fpn_out": fpn_out[0:4],
            # "depth_out": [depth_out[("disp", i)] for i in range(4)],
            "fpn2_out": fpn2_out
        }


def build_backbone(cfg):
    assert cfg.MODEL.BACKBONE.CONV_BODY in registry.BACKBONES, \
        "cfg.MODEL.BACKBONE.CONV_BODY: {} are not registered in registry".format(
            cfg.MODEL.BACKBONE.CONV_BODY
        )
    return registry.BACKBONES[cfg.MODEL.BACKBONE.CONV_BODY](cfg)
