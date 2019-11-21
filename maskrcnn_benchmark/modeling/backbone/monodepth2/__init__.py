from .depth_decoder import DepthDecoder
from .resnet_encoder import ResnetEncoder
import torch
import torch.nn as nn

def build_monodepth2(cfg):
    depthnet_encoder = ResnetEncoder(18, False)
    depthnet_decoder = DepthDecoder(depthnet_encoder.num_ch_enc)
    enc_weights = torch.load(cfg.MODEL.DEPTHNET.MONODEPTH2.ENCODER.PRETRAINED_WEIGHT)
    dec_weights = torch.load(cfg.MODEL.DEPTHNET.MONODEPTH2.DECODER.PRETRAINED_WEIGHT)

    model_dict = depthnet_encoder.state_dict()
    depthnet_encoder.load_state_dict({k: v for k, v in enc_weights.items() if k in model_dict})
    depthnet_decoder.load_state_dict(dec_weights)

    if cfg.MODEL.DEPTHNET.MONODEPTH2.ENCODER.FREEZE_WEIGHT:
        for p in depthnet_encoder.parameters():
            p.requires_grad = False
    if cfg.MODEL.DEPTHNET.MONODEPTH2.DECODER.FREEZE_WEIGHT:
        for p in depthnet_decoder.parameters():
            p.requires_grad = False

    return nn.Sequential(depthnet_encoder, depthnet_decoder)