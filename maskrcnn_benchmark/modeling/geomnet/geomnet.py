import torch
from torch import nn
from torch.nn import functional as F

from .layers import *

class SpatialTransformerGeomNet(nn.Module):
    def __init__(self, output_size, interp_mode="bilinear"):
        super(SpatialTransformerGeomNet, self).__init__()

        self.ipm = InversePerspectiveMapping(output_size, interp_mode=interp_mode)

    def forward(self, features, K, T):
        transformed_features = self.ipm(features, K, T)

        return transformed_features


_DISCRIMINATOR_META_ARCHITECTURES = {
    "SpatialTransformerGeomNet": SpatialTransformerGeomNet,
}

def build_generator(cfg):
    meta_arch = _DISCRIMINATOR_META_ARCHITECTURES[cfg.MODEL.GAN.GENERATOR_ARCHITECTURE]
    return meta_arch(cfg)