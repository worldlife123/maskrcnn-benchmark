# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .generalized_rcnn import GeneralizedRCNN
from .generalized_lr_rcnn import GeneralizedLRRCNN
from .generalized_lrgan_rcnn import GeneralizedLRGANRCNN
from .generalized_rcnn_lrmt import GeneralizedRCNNLRMT

_DETECTION_META_ARCHITECTURES = {
    "GeneralizedRCNN": GeneralizedRCNN,
    "GeneralizedLRRCNN": GeneralizedLRRCNN,
    "GeneralizedLRGANRCNN": GeneralizedLRGANRCNN,
    "GeneralizedRCNNLRMT": GeneralizedRCNNLRMT,
}


def build_detection_model(cfg):
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)
