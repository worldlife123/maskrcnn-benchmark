# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from maskrcnn_benchmark.modeling import registry
from torch import nn

from maskrcnn_benchmark.modeling.utils import cat

@registry.ROI_DEPTH_PREDICTOR.register("FPNDepthPredictor")
class FPNDepthPredictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(FPNDepthPredictor, self).__init__()
        self.cfg = cfg

        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        representation_size = in_channels

        # self.cls_score = nn.Linear(representation_size, num_classes)
        num_bbox_reg_classes = 1 if cfg.MODEL.ROI_DEPTH_HEAD.SINGLE_DEPTH_REG else num_classes
        self.depth_pred = nn.Linear(representation_size, num_bbox_reg_classes)

        if cfg.MODEL.ROI_DEPTH_HEAD.SIGMOID_OUTPUT:
            self.sigmoid = nn.Sigmoid()

        # nn.init.normal_(self.cls_score.weight, std=0.01)
        # if cfg.MODEL.ROI_DEPTH_HEAD.SIGMOID_OUTPUT:
        #     nn.init.normal_(self.depth_pred.weight, std=0.01)
        #     nn.init.constant_(self.depth_pred.bias, 0.)
        # else:
        nn.init.normal_(self.depth_pred.weight, std=0.01)
        nn.init.constant_(self.depth_pred.bias, 0)
        # for l in [self.depth_pred]:
        #     nn.init.constant_(l.bias, 0)

    def forward(self, x):
        # if x.ndimension() == 4:
        #     assert list(x.shape[2:]) == [1, 1]
        #     x = x.view(x.size(0), -1)
        # scores = self.cls_score(x)
        depth = self.depth_pred(x)
        if self.cfg.MODEL.ROI_DEPTH_HEAD.SIGMOID_OUTPUT:
            depth = self.sigmoid(depth)

        return depth

# @registry.ROI_DEPTH_PREDICTOR.register("FPNDepthLRPredictor")
class FPNDepthLRPredictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(FPNDepthLRPredictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        representation_size = in_channels * 2

        # self.cls_score = nn.Linear(representation_size, num_classes)
        num_bbox_reg_classes = 1 if cfg.MODEL.ROI_DEPTH_HEAD.SINGLE_DEPTH_REG else num_classes
        self.depth_pred = nn.Linear(representation_size, num_bbox_reg_classes)

        # nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.depth_pred.weight, std=0.01)
        for l in [self.depth_pred]:
            nn.init.constant_(l.bias, 0)

    def forward(self, x):
        # if x.ndimension() == 4:
        #     assert list(x.shape[2:]) == [1, 1]
        #     x = x.view(x.size(0), -1)
        # scores = self.cls_score(x)
        x = cat(x, dim=-1)
        depth = self.depth_pred(x)

        return depth

@registry.ROI_DEPTH_PREDICTOR.register("FPNBox3dPredictor")
class FPNBox3dPredictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(FPNBox3dPredictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        representation_size = in_channels

        # self.cls_score = nn.Linear(representation_size, num_classes)
        # depth(1) + dimension(3) + orientation(8)
        num_bbox_reg_classes = (1 if cfg.MODEL.ROI_DEPTH_HEAD.SINGLE_DEPTH_REG else num_classes)
        self.depth_pred = nn.Linear(representation_size, num_bbox_reg_classes*1)
        self.dim_pred = nn.Linear(representation_size, num_bbox_reg_classes*3)
        self.ori_pred = nn.Linear(representation_size, num_bbox_reg_classes*8)

        # nn.init.normal_(self.cls_score.weight, std=0.01)
        for l in [self.depth_pred, self.dim_pred, self.ori_pred]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)

    def forward(self, x):
        # if x.ndimension() == 4:
        #     assert list(x.shape[2:]) == [1, 1]
        #     x = x.view(x.size(0), -1)
        # scores = self.cls_score(x)
        depth = self.depth_pred(x)
        dim = self.dim_pred(x)
        ori = self.ori_pred(x)

        return depth, dim, ori

def make_roi_depth_predictor(cfg, in_channels):
    func = registry.ROI_DEPTH_PREDICTOR[cfg.MODEL.ROI_DEPTH_HEAD.PREDICTOR]
    return func(cfg, in_channels)

def make_roi_depth_lr_predictor(cfg, in_channels):
    # func = registry.ROI_DEPTH_PREDICTOR[cfg.MODEL.ROI_DEPTH_HEAD.LR_PREDICTOR]
    func = FPNDepthLRPredictor
    return func(cfg, in_channels)
