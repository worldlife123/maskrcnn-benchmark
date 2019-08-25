# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from maskrcnn_benchmark.modeling import registry
from torch import nn

from maskrcnn_benchmark.modeling.utils import cat


@registry.ROI_BOX3D_PREDICTOR.register("FPNBox3dPredictor")
class FPNBox3dPredictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(FPNBox3dPredictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        representation_size = in_channels

        # self.cls_score = nn.Linear(representation_size, num_classes)
        # center(2) + depth(1) + dimension(3) + orientation(8)
        num_bbox_reg_classes = (1 if cfg.MODEL.ROI_BOX3D_HEAD.SINGLE_BOX3D_REG else num_classes)
        self.center_pred = nn.Linear(representation_size, num_bbox_reg_classes*2)
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
        center = self.center_pred(x)
        depth = self.depth_pred(x)
        dim = self.dim_pred(x)
        ori = self.ori_pred(x)

        return center, depth, dim, ori

def make_roi_box3d_predictor(cfg, in_channels):
    func = registry.ROI_BOX3D_PREDICTOR[cfg.MODEL.ROI_BOX3D_HEAD.PREDICTOR]
    return func(cfg, in_channels)
