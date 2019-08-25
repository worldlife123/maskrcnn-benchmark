# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# import numpy as np
import math
import torch
from torch import nn
from maskrcnn_benchmark.layers.misc import interpolate

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.modeling.box_coder import BoxCoder

def get_alpha(rot):
    # output: (B, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos, 
    #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
    # return rot[:, 0]
    idx = (rot[:, 1] > rot[:, 5]).float()
    alpha1 = torch.atan(rot[:, 2] / rot[:, 3]) + (-0.5 * math.pi)
    alpha2 = torch.atan(rot[:, 6] / rot[:, 7]) + ( 0.5 * math.pi)
    return alpha1 * idx + alpha2 * (1 - idx)

class Box3dPostProcessor(nn.Module):
    """
    From the results of the CNN, post process the depth
    by taking the depth corresponding to the class with max
    probability (which are of fixed size and directly output
    by the CNN).
    """

    def __init__(self, cfg, box_coder):
        super(Box3dPostProcessor, self).__init__()
        self.cfg = cfg.clone()
        self.box_coder = box_coder
        # self.masker = masker

    def forward(self, x, boxes):
        """
        Arguments:
            x (Tensor): the mask logits
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra field mask
        """

        # select depth coresponding to the predicted classes
        centers, depths, dims, rots = x
        num_batches = depths.shape[0]
        labels = [bbox.get_field("labels") for bbox in boxes]
        labels = torch.cat(labels)
        index = torch.arange(num_batches, device=labels.device)
        depths = depths[index, labels][:, None]
        centers = centers.view(-1, centers.shape[1]//2, 2)[index, labels]
        dims = dims.view(-1, dims.shape[1]//3, 3)[index, labels]#[:, None]
        rots = rots.view(-1, rots.shape[1]//8, 8)[index, labels]#[:, None]
        boxes_per_image = [len(box) for box in boxes]
        depths = depths.split(boxes_per_image, dim=0)
        dims = dims.split(boxes_per_image, dim=0)
        rots = rots.split(boxes_per_image, dim=0)
        results = []
        for center, depth, dim, rot, box in zip(centers, depths, dims, rots, boxes):
            bbox = BoxList(box.bbox, box.size, mode="xyxy")
            for field in box.fields():
                bbox.add_field(field, box.get_field(field))
            # decode centers
            center = bbox.bbox[:, 0:2] + center * (bbox.bbox[:, 2:4]-bbox.bbox[:, 0:2])
            center_box = BoxList(torch.cat((center,center), dim=1), box.size, mode="xyxy")
            if self.cfg.MODEL.ROI_BOX3D_HEAD.REG_LOGARITHM:
                depth = torch.exp(depth)
            depth = depth/self.cfg.MODEL.ROI_BOX3D_HEAD.REG_AMPLIFIER
            bbox.add_field("centers", center_box)
            bbox.add_field("depths", depth)
            bbox.add_field("dims", dim)
            bbox.add_field("rots", rot)
            bbox.add_field("alphas", get_alpha(rot))
            results.append(bbox)

        return results

def make_roi_box3d_post_processor(cfg):

    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)
    
    depth_post_processor = Box3dPostProcessor(cfg, box_coder)
    return depth_post_processor
