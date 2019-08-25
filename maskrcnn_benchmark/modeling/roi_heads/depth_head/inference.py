# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np
import torch
from torch import nn
from maskrcnn_benchmark.layers.misc import interpolate

from maskrcnn_benchmark.structures.bounding_box import BoxList

def get_alpha(rot):
    # output: (B, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos, 
    #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
    # return rot[:, 0]
    idx = (rot[:, 1] > rot[:, 5]).float()
    alpha1 = torch.atan(rot[:, 2] / rot[:, 3]) + (-0.5 * np.pi)
    alpha2 = torch.atan(rot[:, 6] / rot[:, 7]) + ( 0.5 * np.pi)
    return alpha1 * idx + alpha2 * (1 - idx)

# TODO check if want to return a single BoxList or a composite
# object
class DepthPostProcessor(nn.Module):
    """
    From the results of the CNN, post process the depth
    by taking the depth corresponding to the class with max
    probability (which are of fixed size and directly output
    by the CNN).
    """

    def __init__(self, cfg):
        super(DepthPostProcessor, self).__init__()
        self.cfg = cfg.clone()
        # self.masker = masker

    def forward(self, x, boxes):
        """
        Arguments:
            x (Tensor): the depth logits
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra field mask
        """

        # select depth coresponding to the predicted classes
        num_batches = x.shape[0]
        labels = [bbox.get_field("labels") for bbox in boxes]
        labels = torch.cat(labels)
        index = torch.arange(num_batches, device=labels.device)
        # print(x.shape, labels.shape)
        x = x[index, labels][:, None]
        # print(x.shape)
        boxes_per_image = [len(box) for box in boxes]
        x = x.split(boxes_per_image, dim=0)
        results = []
        for depth, box in zip(x, boxes):
            bbox = BoxList(box.bbox, box.size, mode="xyxy")
            for field in box.fields():
                bbox.add_field(field, box.get_field(field))
            if self.cfg.MODEL.ROI_DEPTH_HEAD.REG_LOGARITHM:
                depth = torch.exp(depth)
            depth = depth/self.cfg.MODEL.ROI_DEPTH_HEAD.REG_AMPLIFIER
            bbox.add_field("depths", depth)
            # print(depth)
            results.append(bbox)

        return results

class Box3dPostProcessor(nn.Module):
    """
    From the results of the CNN, post process the depth
    by taking the depth corresponding to the class with max
    probability (which are of fixed size and directly output
    by the CNN).
    """

    def __init__(self, cfg):
        super(Box3dPostProcessor, self).__init__()
        self.cfg = cfg.clone()
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
        depths, dims, rots = x
        num_batches = depths.shape[0]
        labels = [bbox.get_field("labels") for bbox in boxes]
        labels = torch.cat(labels)
        index = torch.arange(num_batches, device=labels.device)
        depths = depths[index, labels][:, None]
        dims = dims.view(-1, dims.shape[1]//3, 3)[index, labels]#[:, None]
        rots = rots.view(-1, rots.shape[1]//8, 8)[index, labels]#[:, None]
        boxes_per_image = [len(box) for box in boxes]
        depths = depths.split(boxes_per_image, dim=0)
        dims = dims.split(boxes_per_image, dim=0)
        rots = rots.split(boxes_per_image, dim=0)
        results = []
        for depth, dim, rot, box in zip(depths, dims, rots, boxes):
            bbox = BoxList(box.bbox, box.size, mode="xyxy")
            for field in box.fields():
                bbox.add_field(field, box.get_field(field))
            if self.cfg.MODEL.ROI_DEPTH_HEAD.REG_LOGARITHM:
                depth = torch.exp(depth)
            depth = depth/self.cfg.MODEL.ROI_DEPTH_HEAD.REG_AMPLIFIER
            bbox.add_field("depths", depth)
            bbox.add_field("dims", dim)
            bbox.add_field("rots", rot)
            bbox.add_field("alphas", get_alpha(rot))
            results.append(bbox)

        return results

def make_roi_depth_post_processor(cfg):

    if cfg.MODEL.ROI_DEPTH_HEAD.PREDICTOR == "FPNBox3dPredictor":
        depth_post_processor = Box3dPostProcessor(cfg)
    else:
        depth_post_processor = DepthPostProcessor(cfg)
    return depth_post_processor
