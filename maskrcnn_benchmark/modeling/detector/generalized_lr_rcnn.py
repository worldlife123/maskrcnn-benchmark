# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone, build_depthnet_decoder, build_depthnet_loss
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads
from ..roi_heads.roi_lr_heads import build_roi_lr_heads


class GeneralizedLRRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedLRRCNN, self).__init__()

        self.cfg = cfg.clone()
        
        self.backbone = build_backbone(cfg)
        if cfg.MODEL.DEPTHNET_ON:
            # self.depthnet_decoder = build_depthnet_decoder(cfg)
            self.depthnet_loss = build_depthnet_loss(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)# build_roi_lr_heads(cfg, self.backbone.out_channels)

    def forward(self, images_left, images_right=None, targets_left=None, targets_right=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and (images_right is None or targets_left is None or targets_right is None):
            raise ValueError("In training mode, images_right and targets should be passed")
        images_left = to_image_list(images_left)
        features_left = self.backbone(images_left.tensors)
        if self.cfg.MODEL.DEPTHNET_ON:
            features_left, disp_left = features_left
            # disp_output_left = to_image_list(self.depthnet_decoder(features_left))
            disp_output_left = disp_left # to_image_list(disp_left)

        if self.training:
            images_right = to_image_list(images_right)
            features_right = self.backbone(images_right.tensors)
            if self.cfg.MODEL.DEPTHNET_ON:
                features_right, disp_right = features_right
                # disp_output_right = to_image_list(self.depthnet_decoder(features_right))
                disp_output_right = disp_right # to_image_list(disp_right)
        else:
            features_right = None

        if self.training and self.cfg.MODEL.DEPTHNET_ON:
            disp_loss = self.depthnet_loss(disp_output_left, disp_output_right, images_left.tensors, images_right.tensors)

        # use left image to train rpn
        proposals_left, proposal_losses_left = self.rpn(images_left, features_left, targets_left)
        # if self.training:
        #     proposals_right, proposal_losses_right = self.rpn(images_right, features_right, targets_right)
        # else:
        #     proposals_right = None

        # run roi heads with left image to train supervised
        if self.roi_heads:
            # x, result, detector_losses = self.roi_heads(features_left, proposals_left, features_right, proposals_right, targets_left, targets_right)
            x, result, detector_losses = self.roi_heads(features_left, proposals_left, targets_left)
        else:
            # RPN-only models don't have roi_heads
            x = features_left
            result = proposals_left
            detector_losses = {}
        
            # # TODO: create proposals with the estimated disparity
            # proposals_right = transform_proposals(proposals_left, result)

            # # run roi heads with right image
            # if self.roi_heads:
            #     x_, result_, detector_losses_right = self.roi_heads(features_right, proposals_right, targets)
            # else:
            #     # RPN-only models don't have roi_heads
            #     x_ = features_right
            #     result_ = proposals_right
            #     detector_losses_right = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses_left)
            # losses.update(proposal_losses_right)
            if self.cfg.MODEL.DEPTHNET_ON:
                losses.update(disp_loss)
            return losses

        if self.cfg.MODEL.DEPTHNET_ON:
            [print(i, disp_output_left[i].shape) for i in range(len(disp_output_left))]
            result.add_data("disparity", disp_output_left[-1])
            return result
        
        return result
