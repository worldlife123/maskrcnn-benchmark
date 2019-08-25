# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from .box_head.box_head import build_roi_box_head
from .mask_head.mask_head import build_roi_mask_head
from .keypoint_head.keypoint_head import build_roi_keypoint_head
from .depth_head.depth_lr_head import build_roi_depth_head

class CombinedROILRHeads(torch.nn.ModuleDict):
    """
    Combines a set of individual heads (for box prediction or masks) into a single
    head.
    """

    def __init__(self, cfg, heads):
        super(CombinedROILRHeads, self).__init__(heads)
        self.cfg = cfg.clone()
        if cfg.MODEL.MASK_ON and cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.mask.feature_extractor = self.box.feature_extractor
        if cfg.MODEL.KEYPOINT_ON and cfg.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.keypoint.feature_extractor = self.box.feature_extractor
        if cfg.MODEL.DEPTH_ON and cfg.MODEL.ROI_DEPTH_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.depth.feature_extractor = self.box.feature_extractor

    def forward(self, features_left, proposals_left, features_right=None,  proposals_right=None, targets_left=None, targets_right=None):
        losses = {}
        # TODO rename x to roi_box_features, if it doesn't increase memory consumption
        x, detections, loss_box = self.box(features_left, proposals_left, targets_left)
        losses.update(loss_box)
        if self.cfg.MODEL.MASK_ON:
            mask_features = features_left
            # optimization: during training, if we share the feature extractor between
            # the box and the mask heads, then we can reuse the features already computed
            if (
                self.training
                and self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
            ):
                mask_features = x
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            x, detections, loss_mask = self.mask(mask_features, detections, targets_left)
            losses.update(loss_mask)

        if self.cfg.MODEL.KEYPOINT_ON:
            keypoint_features = features_left
            # optimization: during training, if we share the feature extractor between
            # the box and the mask heads, then we can reuse the features already computed
            if (
                self.training
                and self.cfg.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
            ):
                keypoint_features = x
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            x, detections, loss_keypoint = self.keypoint(keypoint_features, detections, targets_left)
            losses.update(loss_keypoint)

        if self.cfg.MODEL.DEPTH_ON:
            depth_features = features_left
            # optimization: during training, if we share the feature extractor between
            # the box and the mask heads, then we can reuse the features already computed
            if (
                self.training
                and self.cfg.MODEL.ROI_DEPTH_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
            ):
                depth_features = x
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            # overwrite proposals_right
            x, detections, loss_depth, proposals_right = self.depth(depth_features, detections, features_right, targets_left)
            losses.update(loss_depth)

        # process right feature when training
        '''
        if self.training:
            if self.cfg.MODEL.DEPTH_ON:
                # [print(a) for a in [proposals_left, proposals_right, targets_left, targets_right]]
                # resize proposals_right
                # for p in proposals_right: print(p.get_field("labels"))
                for i, p in enumerate(proposals_right):
                    resized_proposal = p.resize(targets_right[i].size)
                    resized_proposal.add_field("labels", p.get_field("labels"))
                    proposals_right[i] = resized_proposal
                    # print(i, proposals_right[i].get_field("labels"))
                x, detections, loss_box = self.box(features_right, proposals_right, targets_right)
                loss_box = dict(loss_classifier_right=loss_box["loss_classifier"], loss_box_reg_right=loss_box["loss_box_reg"])
                losses.update(loss_box)
                depth_features = features_right
                # optimization: during training, if we share the feature extractor between
                # the box and the mask heads, then we can reuse the features already computed
                if (
                    self.training
                    and self.cfg.MODEL.ROI_DEPTH_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
                ):
                    depth_features = x
                # During training, self.box() will return the unaltered proposals as "detections"
                # this makes the API consistent during training and testing
                # for p in proposals_right: print(p.get_field("labels"))
                
                x, detections, loss_depth, _ = self.depth(depth_features, proposals_right, targets_right)
                loss_depth = dict(loss_depth_right=loss_depth["loss_depth"])
                losses.update(loss_depth)
        '''


        return x, detections, losses


def build_roi_lr_heads(cfg, in_channels):
    # individually create the heads, that will be combined together
    # afterwards
    roi_heads = []
    if cfg.MODEL.RETINANET_ON:
        return []

    if not cfg.MODEL.RPN_ONLY:
        roi_heads.append(("box", build_roi_box_head(cfg, in_channels)))
    if cfg.MODEL.MASK_ON:
        roi_heads.append(("mask", build_roi_mask_head(cfg, in_channels)))
    if cfg.MODEL.KEYPOINT_ON:
        roi_heads.append(("keypoint", build_roi_keypoint_head(cfg, in_channels)))
    if cfg.MODEL.DEPTH_ON:
        roi_heads.append(("depth", build_roi_depth_head(cfg, in_channels)))
    # combine individual heads in a single module
    if roi_heads:
        roi_heads = CombinedROILRHeads(cfg, roi_heads)

    return roi_heads
