# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from .box_head.box_head import build_roi_box_head
from .mask_head.mask_head import build_roi_mask_head
from .keypoint_head.keypoint_head import build_roi_keypoint_head
from .depth_head.depth_head import build_roi_depth_head
from .box3d_head.box3d_head import build_roi_box3d_head

class CombinedROIHeads(torch.nn.ModuleDict):
    """
    Combines a set of individual heads (for box prediction or masks) into a single
    head.
    """

    def __init__(self, cfg, heads):
        super(CombinedROIHeads, self).__init__(heads)
        self.cfg = cfg.clone()
        if cfg.MODEL.MASK_ON and cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.mask.feature_extractor = self.box.feature_extractor
        if cfg.MODEL.KEYPOINT_ON and cfg.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.keypoint.feature_extractor = self.box.feature_extractor
        if cfg.MODEL.DEPTH_ON and cfg.MODEL.ROI_DEPTH_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.depth.feature_extractor = self.box.feature_extractor
        if cfg.MODEL.BOX3D_ON and cfg.MODEL.ROI_BOX3D_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.box3d.feature_extractor = self.box.feature_extractor

    def forward(self, features, proposals, targets=None):
        losses = {}
        # TODO rename x to roi_box_features, if it doesn't increase memory consumption
        x, detections, loss_box = self.box(features, proposals, targets)
        losses.update(loss_box)
        if self.cfg.MODEL.MASK_ON:
            mask_features = features
            # optimization: during training, if we share the feature extractor between
            # the box and the mask heads, then we can reuse the features already computed
            if (
                self.training
                and self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
            ):
                mask_features = x
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            if (
                self.training
                and not self.cfg.MODEL.ROI_MASK_HEAD.USE_DECODED_PROPOSAL
            ):
                detections_for_mask = proposals
            else:
                detections_for_mask = detections
            x, detections, loss_mask = self.mask(mask_features, detections_for_mask, targets)
            losses.update(loss_mask)

        if self.cfg.MODEL.KEYPOINT_ON:
            keypoint_features = features
            # optimization: during training, if we share the feature extractor between
            # the box and the mask heads, then we can reuse the features already computed
            if (
                self.training
                and self.cfg.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
            ):
                keypoint_features = x
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            if (
                self.training
                and not self.cfg.MODEL.ROI_KEYPOINT_HEAD.USE_DECODED_PROPOSAL
            ):
                detections_for_keypoint = proposals
            else:
                detections_for_keypoint = detections
            x, detections, loss_keypoint = self.keypoint(keypoint_features, detections_for_keypoint, targets)
            losses.update(loss_keypoint)

        if self.cfg.MODEL.DEPTH_ON:
            depth_features = features
            # optimization: during training, if we share the feature extractor between
            # the box and the mask heads, then we can reuse the features already computed
            if (
                self.training
                and self.cfg.MODEL.ROI_DEPTH_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
            ):
                depth_features = x
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            if (
                self.training
                and not self.cfg.MODEL.ROI_DEPTH_HEAD.USE_DECODED_PROPOSAL
            ):
                detections_for_depth = proposals
            else:
                detections_for_depth = detections
            x, detections, loss_depth = self.depth(depth_features, detections_for_depth, targets)
            losses.update(loss_depth)

        if self.cfg.MODEL.BOX3D_ON:
            box3d_features = features
            # optimization: during training, if we share the feature extractor between
            # the box and the mask heads, then we can reuse the features already computed
            if (
                self.training
                and self.cfg.MODEL.ROI_BOX3D_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
            ):
                box3d_features = x
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            if (
                self.training
                and not self.cfg.MODEL.ROI_BOX3D_HEAD.USE_DECODED_PROPOSAL
            ):
                detections_for_box3d = proposals
            else:
                detections_for_box3d = detections
            x, detections, loss_box3d = self.box3d(box3d_features, detections_for_box3d, targets)
            losses.update(loss_box3d)

        return x, detections, losses


def build_roi_heads(cfg, in_channels):
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
    if cfg.MODEL.BOX3D_ON:
        roi_heads.append(("box3d", build_roi_box3d_head(cfg, in_channels)))
    # combine individual heads in a single module
    if roi_heads:
        roi_heads = CombinedROIHeads(cfg, roi_heads)

    return roi_heads
