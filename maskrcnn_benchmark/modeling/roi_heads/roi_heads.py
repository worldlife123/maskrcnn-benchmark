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

    def forward(self, features, proposals, targets=None, proposals_sampled=None):
        losses = {}
        outputs = {}
        # TODO rename x to roi_box_features, if it doesn't increase memory consumption
        
        features_depth = None
        # if len(features) == 8:
        #     features, features_depth = features[0:4], features[4:8]
        if isinstance(features, dict):
            features_depth = features["features_depth"] if features.get("features_depth") else None
            features = features["features"]
        
        x, detections, loss_box = self.box(features, proposals, targets, proposals_sampled=proposals_sampled)
        # outputs.update(dict(output_box=x))
        if self.training and self.cfg.MODEL.MT_ON:
            # loss_box.pop("class_logits")
            # loss_box.pop("box_logits")
            outputs.update(dict(class_logits=loss_box.pop("class_logits"), box_logits=loss_box.pop("box_logits")))
        losses.update(loss_box)

        mask_features = None
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
            mask_features, detections, loss_mask = self.mask(mask_features, detections_for_mask, targets)
            # outputs.update(dict(output_mask=x))
            if self.training and self.cfg.MODEL.MT_ON:
                outputs.update(dict(mask_logits=loss_mask.pop("mask_logits")))
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
            keypoint_features, detections, loss_keypoint = self.keypoint(keypoint_features, detections_for_keypoint, targets)
            # outputs.update(dict(output_keypoint=x))
            if self.training and self.cfg.MODEL.MT_ON:
                outputs.update(dict(keypoint_logits=loss_keypoint.pop("keypoint_logits")))
            losses.update(loss_keypoint)

        if self.cfg.MODEL.DEPTH_ON:
            depth_features = features if features_depth is None else features_depth
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

            extra_features = mask_features if self.cfg.MODEL.ROI_DEPTH_HEAD.INPUT_MASK_FEATURES else None
            depth_features, detections, loss_depth = self.depth(depth_features, detections_for_depth, targets, extra_features=extra_features)
            # outputs.update(dict(output_depth=x))
            if self.training and self.cfg.MODEL.MT_ON:
                outputs.update(dict(depth_logits=loss_depth.pop("depth_logits")))
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
            box3d_features, detections, loss_box3d = self.box3d(box3d_features, detections_for_box3d, targets)
            # outputs.update(dict(output_box3d=x))
            if self.training and self.cfg.MODEL.MT_ON:
                outputs.update(dict(box3d_logits=loss_box3d.pop("box3d_logits")))
            losses.update(loss_box3d)

        # mt trainer need output
        if self.cfg.MODEL.MT_ON:
            x = outputs

        return x, detections, losses


def build_roi_heads(cfg, in_channels, is_mt=False):
    # individually create the heads, that will be combined together
    # afterwards
    roi_heads = []
    if cfg.MODEL.RETINANET_ON:
        return []

    if not cfg.MODEL.RPN_ONLY:
        roi_heads.append(("box", build_roi_box_head(cfg, in_channels, is_mt)))
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
