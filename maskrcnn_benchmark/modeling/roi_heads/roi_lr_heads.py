# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from maskrcnn_benchmark.structures.bounding_box import BoxList

from .box_head.box_head import build_roi_box_head
from .box_head.box_lr_head import build_roi_box_lr_head
from .mask_head.mask_head import build_roi_mask_head
from .keypoint_head.keypoint_head import build_roi_keypoint_head
from .depth_head.depth_lr_head import build_roi_depth_lr_head

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

    def forward(self, features_left, proposals_left, features_right=None,  proposals_right=None, targets_left=None, targets_right=None, img_info=None):
        losses = {}

        # # calculate proposals_union
        # proposals_union = []
        # # print(len(proposals_left[0]), len(proposals_right[0]))
        # for tl, tr in zip(proposals_left, proposals_right):
        #     assert(tl.size == tr.size)
        #     bbox_left, bbox_right = tl.convert("xyxy").bbox, tr.convert("xyxy").bbox
        #     # print(bbox_left, bbox_right)
        #     new_bbox = torch.stack([
        #         torch.min(bbox_left[:,0], bbox_right[:,0]),
        #         torch.min(bbox_left[:,1], bbox_right[:,1]),
        #         torch.max(bbox_left[:,2], bbox_right[:,2]),
        #         torch.max(bbox_left[:,3], bbox_right[:,3]),
        #     ], dim=1)
        #     # print(new_bbox)
        #     proposals_union.append(BoxList(new_bbox, tl.size, mode="xyxy"))
        # targets_union = None if targets_left is None else []
        # if not targets_left is None:
        #     for targets_per_image in targets_left:
        #         targets_union_per_image = targets_per_image.convert("xyxy")
        #         targets_union_per_image._copy_extra_fields(targets_per_image)
        #         disps = targets_per_image.get_field("depths").convert("disp").depths
        #         targets_union_per_image.bbox[:,0] -= disps
        #         targets_union.append(targets_union_per_image)
        # TODO rename x to roi_box_features, if it doesn't increase memory consumption
        # print(proposals_left[0])
        # TODO: box_lr_head
        # x, detections_left, loss_box_left = self.box(features_left, proposals_union, targets_union)
        # print(detections_left[0])
        # losses.update(loss_box_left)
        # x, detections_right, loss_box_right = self.box(features_right, proposals_union, targets_union)
        # losses.update({(k+"_right"):v for k,v in loss_box_right.items()})
        
        box_features, detections_left, detections_right, loss_box = self.box(features_left, proposals_left, features_right, proposals_right, targets_left, targets_right)
        losses.update(loss_box)

        # for pl,pr,dl,dr in zip(proposals_left, proposals_right, detections_left, detections_right):
        #     print(len(pl), len(pr), len(dl), len(dr))

        # calculate detections_union
        detections_union = []
        # print(len(detections_left[0]), len(detections_right[0]))
        for tl, tr in zip(detections_left, detections_right):
            assert(tl.size == tr.size)
            bbox_left, bbox_right = tl.convert("xyxy").bbox, tr.convert("xyxy").bbox
            # print(bbox_left, bbox_right)
            new_bbox = torch.stack([
                torch.min(bbox_left[:,0], bbox_right[:,0]),
                torch.min(bbox_left[:,1], bbox_right[:,1]),
                torch.max(bbox_left[:,2], bbox_right[:,2]),
                torch.max(bbox_left[:,3], bbox_right[:,3]),
            ], dim=1)
            # print(new_bbox)
            detections_union.append(BoxList(new_bbox, tl.size, mode="xyxy"))

        # x, detections_union, loss_box_union = self.box_union(features_left, proposals_union, targets_left)
        # losses.update({(k+"_union"):v for k,v in loss_box_union.items()})
        if self.cfg.MODEL.MASK_ON:
            mask_features = features_left
            # optimization: during training, if we share the feature extractor between
            # the box and the mask heads, then we can reuse the features already computed
            if (
                self.training
                and self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
            ):
                mask_features = box_features
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            mask_features, detections_left, loss_mask = self.mask(mask_features, detections_left, targets_left)
            losses.update(loss_mask)

        if self.cfg.MODEL.KEYPOINT_ON:
            keypoint_features = features_left
            # optimization: during training, if we share the feature extractor between
            # the box and the mask heads, then we can reuse the features already computed
            if (
                self.training
                and self.cfg.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
            ):
                keypoint_features = box_features
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            keypoint_features, detections_left, loss_keypoint = self.keypoint(keypoint_features, detections_left, targets_left)
            losses.update(loss_keypoint)

        if self.cfg.MODEL.DEPTH_ON:
            depth_features = features_left
            # optimization: during training, if we share the feature extractor between
            # the box and the mask heads, then we can reuse the features already computed
            if (
                self.training
                and self.cfg.MODEL.ROI_DEPTH_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
            ):
                depth_features = box_features
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            # overwrite proposals_right
            if self.cfg.MODEL.ROI_DEPTH_HEAD.INPUT_BOX_FEATURES:
                extra_features = box_features 
            elif self.cfg.MODEL.ROI_DEPTH_HEAD.INPUT_MASK_FEATURES:
                extra_features = mask_features 
            else:
                extra_features = None
            x, detections_left, loss_depth = self.depth(depth_features, detections_left, features_right, detections_right, targets_left, targets_right, extra_features=extra_features, img_info=img_info)
            losses.update(loss_depth)

        # process right feature when training
        # if self.training:
        #     if self.cfg.MODEL.DEPTH_ON:
        #         # [print(a) for a in [proposals_left, proposals_right, targets_left, targets_right]]
        #         # resize proposals_right
        #         # for p in proposals_right: print(p.get_field("labels"))
        #         for i, p in enumerate(proposals_right):
        #             resized_proposal = p.resize(targets_right[i].size)
        #             resized_proposal.add_field("labels", p.get_field("labels"))
        #             proposals_right[i] = resized_proposal
        #             # print(i, proposals_right[i].get_field("labels"))
        #         x, detections, loss_box = self.box(features_right, proposals_right, targets_right)
        #         loss_box = dict(loss_classifier_right=loss_box["loss_classifier"], loss_box_reg_right=loss_box["loss_box_reg"])
        #         losses.update(loss_box)
        #         depth_features = features_right
        #         # optimization: during training, if we share the feature extractor between
        #         # the box and the mask heads, then we can reuse the features already computed
        #         if (
        #             self.training
        #             and self.cfg.MODEL.ROI_DEPTH_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
        #         ):
        #             depth_features = x
        #         # During training, self.box() will return the unaltered proposals as "detections"
        #         # this makes the API consistent during training and testing
        #         # for p in proposals_right: print(p.get_field("labels"))
                
        #         x, detections, loss_depth, _ = self.depth(depth_features, proposals_right, targets_right)
        #         loss_depth = dict(loss_depth_right=loss_depth["loss_depth"])
        #         losses.update(loss_depth)


        return x, detections_left, losses


def build_roi_lr_heads(cfg, in_channels):
    # individually create the heads, that will be combined together
    # afterwards
    roi_heads = []
    if cfg.MODEL.RETINANET_ON:
        return []

    if not cfg.MODEL.RPN_ONLY:
        roi_heads.append(("box", build_roi_box_lr_head(cfg, in_channels)))
        # roi_heads.append(("box_union", build_roi_box_union_head(cfg, in_channels)))
    if cfg.MODEL.MASK_ON:
        roi_heads.append(("mask", build_roi_mask_head(cfg, in_channels)))
    if cfg.MODEL.KEYPOINT_ON:
        roi_heads.append(("keypoint", build_roi_keypoint_head(cfg, in_channels)))
    if cfg.MODEL.DEPTH_ON:
        roi_heads.append(("depth", build_roi_depth_lr_head(cfg, in_channels)))
    # combine individual heads in a single module
    if roi_heads:
        roi_heads = CombinedROILRHeads(cfg, roi_heads)

    return roi_heads
