# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from maskrcnn_benchmark.structures.bounding_box import BoxList
from .roi_depth_feature_extractors import make_roi_depth_feature_extractor
from .roi_depth_predictors import make_roi_depth_predictor, make_roi_depth_lr_predictor
from .inference import make_roi_depth_post_processor
from .loss import make_roi_depth_loss_evaluator, make_roi_depth_lr_loss_evaluator

def keep_only_positive_boxes(boxes):
    """
    Given a set of BoxList containing the `labels` field,
    return a set of BoxList for which `labels > 0`.

    Arguments:
        boxes (list of BoxList)
    """
    assert isinstance(boxes, (list, tuple))
    assert isinstance(boxes[0], BoxList)
    assert boxes[0].has_field("labels")
    positive_boxes = []
    positive_inds = []
    num_boxes = 0
    for boxes_per_image in boxes:
        labels = boxes_per_image.get_field("labels")
        inds_mask = labels > 0
        inds = inds_mask.nonzero().squeeze(1)
        positive_boxes.append(boxes_per_image[inds])
        positive_inds.append(inds_mask)
    return positive_boxes, positive_inds

class ROIDepthLRHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg, in_channels):
        super(ROIDepthLRHead, self).__init__()
        self.cfg = cfg.clone()
        
        self.feature_extractor = make_roi_depth_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_depth_predictor(
            cfg, self.feature_extractor.out_channels)
        if self.training:
            self.predictor_lr = make_roi_depth_lr_predictor(
                cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_depth_post_processor(cfg)
        # self.loss_evaluator = make_roi_depth_loss_evaluator(cfg)
        self.loss_evaluator_lr = make_roi_depth_lr_loss_evaluator(cfg)
        if self.cfg.MODEL.ROI_DEPTH_HEAD.FREEZE_WEIGHT:
            for m in [self.feature_extractor, self.predictor]:
                for p in m.parameters():
                    p.requires_grad = False

    def forward(self, features_left, proposals, features_right=None, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """
        if self.training:
            # during training, only focus on positive boxes
            all_proposals = proposals
            proposals, positive_inds = keep_only_positive_boxes(proposals)
        if self.training and self.cfg.MODEL.ROI_DEPTH_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            x = features_left
            x = x[torch.cat(positive_inds, dim=0)]
        else:
            x = self.feature_extractor(features_left, proposals)
        depth_logits = self.predictor(x)

        # add depth estimation to proposals
        result = self.post_processor(depth_logits, proposals)
        if not self.training:
            return x, result, {}, []
        else:
            # generate proposals_right
            proposals_right = self.generate_right_proposals(result)
            proposals_right, positive_inds_right = keep_only_positive_boxes(proposals_right)
            # print(result[0].bbox, proposals_right[0].bbox)
            if self.cfg.MODEL.ROI_DEPTH_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
                y = features_right
                y = y[torch.cat(positive_inds_right, dim=0)]
            else:
                y = self.feature_extractor(features_right, proposals_right)
            depth_lr_logits = self.predictor_lr((x,y))

            if not self.cfg.MODEL.ROI_DEPTH_HEAD.FREEZE_WEIGHT:
                # loss_depth = self.loss_evaluator(proposals, depth_logits, targets)
                loss_depth, loss_depth_lr, loss_depth_regularizer = self.loss_evaluator_lr(proposals, depth_logits, depth_lr_logits, targets)

                return x, result, dict(loss_depth=loss_depth, loss_depth_lr=loss_depth_lr, loss_depth_regularizer=loss_depth_regularizer), proposals_right

            return x, result, dict(), proposals_right

    def generate_right_proposals(self, boxes):
        boxes_right = []
        for boxes_per_image in boxes:
            if boxes_per_image.mode != "xyxy": boxes_per_image.convert("xyxy")
            bbox_right = boxes_per_image.bbox.clone()
            disp_unity = boxes_per_image.get_field("depths")
            disparity = disp_unity * boxes_per_image.size[0] * 0.209313 # TODO:  embed them in ground truth
            # print(bbox_right)
            bbox_right[:,0::4] -= disparity
            bbox_right[:,2::4] -= disparity
            bbox_right = BoxList(bbox_right, boxes_per_image.size, mode=boxes_per_image.mode)
            bbox_right._copy_extra_fields(boxes_per_image)
            boxes_right.append(bbox_right)
        return boxes_right


def build_roi_depth_head(cfg, in_channels):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIDepthLRHead(cfg, in_channels)
