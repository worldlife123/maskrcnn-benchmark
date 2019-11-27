# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
import torch.nn.functional as F

from maskrcnn_benchmark.structures.bounding_box import BoxList
from .roi_depth_feature_extractors import make_roi_depth_feature_extractor
from .roi_depth_predictors import make_roi_depth_predictor
from .inference import make_roi_depth_post_processor
from .loss import make_roi_depth_loss_evaluator

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

class ROIDepthHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg, in_channels):
        super(ROIDepthHead, self).__init__()
        self.cfg = cfg.clone()
        
        self.feature_extractor = make_roi_depth_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_depth_predictor(
            cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_depth_post_processor(cfg)
        self.loss_evaluator = make_roi_depth_loss_evaluator(cfg)
        if self.cfg.MODEL.ROI_DEPTH_HEAD.FREEZE_WEIGHT:
            for m in [self.feature_extractor, self.predictor]:
                for p in m.parameters():
                    p.requires_grad = False

    def forward(self, features, proposals, targets=None, extra_features=None):
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
            x = features
            x = x[torch.cat(positive_inds, dim=0)]
        else:
            if self.cfg.MODEL.ROI_DEPTH_HEAD.INPUT_MASK_FEATURES:
                x = self.feature_extractor(features, proposals, extra_features=extra_features)
            else:
                x = self.feature_extractor(features, proposals)

        depth_logits = self.predictor(x)

        if not self.training:
            result = self.post_processor(depth_logits, proposals)
            return x, result, {}

        loss_dict = dict()
        
        if self.cfg.MODEL.MT_ON:
            loss_dict.update(dict(depth_logits=depth_logits))
            # loss_dict.update(dict(depth_logits=x))
            # proposals.add_field('depth_logits', depth_logits)

        if not self.cfg.MODEL.ROI_DEPTH_HEAD.FREEZE_WEIGHT:
            loss_depth = self.loss_evaluator(proposals, depth_logits, targets)
            loss_dict.update(dict(loss_depth=loss_depth))

        return x, all_proposals, loss_dict


def build_roi_depth_head(cfg, in_channels):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIDepthHead(cfg, in_channels)