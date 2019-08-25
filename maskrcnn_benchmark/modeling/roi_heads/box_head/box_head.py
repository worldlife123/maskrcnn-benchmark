# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor
from .inference import make_roi_box_post_processor
from .loss import make_roi_box_loss_evaluator

from maskrcnn_benchmark.modeling.box_coder import BoxCoder


class ROIBoxHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg, in_channels):
        super(ROIBoxHead, self).__init__()
        self.cfg = cfg.clone()
        self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_box_predictor(
            cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_box_post_processor(cfg)
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg)
        if self.cfg.MODEL.ROI_BOX_HEAD.FREEZE_WEIGHT:
            for m in [self.feature_extractor, self.predictor]:
                for p in m.parameters():
                    p.requires_grad = False

    def forward(self, features, proposals, targets=None):
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
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with torch.no_grad():
                proposals = self.loss_evaluator.subsample(proposals, targets)

        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        x = self.feature_extractor(features, proposals)
        # final classifier that converts the features into predictions
        class_logits, box_regression = self.predictor(x)

        if not self.training:
            result = self.post_processor((class_logits, box_regression), proposals)
            return x, result, {}

        if not self.cfg.MODEL.ROI_BOX_HEAD.FREEZE_WEIGHT:
            loss_classifier, loss_box_reg = self.loss_evaluator(
                [class_logits], [box_regression]
            )

        if self.cfg.MODEL.ROI_BOX_HEAD.OUTPUT_DECODED_PROPOSAL:
            bbox_reg_weights = self.cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
            box_coder = BoxCoder(weights=bbox_reg_weights)
            boxes_per_image = [len(box) for box in proposals]
            concat_boxes = torch.cat([a.bbox for a in proposals], dim=0)
            decoded_proposals = box_coder.decode(
                box_regression.view(sum(boxes_per_image), -1), concat_boxes
            )
            decoded_proposals = decoded_proposals.split(boxes_per_image, dim=0)
            # decoded_proposals = self.post_processor((class_logits, box_regression), proposals)
            # make sure there are valid proposals
            for i, boxes in enumerate(decoded_proposals):
                if len(boxes) > 0:
                    proposals[i].bbox = boxes.reshape(-1, 4)

        if not self.cfg.MODEL.ROI_BOX_HEAD.FREEZE_WEIGHT:
            return (
                x,
                proposals,
                dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg),
            )

        return x, proposals, dict()
        


def build_roi_box_head(cfg, in_channels):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBoxHead(cfg, in_channels)
