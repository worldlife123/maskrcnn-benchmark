# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor, make_roi_box_lr_predictor
from .inference import make_roi_box_post_processor, make_roi_box_lr_post_processor
from .loss import make_roi_box_loss_evaluator, make_roi_box_lr_loss_evaluator

from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_union

class ROIBoxLRHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg, in_channels, is_mt=False):
        super(ROIBoxLRHead, self).__init__()
        self.cfg = cfg.clone()
        self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_box_lr_predictor(
            cfg, self.feature_extractor.out_channels * 2)
        self.post_processor = make_roi_box_lr_post_processor(cfg)
        self.loss_evaluator = make_roi_box_lr_loss_evaluator(cfg) # for subsampling

        self.is_mt = is_mt

        if self.cfg.MODEL.ROI_BOX_HEAD.FREEZE_WEIGHT:
            for m in [self.feature_extractor, self.predictor]:
                for p in m.parameters():
                    p.requires_grad = False

    def forward(self, features_left, proposals_left, features_right=None, proposals_right=None, targets_left=None, targets_right=None, proposals_sampled=None):
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

        # generate right from left(TODO: TEMP solution for inconsistent ground truth)
        # if not targets_left is None:
        #     targets_right = []
        #     for target in targets_left:
        #         target_right = target.copy_with_fields("labels").convert("xywh")
        #         disps = target.get_field("depths").convert("disp").depths
        #         target_right.bbox[:,0] -= disps
        #         targets_right.append(target_right.convert("xyxy"))

        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            if proposals_sampled is None:
                with torch.no_grad():
                    proposals_sampled_left, proposals_sampled_right = self.loss_evaluator.subsample(proposals_left, proposals_right, targets_left, targets_right)
            proposals_left, proposals_right = proposals_sampled_left, proposals_sampled_right

        # calculate proposals_union
        proposals_union = []
        # print(len(proposals_left[0]), len(proposals_right[0]))
        for tl, tr in zip(proposals_left, proposals_right):
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
            proposals_union.append(BoxList(new_bbox, tl.size, mode="xyxy"))

        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        fl = self.feature_extractor(features_left, proposals_union)
        fr = self.feature_extractor(features_right, proposals_union)
        x = torch.cat([fl, fr], dim=1)
        # final classifier that converts the features into predictions
        class_logits, box_regression_left, box_regression_right = self.predictor(x)

        if not self.training:
            # result_left = self.post_processor((class_logits, box_regression_left), proposals_union)
            # result_right = self.post_processor((class_logits, box_regression_right), proposals_union)
            result_left, result_right = self.post_processor((class_logits, box_regression_left, box_regression_right), proposals_union)
            # resample
            # result_union = [boxlist_union(rl, rr) for rl,rr in zip(result_left, result_right)]
            # fl = self.feature_extractor(features_left, result_union)
            # fr = self.feature_extractor(features_right, result_union)
            # x = torch.cat([fl, fr], dim=1)
            return x, result_left, result_right, {}

        # TODO: loss is not needed for mean teacher when MT_ON
        if not self.cfg.MODEL.ROI_BOX_HEAD.FREEZE_WEIGHT:
            loss_classifier, loss_box_reg, loss_box_reg_right = self.loss_evaluator(
                [class_logits], [box_regression_left], [box_regression_right], proposals_union
            )

        # if self.cfg.MODEL.ROI_BOX_HEAD.OUTPUT_DECODED_PROPOSAL:
        #     bbox_reg_weights = self.cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
        #     box_coder = BoxCoder(weights=bbox_reg_weights)
        #     boxes_per_image = [len(box) for box in proposals]
        #     concat_boxes = torch.cat([a.bbox for a in proposals], dim=0)
        #     decoded_proposals = box_coder.decode(
        #         box_regression_left.view(sum(boxes_per_image), -1), concat_boxes
        #     )
        #     decoded_proposals = decoded_proposals.split(boxes_per_image, dim=0)
        #     # decoded_proposals = self.post_processor((class_logits, box_regression), proposals)
        #     # make sure there are valid proposals
        #     for i, boxes in enumerate(decoded_proposals):
        #         if len(boxes) > 0:
        #             proposals[i].bbox = boxes.reshape(-1, 4)

        loss_dict = dict()
        
        # if self.cfg.MODEL.MT_ON:
        #     loss_dict.update(class_logits=class_logits, box_logits=box_regression_left)
            # loss_dict.update(class_logits=x, box_logits=x)
            # proposals_sampled.add_field('class_logits', class_logits)
            # proposals_sampled.add_field('box_logits', box_regression)

        if not self.is_mt and not self.cfg.MODEL.ROI_BOX_HEAD.FREEZE_WEIGHT:
            loss_dict.update(dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg, loss_box_reg_right=loss_box_reg_right))

        return x, proposals_left, proposals_right, loss_dict
        


def build_roi_box_lr_head(cfg, in_channels, is_mt=False):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBoxLRHead(cfg, in_channels, is_mt)
