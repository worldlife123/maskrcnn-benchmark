# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_union
from maskrcnn_benchmark.structures.point_3d import PointDepth
from maskrcnn_benchmark.layers import smooth_l1_loss

from .roi_depth_feature_extractors import make_roi_depth_feature_extractor
from .roi_depth_predictors import make_roi_depth_predictor, make_roi_depth_lr_predictor
from .inference import make_roi_depth_post_processor, make_roi_depth_lr_post_processor
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
        # self.feature_extractor_extra = make_roi_depth_feature_extractor(cfg, in_channels)
        if self.cfg.MODEL.ROI_HEADS_LR.MONO_HEAD_ON:
            self.feature_extractor_mono = make_roi_depth_feature_extractor(cfg, in_channels, self.feature_extractor.out_channels*2)
            self.predictor_mono = make_roi_depth_predictor(cfg, self.feature_extractor.out_channels*3)
            if self.cfg.MODEL.ROI_HEADS_LR.MONO_HEAD_FREEZE_WEIGHT:
                for p in self.feature_extractor_mono.parameters():
                    p.requires_grad = False
                for p in self.predictor_mono.parameters():
                    p.requires_grad = False
        self.predictor = make_roi_depth_predictor(
            cfg, self.feature_extractor.out_channels*3)
        # if self.training:
        #     self.predictor_lr = make_roi_depth_lr_predictor(
        #         cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_depth_post_processor(cfg)
        self.loss_evaluator = make_roi_depth_loss_evaluator(cfg)
        if self.cfg.MODEL.ROI_DEPTH_HEAD.DEPTH_FROM_LR_DETECTION:
            self.post_processor_lr = make_roi_depth_lr_post_processor(cfg)
            self.loss_evaluator_lr = make_roi_depth_lr_loss_evaluator(cfg)
        # self.loss_evaluator_lr = make_roi_depth_lr_loss_evaluator(cfg)
        if self.cfg.MODEL.ROI_DEPTH_HEAD.FREEZE_WEIGHT or self.cfg.MODEL.ROI_HEADS_LR.STEREO_HEAD_FREEZE_WEIGHT:
            for m in [self.feature_extractor, self.predictor]:
                for p in m.parameters():
                    p.requires_grad = False

    def forward(self, features_left, proposals_left, features_right=None, proposals_right=None, targets_left=None, targets_right=None, extra_features=None, img_info=None):
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

        loss_dict = dict()

        # generate right from left(TODO: TEMP solution for inconsistent ground truth)
        # if not targets_left is None:
        #     targets_right = []
        #     for target in targets_left:
        #         target_right = target.copy_with_fields("labels").convert("xywh")
        #         disps = target.get_field("depths").convert("disp").depths
        #         target_right.bbox[:,0] -= disps
        #         targets_right.append(target_right.convert("xyxy"))

        if self.training:
            # during training, only focus on positive boxes
            all_proposals = proposals_left
            proposals_left, positive_inds = keep_only_positive_boxes(proposals_left)
            proposals_right = [p[idx] for p, idx in zip(proposals_right, positive_inds)]
            if not extra_features is None: extra_features = extra_features[torch.cat(positive_inds, dim=0)]
        
        proposals_union = [boxlist_union(pl, pr) for pl, pr in zip(proposals_left, proposals_right)] # [p[idx] for p, idx in zip(proposals_union, positive_inds)]

        # calculate proposals_union
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

        # if self.cfg.MODEL.ROI_DEPTH_HEAD.DEPTH_FROM_LR_DETECTION:
        #     # estimate depth_logits from detection
        #     results_left, results_right = [], []
        #     for pl, pr in zip(proposals_left, proposals_right):
        #         pl = pl.convert("xyxy")
        #         pr = pr.convert("xyxy")
        #         disps = (pl.bbox[:,0] + pl.bbox[:,2] - pr.bbox[:,0] - pr.bbox[:,2]).unsqueeze(-1) / 2
        #         # TODO: convert to depth
        #         pl.add_field("depths", disps)
        #         pr.add_field("depths", disps)
        #         results_left.append(pl)
        #         results_right.append(pr)
        #     # TODO : loss
        #     return features_left, results_left, {}
        
        if self.training and self.cfg.MODEL.ROI_DEPTH_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            xl = features_left
            xl = xl[torch.cat(positive_inds, dim=0)]
            depth_logits_left = self.predictor(xl)
        else:
            fl = self.feature_extractor(features_left, proposals_left)
            if not features_right is None and (self.training or not self.cfg.MODEL.ROI_HEADS_LR.MONO_HEAD_TEST):
                # fr = self.feature_extractor(features_right, proposals_right)
                # if self.training and self.cfg.MODEL.ROI_HEADS_LR.LOSS_FEATURE_CONSISTENCY > 0.0:
                #     fr2 = self.feature_extractor_extra(features_left, proposals_left)
                #     feature_loss = smooth_l1_loss(
                #         fr, fr2
                #     ) * self.cfg.MODEL.ROI_HEADS_LR.LOSS_FEATURE_CONSISTENCY
                #     loss_dict.update(dict(depth_fc_loss=feature_loss))
                #     fr = fr2
                if self.cfg.MODEL.ROI_DEPTH_HEAD.INPUT_BOX_FEATURES:
                    xu = extra_features # [torch.cat(positive_inds, dim=0)]
                else:
                    ful = self.feature_extractor(features_left, proposals_union)
                    fur = self.feature_extractor(features_right, proposals_union)
                    xu = torch.cat([ful, fur], dim=1)
                xl = torch.cat([fl, xu], dim=1) # torch.cat([fl, fr, xu], dim=1)
                # xr = torch.cat([fr, fu], dim=1)
                depth_logits_left = self.predictor(xl)
                # depth_logits_right = self.predictor(xr)
                # print(fl.shape, fr.shape, xu.shape)
                if self.cfg.MODEL.ROI_HEADS_LR.MONO_HEAD_ON:
                    xu_mono = self.feature_extractor_mono(features_left, proposals_union)
                    xl_mono = torch.cat([fl, xu_mono], dim=1)
                    depth_logits_mono_left = self.predictor_mono(xl_mono)
                    if self.training and self.cfg.MODEL.ROI_HEADS_LR.LOSS_FEATURE_CONSISTENCY > 0.0:
                        feature_loss = smooth_l1_loss(
                            xu, xu_mono
                        ) * self.cfg.MODEL.ROI_HEADS_LR.LOSS_FEATURE_CONSISTENCY
                        loss_dict.update(dict(loss_depth_fc=feature_loss))
                    if self.training and self.cfg.MODEL.ROI_HEADS_LR.LOSS_PREDICTOR_CONSISTENCY > 0.0:
                        head_loss = smooth_l1_loss(
                            depth_logits_left, depth_logits_mono_left
                        ) * self.cfg.MODEL.ROI_HEADS_LR.LOSS_PREDICTOR_CONSISTENCY
                        loss_dict.update(dict(loss_depth_pc=head_loss))
            elif self.cfg.MODEL.ROI_HEADS_LR.MONO_HEAD_ON:
                xu = self.feature_extractor_mono(features_left, proposals_union)
                xl = torch.cat([fl, xu], dim=1) # torch.cat([fl, fr, xu], dim=1)
                # xr = torch.cat([fr, fu], dim=1)
                depth_logits_left = self.predictor_mono(xl)
            else:
                raise NotImplementedError("No heads for roi_depth_lr is given!")

        # modify regression value according to baseline
        # for img_info_single, logit in zip(img_info, depth_logits_left):
        #     if img_info_single["camera_params"]["extrinsic"].get("baseline"):
        #         baseline_modifier = self.cfg.MODEL.ROI_HEADS_LR.REFERENCE_BASELINE / img_info_single["camera_params"]["extrinsic"]["baseline"]
        #         logit *= baseline_modifier

        # add depth estimation to proposals
        if self.cfg.MODEL.ROI_DEPTH_HEAD.DEPTH_FROM_LR_DETECTION:
            result = self.post_processor_lr(depth_logits_left, proposals_left, proposals_right, img_info=img_info)
        else:
            result = self.post_processor(depth_logits_left, proposals_left, img_info=img_info)
        
        if not self.training:
            return xl, result, {} #, []
        else:
            # # generate proposals_right
            # proposals_right = self.generate_right_proposals(result)
            # proposals_right, positive_inds_right = keep_only_positive_boxes(proposals_right)
            # # print(result[0].bbox, proposals_right[0].bbox)
            # if self.cfg.MODEL.ROI_DEPTH_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            #     y = features_right
            #     y = y[torch.cat(positive_inds_right, dim=0)]
            # else:
            #     y = self.feature_extractor(features_right, proposals_right)
            # depth_lr_logits = self.predictor_lr((x,y))

            if not self.cfg.MODEL.ROI_DEPTH_HEAD.FREEZE_WEIGHT:
                if self.cfg.MODEL.ROI_DEPTH_HEAD.DEPTH_FROM_LR_DETECTION:
                    loss_depth = self.loss_evaluator_lr(proposals_left, proposals_right, depth_logits_left, targets_left, targets_right)
                else:
                    loss_depth = self.loss_evaluator(proposals_left, depth_logits_left, targets_left)
                    # loss_depth_right = self.loss_evaluator(proposals_right, depth_logits_right, targets_right)
                # loss_depth, loss_depth_lr, loss_depth_regularizer = self.loss_evaluator_lr(proposals, depth_logits, depth_lr_logits, targets)
                loss_dict.update(dict(loss_depth=loss_depth))
                return xl, result, loss_dict # , loss_depth_right=loss_depth_right) #, proposals_right
            return xl, result, loss_dict #, proposals_right
            

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


def build_roi_depth_lr_head(cfg, in_channels):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIDepthLRHead(cfg, in_channels)
