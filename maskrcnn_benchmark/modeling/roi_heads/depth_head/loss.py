# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch.nn import functional as F

from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_union
from maskrcnn_benchmark.modeling.balanced_positive_negative_sampler import (
    BalancedPositiveNegativeSampler
)
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.structures.point_3d import PointDepth


class MaskRCNNDepthLossComputation(object):
    def __init__(self, cfg, proposal_matcher):
        """
        Arguments:
            proposal_matcher (Matcher)
        """
        self.cfg = cfg
        self.proposal_matcher = proposal_matcher

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # print(match_quality_matrix, matched_idxs)
        target = target.copy_with_fields(["labels", "depths"], skip_missing=True)
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets):
        labels = []
        depths = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            if len(proposals_per_image)==0: continue # skip
            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )

            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # this can probably be removed, but is left here for clarity
            # and completeness
            neg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[neg_inds] = 0

            # depth scores are only computed on positive samples
            positive_inds = torch.nonzero(labels_per_image > 0).squeeze(1)

            if not matched_targets.has_field("depths"):
                depths_per_image = torch.zeros(labels_per_image.size()).to(dtype=torch.float, device=labels_per_image.device)
            else:
                depths_per_image = matched_targets.get_field("depths")
                if isinstance(depths_per_image, PointDepth): 
                    depths_per_image = depths_per_image.depths
            # print(labels_per_image, depths_per_image)
            depths_per_image[neg_inds] = 0

            labels.append(labels_per_image)
            depths.append(depths_per_image)

        return labels, depths

    def __call__(self, proposals, depth_logits, targets):
        """
        Arguments:
            proposals (list[BoxList])
            depth_logits (Tensor)
            targets (list[BoxList])

        Return:
            depth_loss (Tensor): scalar tensor containing the loss
        """
        labels, depth_targets = self.prepare_targets(proposals, targets)
        if len(depth_targets)==0: return 0

        labels = cat(labels, dim=0)
        depth_targets = cat(depth_targets, dim=0)

        if self.cfg.MODEL.ROI_DEPTH_HEAD.REG_LOGARITHM:
            depth_targets = torch.log(depth_targets)

        depth_targets = depth_targets*self.cfg.MODEL.ROI_DEPTH_HEAD.REG_AMPLIFIER

        # positive_inds = torch.nonzero((labels > 0)).squeeze(1)
        positive_inds = torch.nonzero((labels > 0) & (depth_targets > 0)).squeeze(1)
        labels_pos = labels[positive_inds]
        # print(len(labels), len(positive_inds), len(depth_targets))
        # print(depth_logits)

        # torch.mean (in binary_cross_entropy_with_logits) doesn't
        # accept empty tensors, so handle it separately
        if depth_targets.numel() == 0:
            return depth_logits.sum() * 0

        # print(depth_logits[positive_inds, labels_pos], depth_targets[positive_inds])
        depth_loss = smooth_l1_loss(
            depth_logits[positive_inds, labels_pos], 
            depth_targets[positive_inds],
            size_average=False,
            beta=1,
        )
        depth_loss = depth_loss / labels.numel()

        # amplify
        depth_loss *= self.cfg.MODEL.ROI_DEPTH_HEAD.LOSS_AMPLIFIER
        return depth_loss

class MaskRCNNBox3dLossComputation(object):
    def __init__(self, cfg, proposal_matcher):
        """
        Arguments:
            proposal_matcher (Matcher)
        """
        self.cfg = cfg
        self.proposal_matcher = proposal_matcher

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # print(match_quality_matrix, matched_idxs)
        # Mask RCNN needs "labels" and "masks "fields for creating the targets
        target = target.copy_with_fields(["labels", "masks", "depths", "dims", "rotbins", "rotregs"], skip_missing=True)
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets):
        labels = []
        depths, dims, rotbins, rotregs = [], [], [], []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            if len(proposals_per_image)==0: continue # skip
            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )
            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # this can probably be removed, but is left here for clarity
            # and completeness
            neg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[neg_inds] = 0

            labels.append(labels_per_image)

            # depth scores are only computed on positive samples
            positive_inds = torch.nonzero(labels_per_image > 0).squeeze(1)

            depths_per_image = matched_targets.get_field("depths")
            # depths_per_image[neg_inds] = 0
            depths.append(depths_per_image[positive_inds])
            dims_per_image = matched_targets.get_field("dims")
            dims.append(dims_per_image[positive_inds])
            rotbins_per_image = matched_targets.get_field("rotbins")
            rotbins.append(rotbins_per_image[positive_inds])
            rotregs_per_image = matched_targets.get_field("rotregs")
            rotregs.append(rotregs_per_image[positive_inds])

        return labels, depths, dims, rotbins, rotregs

    def compute_res_loss(self, output, target):
        return F.smooth_l1_loss(output, target, reduction='mean')
    
    # TODO: weight
    def compute_bin_loss(self, output, target):
        # mask = mask.expand_as(output)
        # output = output * mask.float()
        return F.cross_entropy(output, target, reduction='mean')
    
    def compute_rot_loss(self, output, target_bin, target_res):
        # output: (B, 128, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos, 
        #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
        # target_bin: (B, 128, 2) [bin1_cls, bin2_cls]
        # target_res: (B, 128, 2) [bin1_res, bin2_res]
        # mask: (B, 128, 1)
        # import pdb; pdb.set_trace()
        output = output.view(-1, 8)
        target_bin = target_bin.view(-1, 2)
        target_res = target_res.view(-1, 2)
        # mask = mask.view(-1, 1)
        loss_bin1 = self.compute_bin_loss(output[:, 0:2], target_bin[:, 0])
        loss_bin2 = self.compute_bin_loss(output[:, 4:6], target_bin[:, 1])
        loss_res = torch.zeros_like(loss_bin1)
        if target_bin[:, 0].nonzero().shape[0] > 0:
            idx1 = target_bin[:, 0].nonzero()[:, 0]
            valid_output1 = torch.index_select(output, 0, idx1.long())
            valid_target_res1 = torch.index_select(target_res, 0, idx1.long())
            loss_sin1 = self.compute_res_loss(
              valid_output1[:, 2], torch.sin(valid_target_res1[:, 0]))
            loss_cos1 = self.compute_res_loss(
              valid_output1[:, 3], torch.cos(valid_target_res1[:, 0]))
            loss_res += loss_sin1 + loss_cos1
        if target_bin[:, 1].nonzero().shape[0] > 0:
            idx2 = target_bin[:, 1].nonzero()[:, 0]
            valid_output2 = torch.index_select(output, 0, idx2.long())
            valid_target_res2 = torch.index_select(target_res, 0, idx2.long())
            loss_sin2 = self.compute_res_loss(
              valid_output2[:, 6], torch.sin(valid_target_res2[:, 1]))
            loss_cos2 = self.compute_res_loss(
              valid_output2[:, 7], torch.cos(valid_target_res2[:, 1]))
            loss_res += loss_sin2 + loss_cos2
        return loss_bin1 + loss_bin2 + loss_res

    def __call__(self, proposals, logits, targets):
        """
        Arguments:
            proposals (list[BoxList])
            logits (Tensor)
            targets (list[BoxList])

        Return:
            depth_loss (Tensor): scalar tensor containing the loss
        """
        labels, depth_targets, dim_targets, rotbin_targets, rotreg_targets = self.prepare_targets(proposals, targets)
        if len(depth_targets)==0: return 0

        depth_logits, dim_logits, rot_logits = logits

        labels = cat(labels, dim=0)
        depth_targets = cat(depth_targets, dim=0)
        dim_targets = cat(dim_targets, dim=0)
        rotbin_targets = cat(rotbin_targets, dim=0)
        rotreg_targets = cat(rotreg_targets, dim=0)

        if self.cfg.MODEL.ROI_DEPTH_HEAD.REG_LOGARITHM:
            depth_targets = torch.log(depth_targets)

        depth_targets = depth_targets*self.cfg.MODEL.ROI_DEPTH_HEAD.REG_AMPLIFIER

        positive_inds = torch.nonzero(labels > 0).squeeze(1)
        labels_pos = labels[positive_inds]
        # print(len(labels), len(positive_inds), len(depth_targets))
        # print(depth_logits)

        depth_loss = smooth_l1_loss(
            depth_logits[positive_inds, labels_pos], 
            depth_targets,
            size_average=False,
            beta=1,
        )
        depth_loss = depth_loss / labels.numel()

        dim_logits = dim_logits.view(-1, dim_logits.shape[1]//3, 3)[positive_inds, labels_pos]#.view(-1)

        dim_loss = smooth_l1_loss(
            dim_logits, 
            dim_targets,
            size_average=False,
            beta=1e-6, # l1 loss
        )
        dim_loss = dim_loss / labels.numel()

        rot_logits = rot_logits.view(-1, rot_logits.shape[1]//8, 8)[positive_inds, labels_pos]#.view(-1)
        rotbin_targets = rotbin_targets.to(torch.long)
        # torch.mean (in binary_cross_entropy_with_logits) doesn't
        # accept empty tensors, so handle it separately
        if rotbin_targets.numel() == 0:
            rot_loss = rotbin_targets.sum() * 0
        else:
            rot_loss = self.compute_rot_loss(rot_logits, rotbin_targets, rotreg_targets)

        # amplify
        depth_loss *= self.cfg.MODEL.ROI_DEPTH_HEAD.LOSS_AMPLIFIER
        dim_loss *= self.cfg.MODEL.ROI_DEPTH_HEAD.LOSS_AMPLIFIER
        rot_loss *= self.cfg.MODEL.ROI_DEPTH_HEAD.LOSS_AMPLIFIER

        # print(depth_loss, dim_loss, rot_loss)

        return depth_loss + dim_loss + rot_loss


class MaskRCNNDepthLRLossComputation(object):
    def __init__(self, cfg, proposal_matcher, box_coder):
        """
        Arguments:
            proposal_matcher (Matcher)
        """
        self.cfg = cfg
        self.proposal_matcher = proposal_matcher
        self.box_coder = box_coder

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # print(match_quality_matrix, matched_idxs)
        target = target.copy_with_fields(["labels", "depths"], skip_missing=True)
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals_left, proposals_right, targets_left, targets_right):
        labels = []
        depths = []
        for proposals_left_per_image, proposals_right_per_image, targets_left_per_image, targets_right_per_image in zip(proposals_left, proposals_right, targets_left, targets_right):

            proposals_union_per_image = boxlist_union(proposals_left_per_image, proposals_right_per_image)
            
            # targets_union_per_image = targets_per_image.copy_with_fields("labels").convert("xyxy")
            # disps = targets_per_image.get_field("depths").convert("disp").depths
            # targets_union_per_image.bbox[:,0] -= disps
            targets_union_per_image = boxlist_union(targets_left_per_image, targets_right_per_image)
            targets_union_per_image._copy_extra_fields(targets_left_per_image)

            matched_targets = self.match_targets_to_proposals(
                proposals_union_per_image, targets_union_per_image
            )
            matched_idxs = matched_targets.get_field("matched_idxs")
            matched_targets_left = targets_left_per_image.copy_with_fields("labels")[matched_idxs.clamp(min=0)]
            matched_targets_right = targets_right_per_image.copy_with_fields("labels")[matched_idxs.clamp(min=0)]

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # Label background (below the low threshold)
            bg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_inds] = 0

            # Label ignore proposals (between low and high thresholds)
            ignore_inds = matched_idxs == Matcher.BETWEEN_THRESHOLDS
            labels_per_image[ignore_inds] = -1  # -1 is ignored by sampler

            # compute regression targets
            regression_targets_left_per_image = self.box_coder.encode(
                matched_targets_left.bbox, proposals_union_per_image.bbox
            )
            regression_targets_right_per_image = self.box_coder.encode(
                matched_targets_right.bbox, proposals_union_per_image.bbox
            )
            # [print(a.bbox[0]) for a in [matched_targets_left, matched_targets_right, proposals_union_per_image]]
            # print(regression_targets_left_per_image[0], regression_targets_right_per_image[0])

            regression_targets_disp = regression_targets_left_per_image[:,0] - regression_targets_right_per_image[:,0]

            labels.append(labels_per_image)
            depths.append(regression_targets_disp)

        return labels, depths

    def __call__(self, proposals_left, proposals_right, depth_logits, targets_left, targets_right, img_info=None):
        """
        Arguments:
            proposals (list[BoxList])
            depth_logits (Tensor)
            targets (list[BoxList])

        Return:
            depth_loss (Tensor): scalar tensor containing the loss
        """
        labels, depth_targets = self.prepare_targets(proposals_left, proposals_right, targets_left, targets_right)
        if len(depth_targets)==0: return 0

        # modify regression value according to baseline
        if self.cfg.MODEL.ROI_HEADS_LR.ENABLE_BASELINE_ADJUST:
            for img_info_single, target in zip(img_info, depth_targets):
                if img_info_single["camera_params"]["extrinsic"].get("baseline"):
                    baseline_modifier = self.cfg.MODEL.ROI_HEADS_LR.REFERENCE_BASELINE / img_info_single["camera_params"]["extrinsic"]["baseline"]
                    target *= baseline_modifier

        labels = cat(labels, dim=0)
        depth_targets = cat(depth_targets, dim=0)

        # if self.cfg.MODEL.ROI_DEPTH_HEAD.REG_LOGARITHM:
        #     depth_targets = torch.log(torch.abs(depth_targets)+1)

        # depth_targets = depth_targets*self.cfg.MODEL.ROI_DEPTH_HEAD.REG_AMPLIFIER

        positive_inds = torch.nonzero(labels > 0).squeeze(1)
        labels_pos = labels[positive_inds]
        # print(len(labels), len(positive_inds), len(depth_targets))
        # print(depth_logits)

        # torch.mean (in binary_cross_entropy_with_logits) doesn't
        # accept empty tensors, so handle it separately
        if depth_targets.numel() == 0:
            return depth_logits.sum() * 0

        depth_loss = smooth_l1_loss(
            depth_logits[positive_inds, labels_pos], 
            depth_targets,
            size_average=False,
            beta=1,
        )
        # depth_loss_lr = smooth_l1_loss(
        #     depth_logits_lr[positive_inds, labels_pos], 
        #     depth_logits[positive_inds, labels_pos] - depth_targets,
        #     size_average=False,
        #     beta=1,
        # )
        # depth_loss_regularizer = smooth_l1_loss(
        #     depth_logits_lr[positive_inds, labels_pos], 
        #     torch.zeros(depth_targets.shape).to(depth_logits_lr.device),
        #     size_average=False,
        #     beta=1,
        # )

        depth_loss = depth_loss / labels.numel()
        # depth_loss_lr = depth_loss_lr / labels.numel()
        # depth_loss_regularizer = depth_loss_regularizer / labels.numel() * self.cfg.MODEL.ROI_DEPTH_HEAD.LR_REGULARIZATION_AMPLIFIER # TODO: REGULARIZER_SCALE
        # print(depth_loss, depth_loss_lr, depth_loss_regularizer)
        # amplify
        depth_loss *= self.cfg.MODEL.ROI_DEPTH_HEAD.LOSS_AMPLIFIER 
        # depth_loss_lr *= self.cfg.MODEL.ROI_DEPTH_HEAD.LOSS_AMPLIFIER
        # depth_loss_regularizer *= self.cfg.MODEL.ROI_DEPTH_HEAD.LOSS_AMPLIFIER
        
        return depth_loss #, depth_loss_lr, depth_loss_regularizer

def make_roi_depth_loss_evaluator(cfg):
    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )

    if cfg.MODEL.ROI_DEPTH_HEAD.PREDICTOR == "FPNBox3dPredictor":
        loss_evaluator = MaskRCNNBox3dLossComputation(
            cfg, matcher
        )
    else:
        loss_evaluator = MaskRCNNDepthLossComputation(
            cfg, matcher
        )

    return loss_evaluator

def make_roi_depth_lr_loss_evaluator(cfg):
    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )
    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)

    loss_evaluator = MaskRCNNDepthLRLossComputation(
        cfg, matcher, box_coder
    )

    return loss_evaluator
