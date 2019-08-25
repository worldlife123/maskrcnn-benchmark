# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch.nn import functional as F

from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.balanced_positive_negative_sampler import (
    BalancedPositiveNegativeSampler
)
from maskrcnn_benchmark.modeling.utils import cat

class MaskRCNNBox3dLossComputation(object):
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
        # Mask RCNN needs "labels" and "masks "fields for creating the targets
        target = target.copy_with_fields(["labels", "masks", "centers", "depths", "dims", "rotbins", "rotregs"], skip_missing=True)
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets):
        labels = []
        centers, depths, dims, rotbins, rotregs = [], [], [], [], []
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

            # compute regression targets
            # center_box = torch.cat((matched_targets.get_field("centers"), matched_targets.get_field("centers")), dim=1)
            # # print(matched_targets.bbox.shape, matched_targets.get_field("centers").shape, center_box.shape)
            # centers_per_image = self.box_coder.encode(
            #     center_box, proposals_per_image.bbox
            # )
            centers_per_image = (matched_targets.get_field("centers").bbox[:, 0:2] - proposals_per_image.bbox[:, 0:2]) / (proposals_per_image.bbox[:, 2:4] - proposals_per_image.bbox[:, 0:2])
            centers.append(centers_per_image[positive_inds])

            depths_per_image = matched_targets.get_field("depths")
            # depths_per_image[neg_inds] = 0
            depths.append(depths_per_image[positive_inds])
            dims_per_image = matched_targets.get_field("dims")
            dims.append(dims_per_image[positive_inds])
            rotbins_per_image = matched_targets.get_field("rotbins")
            rotbins.append(rotbins_per_image[positive_inds])
            rotregs_per_image = matched_targets.get_field("rotregs")
            rotregs.append(rotregs_per_image[positive_inds])

        return labels, centers, depths, dims, rotbins, rotregs

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
        labels, center_targets, depth_targets, dim_targets, rotbin_targets, rotreg_targets = self.prepare_targets(proposals, targets)
        if len(depth_targets)==0: return 0

        center_logits, depth_logits, dim_logits, rot_logits = logits

        labels = cat(labels, dim=0)
        center_targets = cat(center_targets, dim=0)
        depth_targets = cat(depth_targets, dim=0)
        dim_targets = cat(dim_targets, dim=0)
        rotbin_targets = cat(rotbin_targets, dim=0)
        rotreg_targets = cat(rotreg_targets, dim=0)

        if self.cfg.MODEL.ROI_BOX3D_HEAD.REG_LOGARITHM:
            depth_targets = torch.log(depth_targets)

        depth_targets = depth_targets*self.cfg.MODEL.ROI_BOX3D_HEAD.REG_AMPLIFIER

        positive_inds = torch.nonzero(labels > 0).squeeze(1)
        labels_pos = labels[positive_inds]
        # print(len(labels), len(positive_inds), len(depth_targets))
        # print(depth_logits)

        center_logits = center_logits.view(-1, center_logits.shape[1]//2, 2)[positive_inds, labels_pos]
        center_loss = smooth_l1_loss(
            center_logits, 
            center_targets,
            size_average=False,
            beta=1,
        )
        center_loss = center_loss / labels.numel()

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
        # depth_loss *= self.cfg.MODEL.ROI_BOX3D_HEAD.LOSS_AMPLIFIER
        # dim_loss *= self.cfg.MODEL.ROI_BOX3D_HEAD.LOSS_AMPLIFIER
        # rot_loss *= self.cfg.MODEL.ROI_BOX3D_HEAD.LOSS_AMPLIFIER

        # print(center_loss, depth_loss, dim_loss, rot_loss)

        return (depth_loss + dim_loss + rot_loss) * self.cfg.MODEL.ROI_BOX3D_HEAD.LOSS_AMPLIFIER

def make_roi_box3d_loss_evaluator(cfg):
    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )

    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)

    loss_evaluator = MaskRCNNBox3dLossComputation(
        cfg, matcher, box_coder
    )

    return loss_evaluator