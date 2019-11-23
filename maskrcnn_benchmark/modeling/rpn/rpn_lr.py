# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn

from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.rpn.retinanet.retinanet import build_retinanet
from .loss import make_rpn_loss_evaluator, make_rpn_lr_loss_evaluator, make_rpn_lr_hc_loss_evaluator
from .anchor_generator import make_anchor_generator
from .inference import make_rpn_postprocessor, make_rpn_lr_postprocessor
from .rpn import RPNHead

from maskrcnn_benchmark.structures.bounding_box import BoxList

@registry.RPN_HEADS.register("SingleConvRPNLRHead")
class RPNLRHead(nn.Module):
    """
    Adds a simple RPN Head with classification and regression heads
    """

    def __init__(self, cfg, in_channels, num_anchors):
        """
        Arguments:
            cfg              : config
            in_channels (int): number of channels of the input feature
            num_anchors (int): number of anchors to be predicted
        """
        super(RPNLRHead, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=1, stride=1
        )
        self.bbox_pred_right = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=1, stride=1
        )

        for l in [self.conv, self.cls_logits, self.bbox_pred, self.bbox_pred_right]:
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        logits = []
        bbox_reg = []
        bbox_pred_right = []
        for feature in x:
            t = F.relu(self.conv(feature))
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
            bbox_pred_right.append(self.bbox_pred_right(t))
        return logits, bbox_reg, bbox_pred_right


class RPNLRModule(torch.nn.Module):
    """
    Module for RPN computation. Takes feature maps from the backbone and RPN
    proposals and losses. Works for both FPN and non-FPN.
    """

    def __init__(self, cfg, in_channels):
        super(RPNLRModule, self).__init__()

        self.cfg = cfg.clone()

        anchor_generator = make_anchor_generator(cfg)

        rpn_head = registry.RPN_HEADS[cfg.MODEL.RPN.RPN_HEAD]
        head_single = rpn_head(
            cfg, in_channels, anchor_generator.num_anchors_per_location()[0]
        )
        head_dual = rpn_head(
            cfg, in_channels*2, anchor_generator.num_anchors_per_location()[0]
        ) # input is concated left-right features

        # if self.cfg.MODEL.RPN.FREEZE_WEIGHT:
        if self.cfg.MODEL.RPN_LR.MONO_HEAD_FREEZE_WEIGHT:
            for p in head_single.parameters():
                p.requires_grad = False
        if self.cfg.MODEL.RPN_LR.STEREO_HEAD_FREEZE_WEIGHT:
            for p in head_dual.parameters():
                p.requires_grad = False

        rpn_box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        box_selector_train = make_rpn_lr_postprocessor(cfg, rpn_box_coder, is_train=True)
        box_selector_test = make_rpn_lr_postprocessor(cfg, rpn_box_coder, is_train=False)

        loss_evaluator = make_rpn_lr_loss_evaluator(cfg, rpn_box_coder)
        loss_evaluator_hc = make_rpn_lr_hc_loss_evaluator(cfg)

        self.anchor_generator = anchor_generator
        if self.cfg.MODEL.RPN_LR.MONO_HEAD_ON:
            self.head_single = head_single
        if self.cfg.MODEL.RPN_LR.STEREO_HEAD_ON:
            self.head_dual = head_dual
        self.box_selector_train = box_selector_train
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator
        self.loss_evaluator_hc = loss_evaluator_hc

    def forward(self, images_left, features_left, images_right=None, features_right=None, targets_left=None, targets_right=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        anchors = self.anchor_generator(images_left, features_left)
        
        if self.cfg.MODEL.RPN_LR.STEREO_HEAD_ON and not features_right is None and (self.training or not self.cfg.MODEL.RPN_LR.MONO_HEAD_TEST):
            concat_features = [torch.cat([fl, fr], dim=1) for fl,fr in zip(features_left, features_right)]
            objectness, rpn_box_regression, rpn_box_regression_right = self.head_dual(concat_features)
            if self.cfg.MODEL.RPN_LR.MONO_HEAD_ON and self.training and self.cfg.MODEL.RPN_LR.LOSS_HEAD_CONSISTENCY > 0.0:
                objectness2, rpn_box_regression2, rpn_box_regression_right2 = self.head_single(features_left)
                loss_objectness_hc, loss_rpn_box_reg_hc, loss_rpn_box_right_reg_hc = self.loss_evaluator_hc(
                    anchors, objectness, rpn_box_regression, rpn_box_regression_right, 
                    objectness2, rpn_box_regression2, rpn_box_regression_right2,
                )
                hc_losses = {
                    "loss_objectness_hc": loss_objectness_hc,
                    "loss_rpn_box_reg_hc": loss_rpn_box_reg_hc,
                    "loss_rpn_box_right_reg_hc": loss_rpn_box_right_reg_hc,
                }
                # if self.cfg.MODEL.RPN_LR.SINGLE_HEAD_TEST:
                #     objectness, rpn_box_regression, rpn_box_regression_right = objectness2, rpn_box_regression2, rpn_box_regression_right2
        elif self.cfg.MODEL.RPN_LR.MONO_HEAD_ON:
            objectness, rpn_box_regression, rpn_box_regression_right = self.head_single(features_left)
        else:
            raise NotImplementedError("No heads for rpn_lr is given!")

        # combine targets (may vary in views)
        # combined_targets = []
        # for tl, tr in zip(targets_left, targets_right):
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
        #     combined_targets.append(BoxList(new_bbox, tl.size, mode="xyxy"))

        # the regression value is set to estimate stereo box according to the reference baseline
        # so we need to modify regression value according to baseline
        if self.cfg.MODEL.RPN_LR.ENABLE_BASELINE_ADJUST:
            for img_info, reg_left, reg_right in zip(images_left.image_infos, rpn_box_regression, rpn_box_regression_right):
                if img_info["camera_params"]["extrinsic"].get("baseline"):
                    baseline_modifier = self.cfg.MODEL.RPN_LR.REFERENCE_BASELINE / img_info["camera_params"]["extrinsic"]["baseline"]
                    disp = reg_left[:,0] - reg_right[:,0]
                    reg_right[:,0] = reg_left[:,0] - disp / baseline_modifier

        if self.training:
            boxes, boxes_right, losses = self._forward_train(anchors, objectness, rpn_box_regression, rpn_box_regression_right, targets_left, targets_right)
            if self.cfg.MODEL.RPN_LR.LOSS_HEAD_CONSISTENCY > 0.0:
                losses.update(hc_losses)
            return boxes, boxes_right, losses
        else:
            # if self.cfg.MODEL.RPN_LR.SINGLE_HEAD_TEST:
            #     objectness, rpn_box_regression, rpn_box_regression_right = objectness2, rpn_box_regression2, rpn_box_regression_right2
            return self._forward_test(anchors, objectness, rpn_box_regression, rpn_box_regression_right)

    def _forward_train(self, anchors, objectness, rpn_box_regression, rpn_box_regression_right, targets_left, targets_right):
        if self.cfg.MODEL.RPN_ONLY:
            # When training an RPN-only model, the loss is determined by the
            # predicted objectness and rpn_box_regression values and there is
            # no need to transform the anchors into predicted boxes; this is an
            # optimization that avoids the unnecessary transformation.
            boxes = anchors
        else:
            # For end-to-end models, anchors must be transformed into boxes and
            # sampled into a training batch.
            with torch.no_grad():
                # transform left target to combined
                # targets_right = []
                # for tl in targets_left:
                #     bbox_new = tl.convert("xyxy").bbox.clone().detach()
                #     disps = tl.get_field("depths").convert("disp").depths
                #     bbox_new[:, 0] -= disps
                #     bbox_new[:, 2] -= disps
                #     # print(bbox_new)
                #     bbox_new = BoxList(bbox_new, tl.size, mode="xyxy")
                #     # bbox_new.clip_to_image(remove_empty=True)
                #     targets_right.append(bbox_new)
                # boxes_right = self.box_selector_train(
                #     anchors, objectness, rpn_box_regression_right, targets_right
                # )
                boxes, boxes_right = self.box_selector_train(
                    anchors, objectness, rpn_box_regression, rpn_box_regression_right, targets_left, targets_right
                )
                # for tl, tr in zip(boxes, boxes_right):
                #     bbox_left, bbox_right = tl.convert("xyxy").bbox, tr.convert("xyxy").bbox
                #     print(bbox_left, bbox_right)
        if not self.cfg.MODEL.RPN.FREEZE_WEIGHT:
            loss_objectness, loss_rpn_box_reg, loss_rpn_box_right_reg = self.loss_evaluator(
                anchors, objectness, rpn_box_regression, rpn_box_regression_right, targets_left, targets_right
            )
            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg,
                "loss_rpn_box_right_reg": loss_rpn_box_right_reg,
            }
        else:
            losses = {}

        return boxes, boxes_right, losses

    def _forward_test(self, anchors, objectness, rpn_box_regression, rpn_box_regression_right):
        boxes, boxes_right = self.box_selector_test(anchors, objectness, rpn_box_regression, rpn_box_regression_right)
        if self.cfg.MODEL.RPN_ONLY:
            # For end-to-end models, the RPN proposals are an intermediate state
            # and don't bother to sort them in decreasing score order. For RPN-only
            # models, the proposals are the final output and we return them in
            # high-to-low confidence order.
            inds = [
                box.get_field("objectness").sort(descending=True)[1] for box in boxes
            ]
            boxes = [box[ind] for box, ind in zip(boxes, inds)]
            boxes_right = [box[ind] for box, ind in zip(boxes_right, inds)]
        return boxes, boxes_right, {}


def build_rpn_lr(cfg, in_channels):
    """
    This gives the gist of it. Not super important because it doesn't change as much
    """
    if cfg.MODEL.RETINANET_ON:
        return build_retinanet(cfg, in_channels)

    return RPNLRModule(cfg, in_channels)
