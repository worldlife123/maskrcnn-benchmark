# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone, build_depthnet_decoder, build_depthnet_loss
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()
        self.cfg = cfg.clone()

        self.backbone = build_backbone(cfg)
        if cfg.MODEL.DEPTHNET_ON:
            # self.depthnet_decoder = build_depthnet_decoder(cfg)
            self.depthnet_loss = build_depthnet_loss(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        # if self.cfg.MODEL.DEPTHNET_ON:
        #     self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels * 2)
        # else:
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

        if self.cfg.MODEL.MT_ON:
            # mt layers
            self.backbone_mt = build_backbone(cfg)
            self.rpn_mt = build_rpn(cfg, self.backbone.out_channels)
            self.roi_heads_mt = build_roi_heads(cfg, self.backbone.out_channels)

            for m in [self.backbone_mt, self.rpn_mt, self.roi_heads_mt]:
                for p in m.parameters():
                    p.detach_()
                    # p.requires_grad = False

    def init_mt_params(self):
        for model, ema_model in [(self.backbone,self.backbone_mt), (self.rpn,self.rpn_mt), (self.roi_heads,self.roi_heads_mt)]:
            # ema_model_param_table = ema_model.state_dict()
            # for name, param in model.state_dict().items():
            #     ema_model_param_table[name].data.mul_(0).add(param.data)
            ema_model.load_state_dict(model.state_dict())
    
    def mt_params_to_update(self):
        if self.cfg.MODEL.MT_ON:
            # model_param_table = {m.state_dict() for m in [self.backbone, self.rpn, self.roi_heads]}
            model_params, ema_model_params = [], []
            for model, ema_model in [(self.backbone,self.backbone_mt), (self.rpn,self.rpn_mt), (self.roi_heads,self.roi_heads_mt)]:
                model_param_table = model.state_dict()
                for ema_name, ema_param in ema_model.named_parameters():
                    ema_model_params.append(ema_param)
                    model_params.append(model_param_table[ema_name])
            return model_params, ema_model_params

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        outputs = {}

        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        # TODO: temp replace mt model
        # if not self.training:
        #     self.backbone = self.backbone_mt
        #     self.rpn = self.rpn_mt
        #     self.roi_heads = self.roi_heads_mt

        images = to_image_list(images)
        features = self.backbone(images.tensors)
        features_extra = None
        disp_output = None
        if self.cfg.MODEL.DEPTHNET_ON:
            # disp_output = self.depthnet_decoder(features)
            if features.get("fpn2_out"):
                features_extra = features["fpn2_out"]
            if features.get("depth_out"):
                disps = features["depth_out"]
                disp_output = [disp[:,0:,:,:] for disp in disps]
            features = features["fpn_out"]
            # features += tuple(decoder_outputs) # concat
            # [print(f1.shape, f2.shape) for f1,f2 in zip(features, disps)]
            # disp_output = to_image_list(self.depthnet_decoder(features))
            # disp_output = disp # to_image_list(disp)
            
            # TODO: give disp a loss(supervision or lr_consistency)

        if not self.training:
            if self.cfg.TEST.TARGETS_AS_PROPOSALS:
                assert(targets is None, "ERROR: No target found! Please check cfg.TEST.INPUT_TARGETS!")
                device = targets[0].bbox.device
                proposals = [target.copy_with_fields([]) for target in targets]
                for gt_box in proposals:
                    gt_box.add_field("objectness", torch.ones(len(gt_box), device=device))
            elif self.cfg.TEST.INPUT_TARGETS:
                proposals, proposal_losses = self.rpn(images, features, targets)
            else:
                proposals, proposal_losses = self.rpn(images, features)
        else:
            proposals, proposal_losses = self.rpn(images, features, targets)

        # concat extra features to send into roi_heads
        features = dict(features=features)
        if not features_extra is None: 
            # features += features_extra
            features["features_depth"] = features_extra
            # add fpn2 to fpn
            # features = tuple([torch.cat((f1,f2), dim=1) for f1,f2 in zip(features, features_extra)])

        # print("features:")
        # [print(f.shape) for f in features]

        proposals_sampled = None
        if self.roi_heads:
            # take box samples here
            if self.training:
                with torch.no_grad():
                    proposals_sampled = self.roi_heads.box.loss_evaluator.subsample(proposals, targets)
            x, result, detector_losses = self.roi_heads(features, proposals, targets, proposals_sampled=proposals_sampled)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}


        if self.training and self.cfg.MODEL.MT_ON:
            mt_loss = dict()
            features_mt = self.backbone_mt(images.tensors)
            rpn_output_keys = []
            for key in proposal_losses:
                if key.startswith("output_"):
                    rpn_output_keys.append(key)
            if self.cfg.MODEL.MT.RPN_LOGITS_CONSISTENCY_WEIGHT > 0.:
                if self.cfg.MODEL.MT.RPN_LOGITS_FROM_STUDENT_FEATURE:
                    proposals_mt, proposal_mt_losses = self.rpn_mt(images, features["features"], targets)
                else:
                    proposals_mt, proposal_mt_losses = self.rpn_mt(images, features_mt, targets)
                mt_loss["ema_rpn_cons_loss"] = sum([(sum([F.mse_loss(f1, f2) for f1,f2 in zip(proposal_losses.pop(key), proposal_mt_losses.pop(key))]) / len(features_mt)) for key in rpn_output_keys]) / len(rpn_output_keys) * self.cfg.MODEL.MT.RPN_LOGITS_CONSISTENCY_WEIGHT
            else:
                for key in rpn_output_keys:
                    proposal_losses.pop(key)
                    
            features_mt_extra = None
            if self.cfg.MODEL.DEPTHNET_ON:
                # disp_output = self.depthnet_decoder(features)
                if features_mt.get("fpn2_out"):
                    features_mt_extra = features_mt["fpn2_out"]
                features_mt = features_mt["fpn_out"]
            # proposals_mt, proposal_losses = self.rpn_mt(images, features_mt, targets)
            # concat extra features to send into roi_heads
            features_mt = dict(features=features_mt)
            if not features_mt_extra is None: 
                # features += features_extra
                features_mt["features_depth"] = features_mt_extra
            if self.cfg.MODEL.MT.FEATURE_CONSISTENCY_WEIGHT > 0.:
                mt_loss["ema_feat_cons_loss"] =  sum([(sum([F.mse_loss(f1, f2) for f1,f2 in zip(features[key], features_mt[key])]) / len(features_mt[key])) for key in features_mt.keys()]) / len(features_mt.keys()) * self.cfg.MODEL.MT.FEATURE_CONSISTENCY_WEIGHT
            
            if self.cfg.MODEL.MT.ROI_LOGITS_CONSISTENCY_WEIGHT > 0.:
                if self.roi_heads_mt:
                    if self.cfg.MODEL.MT.ROI_LOGITS_FROM_STUDENT_FEATURE:
                        mt_x, result_mt, detector_mt_losses = self.roi_heads_mt(features, result, targets, proposals_sampled=proposals_sampled)
                    else:
                        mt_x, result_mt, detector_mt_losses = self.roi_heads_mt(features_mt, result, targets, proposals_sampled=proposals_sampled)
                else:
                    # RPN-only models don't have roi_heads
                    mt_x = features_mt
                    result_mt = proposals
                    detector_mt_losses = {}

                mt_loss["ema_roi_cons_loss"] = sum([F.mse_loss(x[key], mt_x[key]) for key in mt_x.keys()]) / len(mt_x.keys()) * self.cfg.MODEL.MT.ROI_LOGITS_CONSISTENCY_WEIGHT

            # mt_loss = dict(ema_rpn_cons_loss=ema_rpn_cons_loss, ema_roi_cons_loss=ema_roi_cons_loss, ema_feat_cons_loss=ema_feat_cons_loss)

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            if self.cfg.MODEL.MT_ON:
                losses.update(mt_loss)
            # return losses
            outputs.update(dict(losses=losses))
            # print(losses)

        if self.cfg.MODEL.DEPTHNET_ON:
            # [print(i, disp_output_left[i].shape) for i in range(len(disp_output_left))]
            # for i in range(len(result)):
            #     result[i].add_data("disparity", disp_output_left[0][i].cpu())
            # return result
            if not disp_output is None:
                outputs.update(dict(disparity=disp_output[0].cpu()))

        outputs.update(dict(output=x))
        outputs.update(dict(result=result))
        return outputs
