# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.structures.bounding_box import BoxList

from ..backbone import build_backbone, build_depthnet_decoder, build_depthnet_loss
from ..rpn.rpn import build_rpn
from ..rpn.rpn_lr import build_rpn_lr
from ..roi_heads.roi_heads import build_roi_heads
from ..roi_heads.roi_lr_heads import build_roi_lr_heads


class GeneralizedLRRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedLRRCNN, self).__init__()

        self.cfg = cfg.clone()
        
        self.backbone = build_backbone(cfg)
        if cfg.MODEL.DEPTHNET_ON:
            # self.depthnet_decoder = build_depthnet_decoder(cfg)
            self.depthnet_loss = build_depthnet_loss(cfg)
        if self.cfg.MODEL.USE_LR_RPN:
            self.rpn_lr = build_rpn_lr(cfg, self.backbone.out_channels)
        else:
            self.rpn = build_rpn(cfg, self.backbone.out_channels)
        
        if self.cfg.MODEL.USE_LR_ROI_HEADS:
            self.roi_heads_lr = build_roi_lr_heads(cfg, self.backbone.out_channels)
        else:
            self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)# build_roi_lr_heads(cfg, self.backbone.out_channels)

    def forward(self, images_left, targets_left=None, images_right=None, targets_right=None):
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
        if self.training and (images_right is None or targets_left is None or targets_right is None):
            raise ValueError("In training mode, images_right and targets should be passed")
        images_left = to_image_list(images_left)
        features_left = self.backbone(images_left.tensors)
        features_left_extra = None
        if self.cfg.MODEL.DEPTHNET_ON:
            if features_left.get("fpn2_out"):
                features_left_extra = features_left["fpn2_out"]
            features_left, disps = features_left["fpn_out"], features_left["depth_out"]
            # features_left += tuple(decoder_outputs) # concat
            # [print(f1.shape, f2.shape) for f1,f2 in zip(features_left, disps)]
            # disp_output_left = to_image_list(self.depthnet_decoder(features_left))
            # disp_output_left = disp_left # to_image_list(disp_left)
            disp_output_left = [disp[:,0:1,:,:] for disp in disps]
            

        if not images_right is None:
            images_right = to_image_list(images_right)
            features_right = self.backbone(images_right.tensors)
            if self.cfg.MODEL.DEPTHNET_ON:
                features_right, disp_right = features_right
                # disp_output_right = to_image_list(self.depthnet_decoder(features_right))
                disp_output_right = disp_right # to_image_list(disp_right)
        else:
            features_right = None

        if self.training and self.cfg.MODEL.DEPTHNET_ON:
            disp_output_right = [disp[:,1:2,:,:] for disp in disps]
            if not self.cfg.MODEL.DEPTHNET.FREEZE_WEIGHT:
                disp_loss = self.depthnet_loss(disp_output_left, disp_output_right, images_left.tensors, images_right.tensors)

        # use left image to train rpn
        if self.cfg.MODEL.USE_LR_RPN:
            proposals_left, proposals_right, proposal_losses = self.rpn_lr(images_left, features_left, images_right, features_right, targets_left, targets_right)
            # proposals_left = proposals
            # proposals_right = proposals
        else:
            proposals_left, proposal_losses_left = self.rpn(images_left, features_left, targets_left)
            if self.training:
                proposals_right, proposal_losses_right = self.rpn(images_right, features_right, targets_right)
            else:
                proposals_right = None
            proposal_losses = dict()
            proposal_losses.update(proposal_losses_left)
            proposal_losses.update(proposal_losses_right)

        # concat extra features to send into roi_heads
        # features_left = dict(features=features_left)
        # if not features_left_extra is None: 
        #     # features += features_extra
        #     features_left["features_depth"] = features_left_extra
        #     # add fpn2 to fpn
        #     # features = tuple([torch.cat((f1,f2), dim=1) for f1,f2 in zip(features, features_extra)])

        if self.cfg.MODEL.USE_LR_ROI_HEADS:
            x, result, detector_losses = self.roi_heads_lr(features_left, proposals_left, features_right, proposals_right, targets_left, targets_right, img_info=images_left.image_infos)
        else:
            # run roi heads with left image to train supervised
            if self.roi_heads:
                # x, result, detector_losses = self.roi_heads(features_left, proposals_left, features_right, proposals_right, targets_left, targets_right)
                x, result, detector_losses = self.roi_heads(features_left, proposals_left, targets_left)
            else:
                # RPN-only models don't have roi_heads
                x = features_left
                result = proposals_left
                detector_losses = {}
            
                # # TODO: create proposals with the estimated disparity
                # proposals_right = transform_proposals(proposals_left, result)

            if self.training:
                # run roi heads with right image
                if self.roi_heads:
                    x_, result_, detector_losses_right = self.roi_heads(features_right, proposals_right, targets_right)
                else:
                    # RPN-only models don't have roi_heads
                    x_ = features_right
                    result_ = proposals_right
                    detector_losses_right = {}

                detector_losses_right = {(k+"_right") : v for k,v in detector_losses_right.items()}
                detector_losses.update(detector_losses_right)

                # # run roi heads with union image and proposal
                # # features_union = tuple([f1+f2 for f1,f2 in zip(features_left, features_right)])
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
                # if self.roi_heads:
                #     x_, result_, detector_losses_union_left = self.roi_heads(features_left, proposals_union, targets_left)
                #     x_, result_, detector_losses_union_right = self.roi_heads(features_right, proposals_union, targets_right)
                # else:
                #     # RPN-only models don't have roi_heads
                #     x_ = features_right
                #     result_ = proposals_right
                #     detector_losses_union_left = {}
                #     detector_losses_union_right = {}

                # detector_losses_union_left = {(k+"_union_left") : v for k,v in detector_losses_union_left.items()}
                # detector_losses_union_right = {(k+"_union_right") : v for k,v in detector_losses_union_right.items()}
                # detector_losses.update(detector_losses_union_left)
                # detector_losses.update(detector_losses_union_right)

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            # losses.update(proposal_losses_left)
            # losses.update(proposal_losses_right)
            # losses.update(detector_losses_right)
            if self.cfg.MODEL.DEPTHNET_ON and not self.cfg.MODEL.DEPTHNET.FREEZE_WEIGHT:
                losses.update(disp_loss)
            # print(losses)
            # return losses
            outputs.update(dict(losses=losses))

        if self.cfg.MODEL.DEPTHNET_ON:
            # [print(i, disp_output_left[i].shape) for i in range(len(disp_output_left))]
            # for i in range(len(result)):
            #     result[i].add_data("disparity", disp_output_left[0][i].cpu())
            # return result
            outputs.update(dict(disparity=disp_output_left[0].cpu()))
        
        # return result
        outputs.update(dict(result=result))
        return outputs
