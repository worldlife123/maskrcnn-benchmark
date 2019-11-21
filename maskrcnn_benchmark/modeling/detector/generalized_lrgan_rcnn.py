# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone, build_depthnet_decoder, build_depthnet_loss
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads
from ..roi_heads.roi_lr_heads import build_roi_lr_heads
from ..gan.generators import build_generator
from ..gan.discriminators import build_discriminator
from ..gan.loss import build_gan_loss


class GeneralizedLRGANRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedLRGANRCNN, self).__init__()

        self.cfg = cfg.clone()
        
        self.backbone = build_backbone(cfg)
        if cfg.MODEL.DEPTHNET_ON:
            # self.depthnet_decoder = build_depthnet_decoder(cfg)
            self.depthnet_loss = build_depthnet_loss(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)# build_roi_lr_heads(cfg, self.backbone.out_channels)
        if self.training:
            self.training_state = 'D'
            self.feature_G = build_generator(cfg)
            self.feature_D = build_discriminator(cfg)
            self.gan_loss = build_gan_loss(cfg)
            # self.l1_loss = 

    def setTrainingState(self, state):
        self.training_state = state

    def get_discriminator(self):
        if self.training:
            return self.feature_D
        else:
            return None

    def forward(self, images_left, images_right=None, targets_left=None, targets_right=None):
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
            

        if self.training:
            images_right = to_image_list(images_right)
            features_right = self.backbone(images_right.tensors)
            if self.cfg.MODEL.DEPTHNET_ON:
                features_right, disp_right = features_right
                # disp_output_right = to_image_list(self.depthnet_decoder(features_right))
                disp_output_right = disp_right # to_image_list(disp_right)
            features_left_from_right = self.feature_G(features_right)
        else:
            features_right = None
            features_left_from_right = None

        if self.training and self.cfg.MODEL.DEPTHNET_ON:
            disp_output_right = [disp[:,1:2,:,:] for disp in disps]
            if not self.cfg.MODEL.DEPTHNET.FREEZE_WEIGHT:
                disp_loss = self.depthnet_loss(disp_output_left, disp_output_right, images_left.tensors, images_right.tensors)

        # use left image to train rpn
        proposals_left, proposal_losses_left = self.rpn(images_left, features_left, targets_left)
        # if self.training:
        #     proposals_right, proposal_losses_right = self.rpn(images_right, features_right, targets_right)
        # else:
        #     proposals_right = None

        if self.training:
            # gan loss
            if self.training_state == 'D':
                # self.set_requires_grad(self.feature_D, True)
                for param in self.feature_D.parameters():
                    param.requires_grad = True
                # TODO: Fake; stop backprop to the generator by detaching
                D_real = self.feature_D([f.detach() for f in features_left])
                D_fake = self.feature_D([f.detach() for f in features_left_from_right])
                loss_real = self.gan_loss(D_real, True)
                loss_fake = self.gan_loss(D_fake, False)
                # TODO: change l1 loss
                # loss_l1 = self.l1_loss(features_left, features_left_from_right) * self.cfg.MODEL.GAN.LAMBDA_L1
                gan_losses = dict(loss_GAN=(loss_real+loss_fake)*0.5)#, loss_l1=loss_l1)
            elif self.training_state == 'G':
                # self.set_requires_grad(self.feature_D, False)
                for param in self.feature_D.parameters():
                    param.requires_grad = False
                D_fake = self.feature_D(features_left_from_right)
                loss_GAN = self.gan_loss(D_fake, True)
                # TODO: change l1 loss
                # loss_l1 = self.l1_loss(features_left, features_left_from_right) * self.cfg.MODEL.GAN.LAMBDA_L1
                gan_losses = dict(loss_GAN=loss_GAN)#, loss_l1=loss_l1)


        # concat extra features to send into roi_heads
        if not features_left_extra is None: 
            features_left = features_left + features_left_extra
            features_left_from_right = features_left_from_right + features_left_extra

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

            # # run roi heads with right image
            # if self.roi_heads:
            #     x_, result_, detector_losses_right = self.roi_heads(features_right, proposals_right, targets)
            # else:
            #     # RPN-only models don't have roi_heads
            #     x_ = features_right
            #     result_ = proposals_right
            #     detector_losses_right = {}

        # run roi heads with generated left image feature(unstable)
        # if self.training:
        #     if self.roi_heads:
        #         # x, result, detector_losses = self.roi_heads(features_left, proposals_left, features_right, proposals_right, targets_left, targets_right)
        #         x, result_transformed, detector_transformed_losses_raw = self.roi_heads(features_left_from_right, proposals_left, targets_left)
        #         detector_transformed_losses = {}
        #         for key, loss in detector_transformed_losses_raw.items():
        #             detector_transformed_losses["tr_"+key] = loss
        #     else:
        #         # RPN-only models don't have roi_heads
        #         x = features_left_from_right
        #         result_transformed = proposals_left
        #         detector_transformed_losses = {}


        if self.training:
            losses = {}
            losses.update(detector_losses)
            # losses.update(detector_transformed_losses)
            losses.update(proposal_losses_left)
            losses.update(gan_losses)
            # switch training_state
            # if self.training_state == 'D':
            #     self.training_state = 'G'
            # else:
            #     self.training_state = 'D'
            # losses.update(proposal_losses_right)
            if self.cfg.MODEL.DEPTHNET_ON and not self.cfg.MODEL.DEPTHNET.FREEZE_WEIGHT:
                losses.update(disp_loss)
            # return losses
            print(losses)
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
