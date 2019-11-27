import torch

a = torch.load("model_final.pth")
torch.save(a['model'], "e2e_lr_rpn_mask_rcnn_R_50_FPN_1x_kitti_trained.pth")

