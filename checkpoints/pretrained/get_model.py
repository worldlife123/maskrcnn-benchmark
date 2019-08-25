import torch

a = torch.load("model_final.pth")
torch.save(a['model'], "e2e_mask_rcnn_R_50_FPN_1x_cityscapes_trained.pth")

a = torch.load("e2e_mask_rcnn_R_50_FPN_1x.pth")
torch.save(a['model'], "e2e_mask_rcnn_R_50_FPN_1x_model.pth")
