import torch
import torch.nn as nn
from torch.nn import functional as F

from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.modeling.utils import cat

import numpy as np

def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()

class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)

        return cam_points


class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        P = torch.matmul(K, T)[:, :3, :]

        cam_points = torch.matmul(P, points)

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords

class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)

class DepthNetLossComputation(object):
    def __init__(self, cfg):
        self.cfg = cfg.clone()

        self.ssim = SSIM()

    def compute_reprojection_loss(self, pred, target, lambda_ssim=0.85):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = (1-lambda_ssim) * abs_diff.mean(1, True)

        ssim_loss = (lambda_ssim * self.ssim(pred, target).mean(1, True)) if lambda_ssim > 0 else 0
        reprojection_loss =  ssim_loss + l1_loss

        return reprojection_loss

    def __call__(self, x, targets):
        return 

class DepthNetLRConsistencyLossComputation(object):
    def __init__(self, cfg):
        self.cfg = cfg.clone()

        self.ssim = SSIM()

    def gradient_x(self, img):
        # Pad input to keep output size consistent
        img = F.pad(img, (0, 1, 0, 0), mode="replicate")
        gx = img[:, :, :, :-1] - img[:, :, :, 1:]  # NCHW
        return gx

    def gradient_y(self, img):
        # Pad input to keep output size consistent
        img = F.pad(img, (0, 0, 0, 1), mode="replicate")
        gy = img[:, :, :-1, :] - img[:, :, 1:, :]  # NCHW
        return gy

    def compute_disp_smoothness(self, disp, image_targets):
        
        disp_gradients_x = [self.gradient_x(d) for d in disp]
        disp_gradients_y = [self.gradient_y(d) for d in disp]

        image_gradients_x = [self.gradient_x(img) for img in image_targets]
        image_gradients_y = [self.gradient_y(img) for img in image_targets]

        weights_x = [torch.exp(-torch.mean(torch.abs(g), 1,
                     keepdim=True)) for g in image_gradients_x]
        weights_y = [torch.exp(-torch.mean(torch.abs(g), 1,
                     keepdim=True)) for g in image_gradients_y]

        smoothness_x = [disp_gradients_x[i] * weights_x[i]
                        for i in range(len(disp))]
        smoothness_y = [disp_gradients_y[i] * weights_y[i]
                        for i in range(len(disp))]

        disp_smoothness = [torch.abs(smoothness_x[i]) + torch.abs(smoothness_y[i]) for i in range(len(disp))]

        return disp_smoothness

    def compute_smooth_loss(self, preds, targets):
        """Computes the smoothness loss for a disparity image
        The color image is used for edge-aware smoothness
        """
        losses = []
        for disp, img in zip(preds, targets):
            grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
            grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

            grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
            grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

            grad_disp_x *= torch.exp(-grad_img_x)
            grad_disp_y *= torch.exp(-grad_img_y)

            losses.append(grad_disp_x.mean() + grad_disp_y.mean())

        return sum(losses)

    def compute_reprojection_loss(self, preds, targets, lambda_ssim=0.85):
        """Computes reprojection loss between a batch of predicted and target images
        """
        losses = []
        for pred, target in zip(preds, targets):
            abs_diff = torch.abs(target - pred)
            # l1_loss = abs_diff.mean()#(1, True)
            # ssim_loss = self.ssim(pred, target).mean()#(1, True)
            # reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

            l1_loss = (1-lambda_ssim) * abs_diff.mean()
            ssim_loss = (lambda_ssim * self.ssim(pred, target).mean()) if lambda_ssim > 0 else 0
            reprojection_loss =  ssim_loss + l1_loss

            losses.append(reprojection_loss)

        return sum(losses)

    def generate_images_pred(self, images, disps, images_target, invert_disp=False):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        preds = []
        for image, disp, image_target in zip(images, disps, images_target):
            # print(image.shape, disp.shape, image_target.shape)
            width, height = image_target.shape[3], image_target.shape[2]
            disp = F.interpolate(disp, [height, width], mode="bilinear", align_corners=False)

            # meshgrid = np.meshgrid(np.linspace(-1., 1., num=width), np.linspace(-1., 1., num=height), indexing='xy')
            # pix_coords = torch.from_numpy(np.stack(meshgrid, axis=0).astype(np.float32)).to(image.device)

            # # repeat for batch
            # samples = pix_coords.unsqueeze(0)
            # samples = torch.repeat_interleave(samples, image.shape[0], dim=0)
            # samples[:, 0:1, :, :] -= disp * 2 # disp range from 0-1
            
            # samples = samples.permute(0, 2, 3, 1)
            # # samples = pix_coords

            # pred = F.grid_sample(image, samples, padding_mode="border")

            batch_size = image.shape[0]
            x_base = torch.linspace(0, 1, width).repeat(batch_size,
                    height, 1).type_as(image)
            y_base = torch.linspace(0, 1, height).repeat(batch_size,
                        width, 1).transpose(1, 2).type_as(image)

            # Apply shift in X direction
            x_shifts = disp[:, 0, :, :]  # Disparity is passed in NCHW format with 1 channel
            if invert_disp: x_shifts = -x_shifts
            flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)
            # In grid_sample coordinates are assumed to be between -1 and 1
            pred = F.grid_sample(image, 2*flow_field - 1, mode='bilinear',
                                padding_mode='border')

            # print(image.shape, pred.shape)

            preds.append(pred)

        return preds


    def __call__(self, disp_left, disp_right, images_left_pyramid, images_right_pyramid):
        # TODO: support for multiscale prediction
        # print(images_left.shape, images_right.shape)

        images_warp_right = self.generate_images_pred(images_left_pyramid, disp_left, images_right_pyramid, invert_disp=True)
        images_warp_left = self.generate_images_pred(images_right_pyramid, disp_right, images_left_pyramid, invert_disp=False)
        left_image_loss = self.compute_reprojection_loss(images_warp_left, images_left_pyramid) 
        right_image_loss = self.compute_reprojection_loss(images_warp_right, images_right_pyramid)
        image_loss = (left_image_loss + right_image_loss) * self.cfg.MODEL.DEPTHNET.LOSS_IMAGE

        disp_warp_right = self.generate_images_pred(disp_left, disp_right, disp_right, invert_disp=True)
        disp_warp_left = self.generate_images_pred(disp_right, disp_left, disp_left, invert_disp=False)
        left_consistency_loss = self.compute_reprojection_loss(disp_warp_left, disp_left) 
        right_consistency_loss = self.compute_reprojection_loss(disp_warp_right, disp_right)
        lr_consistency_loss = (left_consistency_loss + right_consistency_loss) * self.cfg.MODEL.DEPTHNET.LOSS_LR_CONSISTENCY

        # disp_left_smoothness = self.compute_disp_smoothness(disp_left, images_left_pyramid)
        # disp_right_smoothness = self.compute_disp_smoothness(disp_right, images_right_pyramid)
        # disp_left_loss = [torch.mean(torch.abs(disp_left_smoothness[i])) / 2 ** i 
        #     for i in range(len(disp_left_smoothness))]
        # disp_right_loss = [torch.mean(torch.abs(disp_right_smoothness[i])) / 2 ** i 
        #     for i in range(len(disp_right_smoothness))]
        disp_left_smooth_loss = self.compute_smooth_loss(disp_left, images_left_pyramid)
        disp_right_smooth_loss = self.compute_smooth_loss(disp_right, images_right_pyramid)
        smoothness_loss = (disp_left_smooth_loss + disp_right_smooth_loss) * self.cfg.MODEL.DEPTHNET.LOSS_SMOOTHNESS

        losses = dict(image_loss=image_loss, lr_consistency_loss=lr_consistency_loss, smoothness_loss=smoothness_loss)
        return losses

class DepthNetImageLRConsistencyLossComputation(DepthNetLRConsistencyLossComputation):

    def scale_pyramid(self, img, num_scales, start_scale=1):
        scaled_imgs = []
        s = img.size()
        h = s[2]
        w = s[3]
        for i in range(num_scales):
            ratio = start_scale * (2 ** i)
            nh = h // ratio
            nw = w // ratio
            scaled_imgs.append(F.interpolate(img,
                               size=[nh, nw], mode='bilinear',
                               align_corners=True))
        return scaled_imgs

    def __call__(self, disp_left, disp_right, images_left, images_right):
        # TODO: support for multiscale prediction
        # print(images_left.shape, images_right.shape)
        images_left_pyramid = self.scale_pyramid(images_left, len(disp_left), start_scale=4)
        images_right_pyramid = self.scale_pyramid(images_right, len(disp_right), start_scale=4)

        return super(DepthNetImageLRConsistencyLossComputation, self).__call__(disp_left, disp_right, images_left_pyramid, images_right_pyramid)

class DepthNetImageFeatureLRConsistencyLossComputation(DepthNetLRConsistencyLossComputation):

    def scale_pyramid(self, img, num_scales, start_scale=1):
        scaled_imgs = []
        s = img.size()
        h = s[2]
        w = s[3]
        for i in range(num_scales):
            ratio = start_scale * (2 ** i)
            nh = h // ratio
            nw = w // ratio
            scaled_imgs.append(F.interpolate(img,
                               size=[nh, nw], mode='bilinear',
                               align_corners=True))
        return scaled_imgs

    def __call__(self, disp_left, disp_right, images_left, images_right, features_left, features_right):
        # TODO: support for multiscale prediction
        # print(images_left.shape, images_right.shape)
        images_left_pyramid = self.scale_pyramid(images_left, len(disp_left), start_scale=4)
        images_right_pyramid = self.scale_pyramid(images_right, len(disp_right), start_scale=4)

        disp_left = disp_left + disp_left # repeat 2 times
        disp_right = disp_right + disp_right
        images_left_pyramid = images_left_pyramid + features_left
        images_right_pyramid = images_right_pyramid + features_right

        return super(DepthNetImageFeatureLRConsistencyLossComputation, self).__call__(disp_left, disp_right, images_left_pyramid, images_right_pyramid)


def build_depthnet_loss(cfg):
    depth_loss_layer = DepthNetImageLRConsistencyLossComputation(cfg)
    return depth_loss_layer