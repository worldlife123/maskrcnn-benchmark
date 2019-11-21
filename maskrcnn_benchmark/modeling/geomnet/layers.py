import torch
from torch import nn
from torch.nn import functional as F

class InversePerspectiveMapping(nn.Module):
    def __init__(self, output_size, interp_mode="bilinear"):
        self.output_size = output_size
        self.interp_mode = interp_mode

    def create_grid(self, output_size, K, T):
        batch_size = K.shape[0]
        im2ground = torch.zeros(batch_size, 2, 3)
        for i in range(batch_size):
            im2ground[i] = torch.matmul(K[i], T[i])[:2,:3]

        grid = F.affine_grid(im2ground, torch.Size([batch_size, 1, output_size[1], output_size[0]])) # NCHW
        return grid

    def forward(self, features, K, T):
        grid = self.create_grid(self.output_size, K, T)
        transformed_features = F.grid_sample(features, grid, mode=self.interp_mode, padding_mode='zeros')
        return transformed_features

if __name__ == "__main__":
    a = torch.zeros((1,1,256,256))
    for i in range(256):
        for j in range(256):
            a[0,0,i,j] = i - j/256.
    l = InversePerspectiveMapping((128,256))
    K = torch.tensor([[[1,0,0],[0,1,0],[0,0,1]]])
    T = torch.tensor([[[1,0,0],[0,1,0],[0,0,1],[0,0,0]]])
    output = l(a,K,T)