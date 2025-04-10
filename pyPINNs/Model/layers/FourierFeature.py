import torch 
import torch.nn as nn
import numpy as np

class FourierFeatureMapping(nn.Module):
    def __init__(self, input_dim, fourier_mapping_size, scale=1.0):
        super(FourierFeatureMapping, self).__init__()
        self.scale = scale
        self.B = nn.Parameter(torch.randn(input_dim, fourier_mapping_size) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = 2 * np.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)