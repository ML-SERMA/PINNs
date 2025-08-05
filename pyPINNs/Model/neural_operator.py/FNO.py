import torch
import torch.nn as nn
import torch.fft

class SpectralConvNd(nn.Module):
    def __init__(self, in_channels, out_channels, modes, dim):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.dim = dim

        shape = (in_channels, out_channels, *modes)
        scale = 1 / (in_channels * out_channels)
        self.weight_real = nn.Parameter(scale * torch.randn(*shape))
        self.weight_imag = nn.Parameter(scale * torch.randn(*shape))

    def forward(self, x):
        # x: (B, C_in, *spatial_dims)
        B, C, *size = x.shape
        x_ft = torch.fft.rfftn(x, dim=tuple(range(2, 2 + self.dim)), norm="ortho")
        out_shape = list(x_ft.shape)
        out_shape[1] = self.out_channels
        out_ft = torch.zeros(*out_shape, dtype=torch.cfloat, device=x.device)

        # Compute truncated modes
        slices = tuple(slice(0, m) for m in self.modes)
        weight = torch.complex(self.weight_real, self.weight_imag)

        out_ft[(..., *slices)] = torch.einsum(
            'bc...,co...->bo...',
            x_ft[(..., *slices)],
            weight
        )

        x = torch.fft.irfftn(out_ft, s=size, dim=tuple(range(2, 2 + self.dim)), norm="ortho")
        return x



class FNO(nn.Module):
    def __init__(self, in_channels, out_channels, dim, modes, width=32, n_layers=4):
        """
        Parameters:
        - in_channels: input feature channels (e.g. D, Sigma_a, S)
        - out_channels: output channels (e.g. scalar flux)
        - dim: spatial dimension (1, 2, or 3)
        - modes: tuple of number of Fourier modes to keep per dimension
        - width: width of the feature space
        - n_layers: number of Fourier layers
        """
        super().__init__()
        self.dim = dim
        self.width = width

        self.in_proj = nn.Linear(in_channels, width)
        self.out_proj1 = nn.Linear(width, 128)
        self.out_proj2 = nn.Linear(128, out_channels)
        self.activation = nn.GELU()

        self.spectral_layers = nn.ModuleList([
            SpectralConvNd(width, width, modes, dim=dim) for _ in range(n_layers)
        ])

        self.pointwise_layers = nn.ModuleList([
            self._make_pointwise_conv(dim, width) for _ in range(n_layers)
        ])

    def _make_pointwise_conv(self, dim, width):
        if dim == 1:
            return nn.Conv1d(width, width, 1)
        elif dim == 2:
            return nn.Conv2d(width, width, 1)
        elif dim == 3:
            return nn.Conv3d(width, width, 1)
        else:
            raise ValueError("Only 1D, 2D, or 3D supported.")

    def forward(self, x):
        # x: (B, *spatial_dims, in_channels)
        B, *spatial_dims, C_in = x.shape
        assert len(spatial_dims) == self.dim, f"Input should be {self.dim}D"

        # Project input features to width
        x = self.in_proj(x)  # (B, *spatial_dims, width)

        # Rearrange to (B, width, *spatial_dims)
        x = x.permute(0, -1, *range(1, self.dim + 1))

        for spec, pw in zip(self.spectral_layers, self.pointwise_layers):
            x = self.activation(spec(x) + pw(x))

        # Rearrange to (B, *spatial_dims, width)
        x = x.permute(0, *range(2, self.dim + 2), 1)

        x = self.activation(self.out_proj1(x))
        x = self.out_proj2(x)
        return x



class NeutronDiffusionFNO(nn.Module):
    def __init__(self, n_groups=2, dim=2, modes=(16, 16), width=32, n_layers=4):
        super().__init__()
        # Input: D_g, Σa_g, Σs_g'g, Σf_g, χ_g
        in_channels = 2 * n_groups + (n_groups**2 - n_groups) + n_groups + n_groups  # D, Σa, Σs, Σf, χ
        out_channels = n_groups  # scalar flux per group

        self.fno = FNO(
            in_channels=in_channels,
            out_channels=out_channels,
            dim=dim,
            modes=modes,
            width=width,
            n_layers=n_layers
        )

    def forward(self, x):
        return self.fno(x)
