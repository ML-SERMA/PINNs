import torch
import torch.nn as nn

class ChebyKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(ChebyKANLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree

        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))
        self.register_buffer("arange", torch.arange(0, degree + 1, 1))

    def forward(self, x):
        x = torch.tanh(x)
        x = x.view((-1, self.inputdim, 1)).expand(
            -1, -1, self.degree + 1
        )  # shape = (batch_size, inputdim, self.degree + 1)
        x = x.acos()
        x *= self.arange
        x = x.cos()
        y = torch.einsum(
            "bid,iod->bo", x, self.cheby_coeffs
        )  # shape = (batch_size, outdim)
        y = y.view(-1, self.outdim)
        return y
    


class KANLayer(nn.Module):
    def __init__(self, in_features, out_features, num_knots=16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weights = nn.Parameter(torch.randn(out_features, in_features))
        self.knots = torch.linspace(0, 1, num_knots)
        self.coeffs = nn.Parameter(torch.randn(out_features, in_features, num_knots))

    def forward(self, x):
        # x: [N, in_features]
        N = x.shape[0]
        phi = torch.zeros(N, self.out_features, device=x.device)
        for i in range(self.in_features):
            xi = x[:, i].unsqueeze(1)  # [N,1]
            basis = self.b_spline(xi, self.knots)  # [N, num_knots]
            for j in range(self.out_features):
                coeff = self.coeffs[j, i, :]  # [num_knots]
                phi[:, j] += self.weights[j, i] * torch.matmul(basis, coeff)
        return phi

    def b_spline(self, x, knots):
        # Placeholder: return B-spline basis functions evaluated at x
        # You can use scipy.interpolate.BSpline or torch implementations
        pass