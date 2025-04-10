import torch
import torch.nn as nn
import numpy as np
# import RBF_kernels as rbf
# from pyDOE import lhs


# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.zeros_(m.bias.data)

class Linear_Layer(nn.Module):
    def __init__(self, n_in, n_out, activation):
        super().__init__()
        self.layer = nn.Linear(n_in, n_out)
        self.activation = activation

    def forward(self, x):
        x = self.layer(x)
        if self.activation:
            x = self.activation(x)
        return x
    
    
class RBF_Layer(nn.Module):
    """
    Arguments:
        in_features: size of each input sample
        out_features: size of each output sample

    Shape:
        - Input: (N, in_features) where N is an arbitrary batch size
        - Output: (N, out_features) where N is an arbitrary batch size

    Attributes:
        centres: the learnable centres of shape (out_features, in_features).
            The values are initialised from a standard normal distribution.
            Normalising inputs to have mean 0 and standard deviation 1 is
            recommended.
        
        log_sigmas: logarithm of the learnable scaling factors of shape (out_features).
        
        basis_func: the radial basis function used to transform the scaled
            distances.
    """

    def __init__(self,ub, lb, in_features, out_features, basis_func,  order_pol, fixed_centroid = False, init_type = 'Gaussian',device=torch.device("cpu")):
        super(RBF_Layer, self).__init__()
        self.ub = ub
        self.lb = lb
        self.in_features = in_features
        self.out_features = out_features
        if fixed_centroid == False:
            self.centres = nn.Parameter(torch.Tensor(out_features, in_features))
            init_type = None
        self.log_sigmas = nn.Parameter(torch.Tensor(out_features))
        self.basis_func = basis_func
        self.order = order_pol
        
        self.reset_parameters(init_type, out_features, in_features)
        self.device = device

    def reset_parameters(self, init_type, num_centroid, dim_in):
        if init_type != None:
            if init_type == "Gaussian":
                self.centres = np.random.normal(0, 1, size=(num_centroid, dim_in))
            if init_type == "Uniform":
                self.centres = np.random.uniform(self.lb, self.ub, size=(num_centroid, dim_in))
            if init_type == "LHS":
                pass
                # self.centres = self.lb + (self.ub - self.lb) * lhs(dim_in, num_centroid)
            if init_type == "Linear":
                x = np.linspace( self.x_min,  self.x_max, int(num_centroid**0.5)).reshape(-1, 1)
                y = np.linspace(self.y_min, self.y_max, int(num_centroid**0.5)).reshape(-1, 1)
                x, y = np.meshgrid(x, y)
                x, y = x.reshape(-1, 1), y.reshape(-1, 1)
                self.centres = np.concatenate([x, y], axis=1)
            self.centres = torch.tensor(self.centres, dtype=torch.float64).to(self.device)
        else:
            nn.init.normal_(self.centres, 0, 1)
        nn.init.constant_(self.log_sigmas, 0)
        
    def poly_features(self, x, order):
        poly_feature = torch.ones(x.shape[0], 1).to(self.device)
        for i in range(1, order+1):
            poly_feature = torch.cat([poly_feature, (x**i).sum(-1).reshape(-1, 1)], dim=1)
        return poly_feature

    def forward(self, input):
        size = (input.size(0), self.out_features, self.in_features)
        x = input.unsqueeze(1).expand(size)
        c = self.centres.unsqueeze(0).expand(size)
        distances = (x - c).pow(2).sum(-1).pow(0.5) / torch.exp(self.log_sigmas).unsqueeze(0)
        rbf_feature = self.basis_func(distances)
        all_feature = rbf_feature
        if self.order != 0:
            poly_input = input
            poly_feature = self.poly_features(poly_input, self.order)
            all_feature = torch.cat([rbf_feature, poly_feature], dim=1)
        return all_feature