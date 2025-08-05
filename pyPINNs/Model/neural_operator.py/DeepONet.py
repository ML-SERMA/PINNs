import torch
import torch.nn as nn

def make_mlp(layer_dims, activation=nn.ReLU):
    """Creates an MLP based on a list of layer dimensions."""
    layers = []
    for i in range(len(layer_dims) - 1):
        layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
        if i < len(layer_dims) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)

class DeepONet(nn.Module):
    def __init__(self, 
                 branch_layer_dims,     # e.g. [10, 128, 128, 100]
                 trunk_layer_dims,      # e.g. [2, 128, 128, 2*100]
                 output_dim=1,
                 activation=nn.ReLU):
        super(DeepONet, self).__init__()

        self.output_dim = output_dim
        self.latent_dim = branch_layer_dims[-1]

        assert trunk_layer_dims[-1] == self.latent_dim * output_dim, \
            f"Last trunk layer must be {self.latent_dim * output_dim} for latent_dim={self.latent_dim} and output_dim={output_dim}"

        self.branch = make_mlp(branch_layer_dims, activation)
        self.trunk = make_mlp(trunk_layer_dims, activation)

    def forward(self, u_sensor, x_eval):
        """
        u_sensor: (B, m)
        x_eval: (B, d)
        Returns: (B, output_dim)
        """
        B = u_sensor.shape[0]
        branch_out = self.branch(u_sensor)                     # (B, latent_dim)
        trunk_out = self.trunk(x_eval).view(B, self.output_dim, self.latent_dim)  # (B, output_dim, latent_dim)
        return torch.sum(branch_out.unsqueeze(1) * trunk_out, dim=-1)             # (B, output_dim)
