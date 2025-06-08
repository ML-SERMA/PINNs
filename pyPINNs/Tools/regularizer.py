
import torch


class ModelRegularizer:
    def __init__(self, weight_lasso=0.0, weight_ridge=0.0):
        self.weight_lasso = weight_lasso
        self.weight_ridge = weight_ridge

    def compute(self, model):
        reg_loss = 0.0
        if self.weight_lasso or self.weight_ridge:
            params = torch.cat([p.view(-1) for p in model.parameters()])
            if self.weight_lasso:
                reg_loss += self.weight_lasso * torch.norm(params, 1)
            if self.weight_ridge:
                reg_loss += self.weight_ridge * torch.norm(params, 2) ** 2
        return reg_loss


class PhysicsRegularizer:
    def __init__(
        self,
        weight_sobolev=0.0,
        weight_mass=0.0,
        weight_interface=0.0,
        weight_data=0.0,
    ):
        self.weight_sobolev = weight_sobolev
        self.weight_mass = weight_mass
        self.weight_interface = weight_interface
        self.weight_data = weight_data

    def compute(self, pde_model, X_train=None, X_interface=None, data=None):
        reg_loss = 0.0

        if self.weight_sobolev and X_train is not None:
            reg_loss += self.weight_sobolev * pde_model.Sobolev_norm(X_train)

        if self.weight_mass and X_train is not None:
            reg_loss += self.weight_mass * pde_model.mass_regularization(X_train)

        if self.weight_interface and X_interface is not None:
            reg_loss += self.weight_interface * pde_model.interface_jump(X_interface)

        if self.weight_data and data is not None:
            reg_loss += self.weight_data * pde_model.loss_EXP(data.X_exp, data.phi_exp)

        return reg_loss
