
from .pdeBase import pdeBase
from ..Tools.quadrature import quadrature
import torch
import numpy as np


class NeutronTransportAngularFluxScalarFlux(pdeBase):
    def __init__(self, domain, model, params_pde, n_quad=10, dim_angular=2, num_groups=1, isotropic_kernel=True, device='cpu'):
        super().__init__(domain,model, device)

        self.Sigma_t_func = params_pde['Sigma_t_func']  # Sigma_t(x) -> [N, G]
        self.Sigma_s_func = params_pde['Sigma_s_func']  # Sigma_s(x) -> [N, G, G]
        self.q_func = params_pde['q_func']              # q(x) -> [N, G]

        self.n_quad = n_quad
        self.dim_angular = dim_angular
        self.num_groups = num_groups
        self.isotropic_kernel = isotropic_kernel
        self.device = device

        self.s_quad, self.w_quad = quadrature.angular_quadrature(n_mu=n_quad,n_phi=n_quad,dim_angular=dim_angular,
                                                                phi_rule='legendre',device=self.device)
        # self.s_train,_ = quadrature.angular_quadrature(n_mu=n_quad,n_phi=n_quad,dim_angular=dim_angular,
        #                                                 phi_rule='legendre',device=self.device)

    

    def residual_PDE(self, x):
        Q_quad = self.s_quad.shape[0]
        G = self.num_groups
        N_in = x.shape[0]

        x_in = x[:, 0:self.dim_angular]  # [N_in, d]
        s_in = x[:, self.dim_angular:]  # [N_in, d]

        # ----- Model forward -----
        psi_phi = self.model(x)  # [N_in, 2G]
        psi_flat = psi_phi[:, :G]    # Angular flux ψ [N_in, G]
        phi_flat = psi_phi[:, G:]   # Scalar flux φ [N_in, G]

        # ----- Gradients -----
        grads = [
            torch.autograd.grad(
                psi_flat[:, g].sum(), x, create_graph=True, retain_graph=True, allow_unused=True
            )[0] for g in range(G)
        ]
        grads = [g[:, :self.dim_angular] if g is not None else torch.zeros_like(x_in) for g in grads]
        grad_psi = torch.stack(grads, dim=1)  # [N_in, G, d]
        grad_dot_s = torch.einsum('ngd,nd->ng', grad_psi, s_in)  # [N_in, G]

        # ----- Physical data -----
        Sigma_t_vals = self.Sigma_t_func(x_in)       # [N_in, G]
        Sigma_s_vals = self.Sigma_s_func(x_in)       # [N_in, G, G]
        q_vals = self.q_func(x_in)                   # [N_in, G]

        # ----- Angular Integration for ψ_g -----
        x_quad = x_in.repeat_interleave(Q_quad, dim=0)    # [N_in * Q_quad, d]
        s_quad = self.s_quad.repeat(N_in, 1)              # [N_in * Q_quad, d]
        input_quad = torch.cat([x_quad, s_quad], dim=1)   # [N_in * Q_quad, 2d]

        psi_phi_quad = self.model(input_quad).reshape(N_in, Q_quad, 2*G)  # [N_in, Q_uad, 2G]
        psi_quad = psi_phi_quad[:, :, :G]  # [N_in, Q_uad, G]
        # phi_quad = psi_phi_quad[:, 0, G:]  # [N_in, G] — extract once, same φ for all ω

        # Angular integration
        psi_integrated = torch.einsum('nqg,q->ng', psi_quad, self.w_quad)  # [N_in, G]

        # Normalize if isotropic
        if self.isotropic_kernel:
            psi_integrated = psi_integrated / self.w_quad.sum()
            # phi_flat       = phi_flat/self.w_quad.sum()
            q_vals = q_vals / self.w_quad.sum()

        # ----- Scattering term -----
        scatter = torch.bmm(Sigma_s_vals, phi_flat.unsqueeze(2)).squeeze(2)  # [N_in, G]

        # ----- Transport residual -----
        residual_transport = grad_dot_s + Sigma_t_vals * psi_flat -scatter - q_vals  # [N_in, G]

        # ----- Flux consistency residual: <φ> ≈ <∫ψ dΩ>
        residual_flux = phi_flat - psi_integrated  # [N_in, G]

        residual = torch.cat([residual_transport, residual_flux], dim=1)  # [N_in, 2G]

        return residual

    
    
   
    def residual_BC(self, x_b_list):
        """
        Enforce vacuum (zero inflow) boundary conditions:
            ψ(x_b, Ω) = 0 for all Ω such that Ω ⋅ n(x_b) < 0

        Args:
            x_b_list: list of lists of tensors [dim][2], each [N, 2d], where:
                    - dim = spatial dimension (x, y, ...)
                    - side = 0 (min) or 1 (max)
                    Each tensor contains [x, s] with shape [N, 2d].

        Returns:
            Tensor: concatenated boundary residuals of shape [N_total, G]
        """
        residuals = []
        d = self.dim_angular
        device = self.device
        eps = 1e-10  # to avoid floating point issues in dot products

        for dim in range(len(x_b_list)):
            for side in range(2):  # 0: min side, 1: max side
                xs_bc = x_b_list[dim][side]       # [N, 2d]
                x_bc = xs_bc[:, :d]               # [N, d]
                s_bc = xs_bc[:, d:]               # [N, d]

                # Model prediction: ψ and φ, keep only ψ for BC
                psi_phi = self.model(xs_bc)       # [N, 2G]
                psi_pred = psi_phi[:, :self.num_groups]  # [N, G]

                # Normal vector
                n_hat = torch.zeros_like(x_bc, device=device)  # [N, d]
                n_hat[:, dim] = 1.0 if side == 1 else -1.0

                # Inflow condition: s · n < 0
                dot_prod = torch.sum(s_bc * n_hat, dim=1)       # [N]
                inflow_mask = (dot_prod < -eps).unsqueeze(-1)   # [N, 1]

                # Enforce ψ = 0 for incoming directions
                psi_inflow = psi_pred * inflow_mask             # [N, G]
                residuals.append(psi_inflow)

        return torch.cat(residuals, dim=0)  # [N_total, G]
    
    def loss_PDE(self,X_train):
        residu = self.residual_PDE(X_train)
        # loss_f = torch.mean(torch.sum(residu**2, dim=-1))
        loss_f = torch.mean(torch.sum(residu**2, dim=tuple(range(1, residu.ndim))))
        return loss_f
    
    def loss_BC(self,X_bc):
        residu = self.residual_BC(X_bc)
        # loss_bc = torch.mean(torch.sum(residu**2, dim=-1))
        loss_bc = torch.mean(torch.sum(residu**2, dim=tuple(range(1, residu.ndim))))
        return loss_bc
    

    
    def compute_scalar_flux(self, x):
        """
        Compute the scalar flux φ_g(x) = ∫ ψ_g(x, Ω) dΩ ≈ Σ w_q ψ_g(x, Ω_q)
        Args:
            x (Tensor): Input spatial points, shape [N, d]
        Returns:
            scalar_flux (Tensor): Scalar flux per group, shape [N, G]
        """
        # s_quad, w_quad = self.get_angular_quadrature(20)
    
        s_quad , w_quad = quadrature.angular_quadrature(n_phi=8,dim_angular=2,phi_rule='legendre',device=self.device)

        N = x.shape[0]
        Q = s_quad.shape[0]

        # Repeat inputs for all directions
        x_in = x.repeat_interleave(Q, dim=0)       # [N*Q, d]
        s_in = s_quad.repeat(N, 1)                 # [N*Q, dim]

        input_tensor = torch.cat([x_in, s_in], dim=1)  # [N*Q, d + dim_angular]
        # psi = self.model(input_tensor)

        # Model prediction: ψ and φ, keep only ψ 
        psi_phi = self.model(input_tensor)       # [N*Q, 2G]
        psi = psi_phi[:, :self.num_groups]  # [N*Q, G]
        phi = psi_phi[:, self.num_groups:]  # [N*Q, G]

        psi = psi.view(N, Q, self.num_groups)                     # [N, Q, G]
        phi = phi.view(N, Q, self.num_groups)                     # [N, Q, G]
        # Scalar flux: sum over quadrature
        scalar_flux = torch.einsum('nqg,q->ng', psi, w_quad)  # [N, G]

        phi = phi[:,0,:] # [N, G]
        # # Expand weights to match psi
        # w_expanded = w_quad.view(1, Q, 1)  # [1, Q, 1]
        # # Element-wise multiply and sum over Q
        # scalar_flux_new = (psi * w_expanded).sum(dim=1)  # [N, G]


        return scalar_flux, phi
    
    
    def compute_angular_flux(self, x):
        """
        Returns the full angular flux tensor ψ(x, Ω) for each group.
        Output shape: [N, Q, G] where:
            N: number of spatial points
            Q: number of angular quadrature points
            G: number of energy groups
        """
        N = x.shape[0]
        Q_eval = self.s_quad.shape[0]
        G = self.num_groups

        x_in = x.repeat_interleave(Q_eval, dim=0)     # [N * Q_eval, d]
        s_in = self.s_quad.repeat(N, 1)               # [N * Q_eval, d]
        input_tensor = torch.cat([x_in, s_in], dim=1) # [N * Q_eval, 2d]

        psi = self.model(input_tensor)                # [N * Q_eval, G]
        psi = psi.view(N, Q_eval, G)                  # [N, Q, G]
        return psi
    
    def compute_group_current(self, x):
        """
        Compute neutron current J_g(x) = ∫ Ω ψ_g(x, Ω) dΩ ≈ Σ w_q Ω_q ψ_g(x, Ω_q)
        Returns:
            Tensor of shape [N, G, d]: neutron current vector for each group
        """
        psi = self.compute_angular_flux(x)  # [N, Q, G]
        s_eval = self.s_quad  # [Q, d]
        w_eval = self.w_quad.squeeze()  # [Q]

        # Multiply each direction by its weight: [Q, d]
        weighted_s = s_eval * w_eval.view(-1, 1)

        # Contract over Q (angular direction index): [N, Q, G] x [Q, d] -> [N, G, d]

        current = torch.einsum('nqg,qd->ngd', psi, weighted_s)  # [N, G, d]
        return current

    def plot_scalar_flux(self, x, group=0):
        """
        Visualize scalar flux φ(x) for a given energy group.
        Only works for 1D or 2D spatial domain.
        """
        import matplotlib.pyplot as plt
        scalar_flux = self.compute_scalar_flux(x)[:, group].detach().cpu()
        if x.shape[1] == 1:
            plt.plot(x.detach().cpu(), scalar_flux)
            plt.xlabel('x'), plt.ylabel(f'ϕ_{group}(x)')
        elif x.shape[1] == 2:
            from matplotlib.tri import Triangulation
            tri = Triangulation(x[:, 0].cpu(), x[:, 1].cpu())
            plt.tricontourf(tri, scalar_flux, levels=50)
            plt.colorbar(label=f'ϕ_{group}(x)')
        else:
            raise NotImplementedError("Only 1D or 2D supported for plotting")
        plt.title(f'Scalar Flux Group {group}')
        plt.show()







