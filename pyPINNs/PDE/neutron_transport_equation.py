
from .pdeBase import pdeBase
from ..Tools.quadrature import quadrature
import torch
import numpy as np


class NeutronTransportMultiOutput(pdeBase):
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
        self.s_train,_ = quadrature.angular_quadrature(n_mu=n_quad,n_phi=n_quad,dim_angular=dim_angular,
                                                        phi_rule='legendre',device=self.device)

    def residual_PDE(self, x):

        N = x.shape[0]
        Q_train = self.s_train.shape[0]
        Q_quad = self.s_quad.shape[0]
        G = self.num_groups

        # Repeat the angular directions for each input point
        x_in = x.repeat_interleave(Q_train, dim=0) # [N × Q_train, d]
        s_in = self.s_train.repeat(N, 1)                # [N × Q_train, d]
        input_tensor = torch.cat([x_in, s_in], dim=1)      # [N × Q_train, 2d] 

        # Evaluate the solution
        psi_flat = self.model(input_tensor) # [N_in, G]
        # psi = psi_flat.reshape(N, Q_train, G)  # [N,  Q_train, G]

        # Compute gradients
    
        grads = [torch.autograd.grad(psi_flat[:, g].sum(), x_in, create_graph=True, retain_graph=True)[0] for g in range(G)] # list of [N*Q, d]
        grad_psi = torch.stack(grads, dim=1)  # [N_in, G, d]
        # grad_psi = grad_psi.view(N, Q_train, G, -1)  # [N, Q_train,G, d]


        # Compute directional derivative ∇ψ · s for each group
        grad_dot_s = torch.einsum('ngd,nd->ng', grad_psi, s_in)  # [N_in, G]
        # grad_dot_s = torch.sum(grad_psi * s_in[ :, None, :], dim=-1)  # [N_in, G]


        # Physical terms
        Sigma_t_vals = self.Sigma_t_func(x_in)  # [N_in, G]
        Sigma_s_vals = self.Sigma_s_func(x_in)  # [N_in, G, G]
        q_vals = self.q_func(x_in)              # [N_in, G]

        # --- Compute scattering source using quadrature ---
        
        N_in = x_in.shape[0]
        x_quad = x_in.repeat_interleave(Q_quad, dim=0) # [N_in × Q_quad, d]
        s_quad = self.s_quad.repeat(N_in, 1) # # [N_in × Q_quad, d]
        w_quad = self.w_quad  # Now shape is [Q_quad]
        input_quad = torch.cat([x_quad, s_quad], dim=1) # [N_in × Q_quad, d+d]

        psi_quad = self.model(input_quad).reshape(N_in, Q_quad, G)  # [N_in, Q_quad, G]

        # Integrate psi over angular directions: [N_in, G]
        psi_integrated = torch.einsum('nqg,q->ng', psi_quad, w_quad)  # [N_in, G]
        # psi_integrated = (psi_quad * w_quad[None, :, None]).sum(dim=1)  # [N_in, G]
        

        if self.isotropic_kernel:
            psi_integrated = (1 / w_quad.sum()) *psi_integrated
            q_vals = (1 / w_quad.sum()) *q_vals
        

        # Scatter term: [N, G] = sum_g' Σ_s[g, g'] * ∫ψ_g'
        # scatter_new = torch.einsum('ngg,ng->ng', Sigma_s_vals, psi_integrated)
        scatter = torch.bmm(Sigma_s_vals, psi_integrated.unsqueeze(2)).squeeze(2)  # [N_in, G]

        # calculate the residual
        residual = grad_dot_s + Sigma_t_vals * psi_flat - scatter - q_vals # [N_in,G]

        residual = residual.reshape(N,Q_train,G) 
        return residual


    def residual_BC(self, x_b_list):
        """
        Enforce zero inflow flux on all boundary faces:
            ψ(x_b, Ω) = 0 for all Ω such that Ω ⋅ n(x_b) < 0

        Args:
            x_b_list: list of lists of tensors [dim][2], where each sub-tensor is [N, d]

        Returns:
            torch.Tensor of shape [total_N, Q, G]
        """
        residuals = []

        for dim in range(len(x_b_list)):
            for side in range(2):  # 0: min side, 1: max side
                x_b = x_b_list[dim][side]  # [N, d]
                N = x_b.shape[0]
                Q_train = self.s_train.shape[0]

                # Repeat directions and spatial points
                s_in = self.s_train.repeat(N, 1)                   # [N*Q, d]
                x_in = x_b.repeat_interleave(Q_train, dim=0)       # [N*Q, d]
                input_tensor = torch.cat([x_in, s_in], dim=1)      # [N*Q, 2d]

                psi_pred = self.model(input_tensor)                # [N*Q, G]
                psi_pred = psi_pred.reshape(N, Q_train, -1)        # [N, Q, G]

                # Compute normal vector
                n_hat = torch.zeros_like(x_b, device=self.device)
                n_hat[:, dim] = 1.0 if side == 1 else -1.0

                dot_prod = torch.einsum('qd,nd->nq', self.s_train, n_hat)  # [N, Q]
                inflow_mask = (dot_prod < 0).unsqueeze(-1)  # [N, Q, 1]

                psi_inflow = psi_pred * inflow_mask         # [N, Q, G]
                residuals.append(psi_inflow)

        return torch.cat(residuals, dim=0)  # [total_N, Q, G]
    
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
    

    def get_normal(self, x):
        N, d = x.shape
        normal = torch.zeros((N, d), device=self.device)
        normal[:, 0] = 1.0
        return normal
    
    def compute_scalar_flux(self, x):
        """
        Compute the scalar flux φ_g(x) = ∫ ψ_g(x, Ω) dΩ ≈ Σ w_q ψ_g(x, Ω_q)
        Args:
            x (Tensor): Input spatial points, shape [N, d]
        Returns:
            scalar_flux (Tensor): Scalar flux per group, shape [N, G]
        """
        # s_quad, w_quad = self.get_angular_quadrature(20)
       
        s_quad , w_quad = quadrature.angular_quadrature(n_phi=5,dim_angular=2,phi_rule='legendre',device=self.device)

        N = x.shape[0]
        Q = s_quad.shape[0]

        # Repeat inputs for all directions
        x_in = x.repeat_interleave(Q, dim=0)       # [N*Q, d]
        s_in = s_quad.repeat(N, 1)                 # [N*Q, dim]

        input_tensor = torch.cat([x_in, s_in], dim=1)  # [N*Q, d + dim_angular]

        # Model prediction: angular flux [N*Q, G]
        psi = self.model(input_tensor)
        G = psi.shape[1]
        psi = psi.view(N, Q, G)                     # [N, Q, G]
        # Scalar flux: sum over quadrature
        scalar_flux = torch.einsum('nqg,q->ng', psi, w_quad)  # [N, G]


        # # Expand weights to match psi
        # w_expanded = w_quad.view(1, Q, 1)  # [1, Q, 1]
        # # Element-wise multiply and sum over Q
        # scalar_flux_new = (psi * w_expanded).sum(dim=1)  # [N, G]


        return scalar_flux
    
    
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







