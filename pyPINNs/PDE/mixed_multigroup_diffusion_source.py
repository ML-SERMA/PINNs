import torch
import numpy as np
from .pdeBase import pdeBase
from .mixed_multigroup_diffusion import mixed_multigroup_diffusion
import matplotlib.pyplot as plt
from ..Tools.operator import operator


class mixed_multigroup_diffusion_source(mixed_multigroup_diffusion):
    def __init__(self,domain,model,params_pde,params_solver,device):
        super().__init__(domain,model,params_pde,device)
        self.params_pde = params_pde
        self.params_solver = params_solver
        self.initialized = False
        self.scaling_loss = params_solver.get('scaling_loss',True)
        if self.scaling_loss:
            print("Using physics based scaling loss for PINNs !")

    def get_cross_sections(self,X):
        if self.params_pde.get('XsEvaluater',None):
            print(" Get cross sections from XsEvaluater !!")
            xsEvaluater = self.params_pde['XsEvaluater']
            self.D = xsEvaluater.diffusion(X)
            self.Sigma_r = xsEvaluater.sigma_r(X)
            self.Sigma_s = xsEvaluater.sigma_s(X)
            self.Sf = xsEvaluater.source(X)
        else:
            print(" get cross sections from direct functions")
            self.D = self.params_pde['func_D'](X).to(self.device)              # [N, G]
            self.Sigma_r = self.params_pde['func_Sigma_r'](X).to(self.device)  # [N, G]

            if self.params_pde.get('func_Sigma_s',None):
                self.Sigma_s = self.params_pde['func_Sigma_s'](X).to(self.device)  # [N, G, G]
            else:
                self.Sigma_s = torch.zeros(X.shape[0],1,1,device=self.device)
            # source problem
            if self.params_pde.get('func_Sf',None):
                self.Sf = self.params_pde['func_Sf'](X).to(self.device) # [N, G]  
            
            
    def residual_PDE(self, X):
        """
        Calculate the residual of the multigroup neutron diffusion  with output shape [N, G]
        """
        """
        Multigroup neutron diffusion residual using Te operator:
            T_e(g, g') = Σ_r if g = g'
                        -Σ_s(g'→g) if g ≠ g'
        """

        # self.get_cross_sections(X)
        if not self.initialized:
            self.get_cross_sections(X) # get cross sections value 
            self.initialized = True


        # Evaluate model predictions
        zeta = self.model(X)
        phi_list, p_list = self.unpack_solution(zeta,self.output_format)  # unpack φ and p
        phi = torch.cat(phi_list, dim=1)      # [N, G]
        p = torch.stack(p_list, dim=1)        # [N, G, d]

        # Te matrix: T_e(g, g') = Sigma_r if g=g' else -Sigma_s[g',g]
        Te = -self.Sigma_s.clone()                  # [N, G, G]
        Te[:, torch.arange(self.G), torch.arange(self.G)] = self.Sigma_r                      
    
        # Compute divergence of current: div(p_g)
        div_p = torch.stack([operator.div(p[:, g, :], X) for g in range(self.G)], dim=1)   # [N, G, 1]
        # removal_term = torch.einsum('nij,nj->ni', Te, phi)
        flux_constrain = div_p + torch.bmm(Te, phi.unsqueeze(-1)) - self.Sf.unsqueeze(-1)     # [N, G, 1]

        # Compute gradient constraints: 1/D * p_g + ∇phi_g
        grad_constraints = torch.stack([(p[:, g, :] / self.D[:, g:g+1]) + operator.grad(phi[:, g:g+1], X) for g in range(self.G)], dim=1)  # [N, G, d]

        
        if self.scaling_loss:
            dTe_inv_sqrt = 1.0 / torch.sqrt(self.Sigma_r + 1e-12)   # [N, G]
            flux_constrain_scaled = flux_constrain * dTe_inv_sqrt.unsqueeze(-1)  # [N, G, 1]
            grad_constraints_scaled = grad_constraints * torch.sqrt(self.D + 1e-12).unsqueeze(-1)
            residuals = torch.cat([flux_constrain_scaled, grad_constraints_scaled], dim=-1)  # [N, G, 1+d]
        else:
            residuals = torch.cat([flux_constrain, grad_constraints], dim=-1)  # [N, G, 1+d]


        # # Reconstruct total cross section: Σ_t = Σ_r + Σ_s[g→g]
        # Sigma_sself = torch.diagonal(Sigma_s, dim1=1, dim2=2)  # [N, G]
        # Sigma_t = Sigma_r + Sigma_sself                        # [N, G]
        # # Full in-scattering source: ∑_{g'} Σ_s[g'→g] * φ_{g'}
        # scatter_source = torch.bmm( Sigma_s, phi.unsqueeze(2)).squeeze(2)  # [N, G]
        # residual_flux = div_p + Sigma_t * phi - scatter_source - S  # [N, G]
        
        return residuals   # [N, G, 1+d]

    def new_residual_PDE(self, X):
        """
        Calculate the residual of the multigroup neutron diffusion (mixed form)
        using nondimensionalization. Output shape: [N, G, 1+d]
        """

        # ----------------------------
        # 1) Initialize cross sections & reference scales
        # ----------------------------
        if not self.initialized:
            # Load cross sections for the first batch
            self.get_cross_sections(X)  # fills self.D, self.Sigma_r, self.Sigma_s, self.Sf

            # ---- Compute reference scales using geometric mean ----
            eps = 1e-12  # avoid log(0)
            Sigma_r_all = self.Sigma_r.reshape(-1)
            D_all = self.D.reshape(-1)

            self.Sigma_ref = torch.exp(torch.mean(torch.log(Sigma_r_all + eps)))
            self.D_ref = torch.exp(torch.mean(torch.log(D_all + eps)))
            self.L_ref = torch.sqrt(self.D_ref / self.Sigma_ref)

            # Ensure constants are detached
            self.Sigma_ref = self.Sigma_ref.detach()
            self.D_ref = self.D_ref.detach()
            self.L_ref = self.L_ref.detach()

            self.initialized = True

        # ----------------------------
        # 2) Nondimensionalize inputs & materials
        # ----------------------------
        X_tilde = X / self.L_ref
        D_tilde = self.D / self.D_ref                      # [N, G]
        Sigma_r_tilde = self.Sigma_r / self.Sigma_ref      # [N, G]
        Sigma_s_tilde = self.Sigma_s / self.Sigma_ref      # [N, G, G]
        Sf_tilde = self.Sf / self.Sigma_ref                # [N, G]

        # ----------------------------
        # 3) Model evaluation
        # ----------------------------
        zeta = self.model(X_tilde)
        phi_list, p_list = self.unpack_solution(zeta, self.output_format)

        phi = torch.cat(phi_list, dim=1)        # [N, G]
        p = torch.stack(p_list, dim=1)          # [N, G, d]

        # ----------------------------
        # 4) Te operator (scaled)
        # ----------------------------
        Te = -Sigma_s_tilde.clone()                         # [N, G, G]
        Te[:, torch.arange(self.G), torch.arange(self.G)] = Sigma_r_tilde

        # ----------------------------
        # 5) Divergence of current (scaled coords)
        # ----------------------------
        div_p = torch.stack(
            [operator.div(p[:, g, :], X_tilde) for g in range(self.G)],
            dim=1
        )  # [N, G, 1]

        # ----------------------------
        # 6) Flux equation residual (dimensionless)
        # ----------------------------
        flux_constrain = (
            div_p
            + torch.bmm(Te, phi.unsqueeze(-1))
            - Sf_tilde.unsqueeze(-1)
        )  # [N, G, 1]

        # ----------------------------
        # 7) Current equation residual (dimensionless)
        # ----------------------------
        grad_constraints = torch.stack(
            [
                (p[:, g, :] / D_tilde[:, g:g+1]) + operator.grad(phi[:, g:g+1], X_tilde)
                for g in range(self.G)
            ],
            dim=1
        )  # [N, G, d]

        # ----------------------------
        # 8) Combine residuals
        # ----------------------------
        residuals = torch.cat([flux_constrain, grad_constraints], dim=-1)  # [N, G, 1+d]

        return residuals