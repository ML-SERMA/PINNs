import torch
import numpy as np
from .pdeBase import pdeBase
from .mixed_multigroup_diffusion import mixed_multigroup_diffusion
import matplotlib.pyplot as plt
from ..Tools.operator import operator


class mixed_multigroup_diffusion_source(mixed_multigroup_diffusion):
    def __init__(self,domain,model,params_pde,device):
        super().__init__(domain,model,params_pde,device)
        self.params_pde = params_pde
            
    def residual_PDE(self, X):
        """
        Calculate the residual of the multigroup neutron diffusion  with output shape [N, G]
        """
        """
        Multigroup neutron diffusion residual using Te operator:
            T_e(g, g') = Σ_r if g = g'
                        -Σ_s(g'→g) if g ≠ g'
        """
        # Get cross sections
        # D = self.params_pde['func_D'](X).to(self.device)               # [N, G]
        # Sigma_r = self.params_pde['func_Sigma_r'](X).to(self.device)   # [N, G]
        # Sigma_s = self.params_pde['func_Sigma_s'](X).to(self.device)   # [N, G, G]
        # Sf = self.params_pde['func_Sf'](X).to(self.device)               # [N, G]

        self.get_cross_sections(X)

        # Evaluate model predictions
        zeta = self.model(X)
        phi_list, p_list = self.unpack_solution(zeta)  # unpack φ and p
        phi = torch.cat(phi_list, dim=1)      # [N, G]
        p = torch.stack(p_list, dim=1)        # [N, G, d]

        # Te matrix: T_e(g, g') = Sigma_r if g=g' else -Sigma_s[g',g]
        Te = -self.Sigma_s.clone()                  # [N, G, G]
        idx = torch.arange(self.G)
        Te[:, idx, idx] = self.Sigma_r                        
      
        # Compute divergence of current: div(p_g)
        div_p = torch.stack([operator.div(p[:, g, :], X) for g in range(self.G)], dim=1).squeeze(-1)   # [N, G]
        # Compute gradient constraints: 1/D * p_g + ∇phi_g
        grad_constraints = torch.stack([(p[:, g, :] / self.D[:, g:g+1]) + operator.grad(phi[:, g:g+1], X) for g in range(self.G)], dim=1)  # [N, G, d]

        # # Reconstruct total cross section: Σ_t = Σ_r + Σ_s[g→g]
        # Sigma_sself = torch.diagonal(Sigma_s, dim1=1, dim2=2)  # [N, G]
        # Sigma_t = Sigma_r + Sigma_sself                        # [N, G]
        # # Full in-scattering source: ∑_{g'} Σ_s[g'→g] * φ_{g'}
        # scatter_source = torch.bmm( Sigma_s, phi.unsqueeze(2)).squeeze(2)  # [N, G]
        # residual_flux = div_p + Sigma_t * phi - scatter_source - S  # [N, G]
        
     

        # Matrix-vector product: T_e φ
        # Te_phi =  torch.einsum('nij,nj->ni', Te, phi)     # [N, G]
        Te_phi = torch.bmm(Te, phi.unsqueeze(2)).squeeze(2)
        residual_flux = div_p + Te_phi - self.Sf              # [N, G]
       

        flux_constraints = residual_flux.unsqueeze(-1) # [N, G, 1]
        residuals = torch.cat([flux_constraints, grad_constraints], dim=-1)  # [N, G, d+1]
        return residuals   # [N, G, d+1]
    

 
    
    