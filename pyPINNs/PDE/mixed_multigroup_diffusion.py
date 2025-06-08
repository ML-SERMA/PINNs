import torch
import numpy as np
from .pdeBase import pdeBase
import matplotlib.pyplot as plt
from ..Tools.operator import operator

class mixed_multigroup_diffusion(pdeBase):
    def __init__(self,domain,model,params_pde,device):
        super().__init__(domain,model,device)
        self.params_pde = params_pde
        self.G = params_pde.get('num_groups',1)
        self.output_format = params_pde.get('output_format',"interleave")

    def unpack_solution(self, zeta,output_format="interleave"):
        """Unpack neural network output zeta into phi_list and p_list depending on output_format.

        Returns:
            phi_list: list of tensors [N,1], length G
            p_list: list of tensors [N,d], length G 
        """

        d = self.domain.input_dim
        G = self.G
        if output_format == "interleave":
            phi_list = [zeta[:, (d+1)*g : (d+1)*g + 1] for g in range(G)]
            p_list   = [zeta[:, (d+1)*g + 1 : (d+1)*(g + 1)] for g in range(G)]
                
        else:  # block format: 
            phi_list = [zeta[:, g:g+1] for g in range(G)]           #  G elements of [N, 1]
            p_list = [zeta[:, G + g*d: G + (g+1)*d]   for g in range(G)] #  Gelement of [N, d]
        return phi_list, p_list
    
    def get_cross_sections(self,X):
        if self.params_pde.get('XsEvaluater',None):
            print(" get cross sections from XsEvaluater")
            xsEvaluater = self.params_pde['XsEvaluater']
            self.D = xsEvaluater.diffusion(X)
            self.Sigma_r = xsEvaluater.sigma_r(X)
            self.Sigma_s = xsEvaluater.sigma_s(X)
            self.NuSigma_f = xsEvaluater.nusigma_f(X)
            self.chi = xsEvaluater.fission_spectrum()
        else:
            print(" get cross sections from direct functions")
            self.D = self.params_pde['func_D'](X).to(self.device)              # [N, G]
            self.Sigma_r = self.params_pde['func_Sigma_r'](X).to(self.device)  # [N, G]

            if self.params_pde.get('func_Sigma_s',None):
                self.Sigma_s = self.params_pde['func_Sigma_s'](X).to(self.device)  # [N, G, G]
            else:
                self.Sigma_s = torch.zeros(X.shape[0],1,1,device=self.device)
                
            if self.params_pde.get('func_NuSigma_f',None):
                self.NuSigma_f = self.params_pde['func_NuSigma_f'](X).to(self.device) # [N, G]

            if self.params_pde.get('func_Sf',None):
                self.Sf = self.params_pde['func_Sf'](X).to(self.device) # [N, G]
            self.chi = self.params_pde.get('chi',None).to(self.device) # [G]


    def residual_PDE(self, X):
        """
        Calculate the residual of the multigroup neutron diffusion  with output shape [N, G]
        """
        """
        Multigroup neutron diffusion residual using Te operator:
            T_e(g, g') = Σ_r if g = g'
                        -Σ_s(g'→g) if g ≠ g'
        """
       
        NotImplemented("The function Dirichlet_BC is not implemented") 
    
    
    def residual_BC(self, X_bc):
        """
        Compute boundary residuals and return them in shape [N_total, G],
        where N_total is the total number of boundary points across all faces.

        Parameters:
            X_bc: list of shape [ndim][2], each entry is a tensor of shape [N_face, d]

        Returns:
            residu_BC: tensor of shape [N_total, G]
        """

        # nunk = ndim + 1  # unknowns per group: φ + p
        list_residu_BC_groupwise = [[] for _ in range(self.G)]

        for idim in range(self.domain.input_dim):
            for id in range(2):  # 0: min, 1: max
                X_face = X_bc[idim][id]
                zeta = self.model.forward(X_face)  # shape [N_face, G * nunk]
                phi_list, p_list = self.unpack_solution(zeta,output_format=self.output_format)

                for g in range(self.G):
                    phi = phi_list[g]
                    p   = p_list[g]
                    bc_type = self.domain.boundary_conditions[idim][id]
                    if bc_type == 'ZERO_FLUX':
                        residu = phi  # φ = 0

                    elif bc_type == 'REFLECTION':
                        normal_sign = (-1) ** (id + 1)
                        pn = p[:, idim:idim+1] * normal_sign
                        residu = pn  # p · n = 0

                    elif bc_type == 'VACUUM':
                        normal_sign = (-1) ** (id + 1)
                        pn = p[:, idim:idim+1] * normal_sign
                        residu = -pn + 0.5 * phi  # Robin BC

                    else:
                        residu = torch.zeros_like(phi)  # unknown BC
                    list_residu_BC_groupwise[g].append(residu)

        # Stack residuals per group: each has shape [N_total, 1]
        group_residuals = [torch.vstack(res_list) for res_list in list_residu_BC_groupwise]

        # Concatenate along last dimension → shape [N_total, G]
        residu_BC = torch.cat(group_residuals, dim=1)
        return residu_BC  # shape [N_total, G]
    

    def loss_PDE(self,X_train):
        residu = self.residual_PDE(X_train)
        loss_f = torch.mean(torch.sum(residu**2, dim=tuple(range(1, residu.ndim))))
        return loss_f
    
    def loss_BC(self,X_bc):
        residu = self.residual_BC(X_bc)
        loss_bc = torch.mean(torch.sum(residu**2, dim=tuple(range(1, residu.ndim))))
        return loss_bc
    
