import torch
import numpy as np
from .pdeBase import pdeBase
import matplotlib.pyplot as plt
from ..Tools.operator import operator

class DS_primal_diffusion_one_group(pdeBase):
    def __init__(self,domain,model,params_pde,device):
        super().__init__(domain,model,device)
        self.params_pde = params_pde
        
    
    def residual_PDE(self,X):
        D = self.params_pde['func_D'](X).to(self.device)
        sigma_a = self.params_pde['func_Sigma_a'](X).to(self.device)
        s = self.params_pde['func_Source'](X).to(self.device)

        z = self.model.forward(X).to(self.device)
        phi = z[:,0:1]
        D_theta = z[:,1:2]
        
        eq_D = D-D_theta
        # p = - D_theta*operator.grad(phi,X)
        # pt  = torch.hstack((p[:,1:2],-p[:,0:1]))
        # eq_curl = operator.div(pt,X)
        # eq_f = operator.div(p,X) +sigma_a*phi -s
        
        eq_f = -operator.div(D_theta*operator.grad(phi,X),X) +sigma_a*phi -s
        eq = torch.hstack((eq_D,eq_f))
        return eq



    def residual_BC(self, X_bc):
        # define residual of each boundary
        ndim = len(X_bc)
        list_residu_BC = []
        for idim in range(ndim):
            for id in range(2):
                if self.domain.boundary_conditions[idim][id] == 'ZERO_FLUX':
                    phi = self.model.forward(X_bc[idim][id])[:,0:1]
                    g_D = torch.zeros_like(phi)
                    residu = phi-g_D

                else:
                    # print('Check again boundary condition')
                    residu = torch.zeros_like(X_bc[idim][id])[:,0:1]
                list_residu_BC.append(residu)
        residu_BC = torch.vstack(list_residu_BC)
        return residu_BC
    
    # def interface_jump(self,X_interface):
    #     ndim = len(X_interface)
    #     error = 0
    #     for idim in range(ndim):
    #         phi = self.model.forward(X_interface[idim])[:,0:1]
    #         grad = operator.grad(phi,X_interface[idim])
    #         jump = 9*grad[:,idim]
    #         jump_error =  torch.mean(jump**2)
    #         # jump_error = torch.mean(torch.sum(torch.pow(9*grad,2),dim=1,keepdim=True))
    #         error = error + jump_error
    #     return error


    
  