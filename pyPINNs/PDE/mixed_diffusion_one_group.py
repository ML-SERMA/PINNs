import torch
import numpy as np
from .pdeBase import pdeBase
import matplotlib.pyplot as plt
from ..Tools.operator import operator
import time

class mixed_diffusion_one_group(pdeBase):
    def __init__(self,domain,model,params_pde,device):
        super().__init__(domain,model,device)
        self.params_pde = params_pde
        self.stopping = False
        
    def residual_PDE(self,X):
        D = self.params_pde['func_D'](X).to(self.device)
        sigma_a = self.params_pde['func_Sigma_a'](X).to(self.device)
        s = self.params_pde['func_Source'](X).to(self.device)
        # tin=time.time()
        zeta = self.model.forward(X)
        phi = zeta[:,0:1]
        p   = zeta[:,1:]
        # pt  = 1.0/D*torch.hstack((p[:,1:2],-p[:,0:1]))
        # eq_curl = operator.div(pt,X)

        eq_c = 1.0/D*p + operator.grad(phi,X)
        eq_f = operator.div(p,X)+ sigma_a*phi -s
        eq = torch.hstack((eq_c,eq_f))
        # tout=time.time()
        # print("Elapsed: ",tout-tin," seconds")
        return eq


    # def Dirichlet_BC(self,X_bc_all):
    #     phi = self.model.forward(X_bc_all)[:,0:1]
    #     phi_bc = torch.zeros_like(phi)
    #     residu = phi-phi_bc
    #     return residu
    
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
                elif self.domain.boundary_conditions[idim][id] == 'REFLECTION':
                    p = self.model.forward(X_bc[idim][id])[:,1:]
                    n = (-1)**(id+1) # square domain
                    pn = p[:,idim:idim+1]*n
                    g_N = torch.zeros_like(pn)
                    residu = pn-g_N
                elif self.domain.boundary_conditions[idim][id] == 'VACUUM':
                    zeta = self.model.forward(X_bc[idim][id])
                    phi = zeta[:,0:1]
                    p   = zeta[:,1:]
                    # phi = self.model.forward(X_bc[idim][id])[:,0:1]
                    # p = self.model.forward(X_bc[idim][id])[:,1:]
                    n = (-1)**(id+1)
                    pn = p[:,idim:idim+1]*n
                    g_R = torch.zeros_like(phi)
                    residu = -pn+ 0.5*phi-g_R
                else:
                    # print('Check again boundary condition')
                    residu = torch.zeros_like(X_bc[idim][id])[:,0:1]
                list_residu_BC.append(residu)
        residu_BC = torch.vstack(list_residu_BC)
        return residu_BC
        
