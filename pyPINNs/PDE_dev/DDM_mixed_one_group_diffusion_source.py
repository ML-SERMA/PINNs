import torch
import numpy as np
from .pdeBase import pdeBase
import matplotlib.pyplot as plt
from ..Tools.operator import operator

class DDM_mixed_one_group_diffusion_source(pdeBase):
    def __init__(self,domain,model,params_pde,device):
        super().__init__(domain,model,device)
        self.params_pde = params_pde
        self.stopping = False
        self.g_bc = None
        
    def residual_PDE(self,X):
        D = self.params_pde['func_D'](X).to(self.device)
        sigma_a = self.params_pde['func_Sigma_a'](X).to(self.device)
        s = self.params_pde['func_Source'](X).to(self.device)

        zeta = self.model.forward(X)
        phi = zeta[:,0:1]
        p   = zeta[:,1:]
        # pt  = 1.0/D*torch.hstack((p[:,1:2],-p[:,0:1]))
        # eq_curl = operator.div(pt,X)

        eq_c = 1.0/D*p + operator.grad(phi,X)
        eq_f = operator.div(p,X)+ sigma_a*phi -s
        eq = torch.hstack((eq_c,eq_f))
        return eq


    # def set_Dirichlet_BC(self,X_bc,g_bc):
    #     ndim = self.domain.input_dim
    #     for idim in range(ndim):
    #         for id in range(2):
    #             if self.domain.boundary_conditions[idim][id] == 'ZERO_FLUX':
    #                 zeta = self.model.forward(X_bc[idim][id])
    #                 residu = torch.zeros_like(zeta)
    

    
    def residual_BC(self, X_bc):
        # define residual of each boundary
        ndim = len(X_bc)
        list_residu_BC = []
        for idim in range(ndim):
            for id in range(2):
                if self.domain.boundary_conditions[idim][id] == 'ZERO_FLUX':
                    zeta = self.model.forward(X_bc[idim][id])
                    residu = torch.zeros_like(zeta)
                    phi = zeta[:,0:1]
                    g_D = torch.zeros_like(phi)
                    residu[:,0:1] = phi-g_D

                elif self.domain.boundary_conditions[idim][id] == 'REFLECTION':
                    zeta = self.model.forward(X_bc[idim][id])
                    residu = torch.zeros_like(zeta)
                    p   = zeta[:,1:]
                    n = (-1)**(id+1) # square domain
                    pn = p[:,idim:idim+1]*n

                    g_N = torch.zeros_like(pn)
                    residu[:,idim+1:idim+2] = pn-g_N

                elif self.domain.boundary_conditions[idim][id] == 'VACUUM':
                    zeta = self.model.forward(X_bc[idim][id])
                    residu = torch.zeros_like(zeta)
                    phi = zeta[:,0:1]
                    p   = zeta[:,1:]

                    n = (-1)**(id+1)
                    pn = p[:,idim:idim+1]*n
                    g_R = torch.zeros_like(phi)
                    residu[:,0] = -pn+ 0.5*phi-g_R
                elif self.domain.boundary_conditions[idim][id] == 'INTERFACE':
                    zeta = self.model.forward(X_bc[idim][id])
                    residu = torch.zeros_like(zeta)

                    phi = zeta[:,0:1]
                    p   = zeta[:,1:]
                    n = (-1)**(id+1)
                    pn = p[:,idim:idim+1]*n

                    if self.g_bc[idim][id] is not None:
                        g_D = self.g_bc[idim][id][:,0:1]
                        g_N = self.g_bc[idim][id][:,idim+1:idim+2]
                        residu[:,0:1] = phi-g_D
                        residu[:,idim+1:idim+2] = pn -g_N
                        # residu[:,idim+1:idim+2] = pn +2*phi-g_N
      
                else:
                    # print('Check again boundary condition')
                    residu = torch.zeros_like(X_bc[idim][id])

                list_residu_BC.append(residu)
        residu_BC = torch.vstack(list_residu_BC)
        return residu_BC
    
    def loss_BC(self,X_bc):

        residu_bc = self.residual_BC(X_bc)
        #loss_bc = torch.mean((residu_bc)**2)
        loss_bc = torch.mean(torch.sum(torch.pow(residu_bc,2),dim=1,keepdim=True))
        return loss_bc
        
