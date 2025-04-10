import torch
import numpy as np
from .pdeBase import pdeBase
import matplotlib.pyplot as plt
from ..Tools.operator import operator
import time
class primal_diffusion_one_group(pdeBase):
    def __init__(self,domain,model,params_pde,device):
        super().__init__(domain,model,device)
        self.params_pde = params_pde
        
    
    def residual_PDE(self,X):
        D = self.params_pde['func_D'](X).to(self.device)
        sigma_a = self.params_pde['func_Sigma_a'](X).to(self.device)
        s = self.params_pde['func_Source'](X).to(self.device)
        # tin=time.time()
        u = self.model.forward(X).to(self.device)
        eq = -operator.div(D*operator.grad(u,X),X) +sigma_a*u -s
        # tout=time.time()
        # print("Elapsed: ",tout-tin," seconds")
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
                # elif self.domain.boundary_conditions[idim][id] == 'REFLECTION':
                #     p = self.model.forward(X_bc[idim][id])[:,1:]
                #     n = (-1)**(id+1) # square domain
                #     pn = p[:,idim:idim+1]*n
                #     g_N = torch.zeros_like(pn)
                #     residu = pn-g_N
                # elif self.domain.boundary_conditions[idim][id] == 'VACUUM':
                #     zeta = self.model.forward(X_bc[idim][id])
                #     phi = zeta[:,0:1]
                #     p   = zeta[:,1:]
                #     # phi = self.model.forward(X_bc[idim][id])[:,0:1]
                #     # p = self.model.forward(X_bc[idim][id])[:,1:]
                #     n = (-1)**(id+1)
                #     pn = p[:,idim:idim+1]*n
                #     g_R = torch.zeros_like(phi)
                #     residu = -pn+ 0.5*phi-g_R
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
    #         phi = self.model.forward(X_interface[idim])
    #         grad = operator.grad(phi,X_interface[idim])
    #         jump = 9*grad[:,idim]
    #         jump_error =  torch.mean(jump**2)
    #         # jump_error = torch.mean(torch.sum(torch.pow(9*grad,2),dim=1,keepdim=True))
    #         error = error + jump_error
    #     return error


    
    def Sobolev_norm(self,X):
        u = self.model.forward(X)
        x=X[:,0:1]
        y=X[:,1:2]
        u_x  = torch.autograd.grad(u  , x, torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_y  = torch.autograd.grad(u  , y, torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        norm = torch.mean(torch.square(u_x)+torch.square(u_xx)+torch.square(u_y)+torch.square(u_yy))
        return norm