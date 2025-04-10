import torch
import numpy as np
from .pdeBase import pdeBase
import matplotlib.pyplot as plt
from ..Tools.operator import operator
import time

class curl_mixed_diffusion_one_group(pdeBase):
    def __init__(self,domain,model,params_pde,device):
        super().__init__(domain,model,device)
        self.params_pde = params_pde
        self.stopping = False
        
    def residual_PDE(self,X):
        
        # Retrieve PDE parameters
        params = self.params_pde
        D, sigma_a, s = (params['func_D'](X).to(self.device), 
                        params['func_Sigma_a'](X).to(self.device), 
                        params['func_Source'](X).to(self.device))

        # Forward pass through the model
        zeta = self.model(X)
        assert zeta.shape[1] >= 5, "Error: Model output (zeta) must have at least 5 features!"
        # Extract model outputs
        phi, px, py, qx, qy = zeta[:, 0:1], zeta[:, 1:2], zeta[:, 2:3], zeta[:, 3:4], zeta[:, 4:5]
        p = torch.hstack((px, py))  # Reconstruct vector field p

        # Compute PDE residuals
        eq_c = 1.0/D*p + operator.grad(phi,X)
        eq_f = operator.div(p,X)+ sigma_a*phi -s

        # pt  = torch.hstack((py, -px))
        # eq_curl = operator.div(pt,X)
      
        # # Compute curl residuals
        # grad_px = torch.autograd.grad(px, X, torch.ones_like(px), retain_graph=True, create_graph=True)[0]
        # grad_py = torch.autograd.grad(py, X, torch.ones_like(py), retain_graph=True, create_graph=True)[0]

        # # grad_px = torch.autograd.grad(px.sum(), X,create_graph=True, allow_unused=True)[0]
        # # grad_py = torch.autograd.grad(py.sum(), X,create_graph=True, allow_unused=True)[0]

        # # Extract specific derivatives
        # py_x = grad_py[:, 0:1]  # ∂py/∂x
        # px_y = grad_px[:, 1:2]  # ∂px/∂y
        # eq_qx, eq_qy = qx - py_x, qy - px_y
        # # eq_curl =  qx-qy
        ##=================================================
        eq_qx = qx -1.0/D*px
        eq_qy = qy -1.0/D*py
        qt  = torch.hstack((qy, -qx))
        eq_curl = operator.div(qt,X)

        eq = torch.hstack((eq_c,eq_f,eq_qx,eq_qy,eq_curl))
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
                    p = self.model.forward(X_bc[idim][id])[:,1:3]
                    n = (-1)**(id+1) # square domain
                    pn = p[:,idim:idim+1]*n
                    g_N = torch.zeros_like(pn)
                    residu = pn-g_N
                elif self.domain.boundary_conditions[idim][id] == 'VACUUM':
                    zeta = self.model.forward(X_bc[idim][id])
                    phi = zeta[:,0:1]
                    p   = zeta[:,1:3]
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
        
        
    