import torch
import numpy as np
from .pdeBase import pdeBase
import matplotlib.pyplot as plt
from ..Tools.operator import operator
from ..Tools.multiGroupLoss import multiGroupLoss

class mixed_diffusion_two_group(pdeBase):
    def __init__(self,domain,model,params_pde,device):
        super().__init__(domain,model,device)
        self.params_pde = params_pde
        self.loss_function = multiGroupLoss(num_groups=2, method="weighted").to(device)
        
    def residual_PDE(self,X):
        D1 = self.params_pde['func_D1'](X).to(self.device)
        D2 = self.params_pde['func_D2'](X).to(self.device)
        Sigma_a1  = self.params_pde['func_Sigma_a1'](X).to(self.device)
        Sigma_a2  = self.params_pde['func_Sigma_a2'](X).to(self.device)
        Sigma_s12 = self.params_pde['func_Sigma_s12'](X).to(self.device)
        S1 = self.params_pde['func_S1'](X).to(self.device)
        S2 = self.params_pde['func_S2'](X).to(self.device)
        Sigma_r1 = Sigma_a1+Sigma_s12

        # We have to ensure all variables are 2D tensor!!!
        zeta = self.model.forward(X)
        phi1 = zeta[:,0:1]
        p1   = zeta[:,1:3]
        phi2 = zeta[:,3:4]
        p2   = zeta[:,4:6]

        eq_g1_f = operator.div(p1,X)+ Sigma_r1*phi1 -S1
        eq_g1_c = 1.0/D1*p1 + operator.grad(phi1,X)
        
        eq_g2_f =  operator.div(p2,X)+ Sigma_a2*phi2 -Sigma_s12*phi1-S2
        eq_g2_c = 1.0/D2*p2 + operator.grad(phi2,X)
        
        eq = torch.hstack((eq_g1_c,eq_g1_f,eq_g2_c,eq_g2_f))
        
        return eq
    
    def loss_PDE(self,X_train):
        residu = self.residual_PDE(X_train)

        loss_f1 = torch.mean(torch.sum(torch.pow(residu[:,0:3],2),dim=1,keepdim=True))
        loss_f2 = torch.mean(torch.sum(torch.pow(residu[:,3:6],2),dim=1,keepdim=True))
        loss_terms = [loss_f1, loss_f2]

        loss_f = self.loss_function(loss_terms,model=self.model)
        # update weights 
        # self.loss_function.update_weights(loss_f)
        return loss_f


    # def Dirichlet_BC(self,X_bc):
    #     phi_g1 = self.model.forward(X_bc)[:,0:1]
    #     phiBc_g1 = torch.zeros_like(phi_g1)
    #     residu_g1 = phi_g1-phiBc_g1

    #     phi_g2 = self.model.forward(X_bc)[:,3:4]
    #     phiBc_g2 = torch.zeros_like(phi_g2)
    #     residu_g2 = phi_g2-phiBc_g2
    #     residu = torch.hstack((residu_g1,residu_g2))

        # return residu
    
    def residual_BC(self, X_bc):
        # define residual of each boundary
        ngroup=2
        ndim = len(X_bc)
        nunk = ndim+1 # the number of unknowns for one group
        list_residu_BC = []
        for ig in range(ngroup):
            # print(f"==>> ig: {ig}")
            list_residu_BC_ig = []
            for idim in range(ndim):
                for id in range(2):
                    # print(f'index flux: {nunk*ig}:{nunk*ig+1}')
                    # print(f'index current: {nunk*ig+1}:{nunk*ig+nunk}')
                    if self.domain.boundary_conditions[idim][id] == 'ZERO_FLUX':
                        phi = self.model.forward(X_bc[idim][id])[:,nunk*ig:nunk*ig+1]
                        g_D = torch.zeros_like(phi)
                        residu = phi-g_D
                    elif self.domain.boundary_conditions[idim][id] == 'REFLECTION':
                        p = self.model.forward(X_bc[idim][id])[:,nunk*ig+1:nunk*ig+nunk]
                        n = (-1)**(id+1) # square domain
                        pn = p[:,idim:idim+1]*n
                        g_N = torch.zeros_like(pn)
                        residu = pn-g_N
                    elif self.domain.boundary_conditions[idim][id] == 'VACUUM':
                        zeta = self.model.forward(X_bc[idim][id])
                        phi = zeta[:, nunk*ig:nunk*ig+1]
                        p   = zeta[:, nunk*ig+1:nunk*ig+nunk]
                        n = (-1)**(id+1)
                        pn = p[:,idim:idim+1]*n
                        g_R = torch.zeros_like(phi)
                        residu = -pn+ 0.5*phi-g_R
                    else:
                        # print('Check again boundary condition')
                        residu = torch.zeros_like(X_bc[idim][id])[:,0:1]
                    list_residu_BC_ig.append(residu)
            residu_BC_ig = torch.vstack(list_residu_BC_ig)
            # print(f"==>> residu_BC_ig: {residu_BC_ig.shape}")
            list_residu_BC.append(residu_BC_ig)
        residu_BC = torch.vstack(list_residu_BC)
        
        return residu_BC
        
