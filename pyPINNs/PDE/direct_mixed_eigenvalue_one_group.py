import torch
import numpy as np
from .pdeBase import pdeBase
import matplotlib.pyplot as plt
from ..Tools.operator import operator
import copy

class direct_mixed_eigenvalue_one_group(pdeBase):
    def __init__(self,domain,model,params_pde,params_solver,device):
        super().__init__(domain,model,device)
        self.params_pde = params_pde
        self.params_solver = params_solver
        self.keff = torch.nn.Parameter(torch.tensor(1.0).float().to(device))
        # self.lamda = torch.nn.Parameter(torch.tensor(1.0).float().to(device))
        self.it = 0
        self.evoKeff = []
        self.accKeff = []
        self.C = torch.tensor([1.0]).to(device)
        self.verbose  = params_solver.get('verbose',0)
        if params_solver['keff_ref'] is not None:
            self.keff_ref = torch.tensor([params_solver['keff_ref']]).to(device)
        else:
            self.keff_ref = None


        
    def residual_PDE(self,X):
        D = self.params_pde['func_D'](X).to(self.device)
        sigma_a = self.params_pde['func_Sigma_a'](X).to(self.device)
        sigma_f = self.params_pde['func_Sigma_f'](X).to(self.device)

        # source problem
        zeta = self.model.forward(X)
        phi = zeta[:,0:1]
        p   = zeta[:,1:]

        eq_c = 1.0/D*p + operator.grad(phi,X)
        eq_f = operator.div(p,X)+ sigma_a*phi-1.0/self.keff*sigma_f*phi
        # eq_f = operator.div(p,X)+ sigma_a*phi-self.lamda*sigma_f*phi
        
        eq = torch.hstack((eq_c,eq_f))

        self.evoKeff.append(self.keff.cpu().detach().item())
        # self.evoKeff.append(1.0/self.lamda.cpu().detach().item())

        if self.keff_ref is not None:
            accKeff = torch.abs(self.keff-self.keff_ref)/torch.abs(self.keff_ref)
            # accKeff = torch.abs(1.0/self.lamda-self.keff_ref)/torch.abs(self.keff_ref)
            self.accKeff.append(accKeff.cpu().detach().item())
            
        if self.it % 2000 ==0:
            if self.verbose >= 1:
                print(f"Iter= {self.it}, \t keff = {self.keff.cpu().detach().item()}")
                # print(f"Iter= {self.it}, \t keff = {1.0/self.lamda.cpu().detach().item()}")
        self.it = self.it + 1
        return eq
    
    def loss_PDE(self,X_train):
        residu = self.residual_PDE(X_train)
        loss_f = torch.mean(torch.sum(torch.pow(residu,2),dim=1,keepdim=True))
        # Normalization loss
        phi = self.model.forward(X_train)[:,0:1]
        loss_norm = (1 - torch.mean(phi**2))**2
        # Flux positivity constraint (penalizing negative flux)
        # loss_positivity = torch.mean(torch.relu(-phi))
        # Regularization on k_eff
        loss_keff = 1.0 / self.keff + torch.log(self.keff)
        # loss_log = torch.log(1 + self.keff ** 2)
        
        # Entropy-based stabilization: k_eff log(k_eff)
        # loss_entropy = self.keff * torch.log(self.keff + 1e-6)

        #
        # Compute integrated values using trapezoidal rule
        # sigma_a = self.params_pde['func_Sigma_a'](X_train).to(self.device)
        # sigma_f = self.params_pde['func_Sigma_f'](X_train).to(self.device)
        # # Compute integrated values using 2D numerical quadrature (trapezoidal rule)
        # F_phi = torch.trapz(torch.trapz(sigma_f * phi, X_train[:,0:1], dim=1), X_train[:,1:], dim=0)
        # A_phi = torch.trapz(torch.trapz(sigma_a * phi, X_train[:,0:1], dim=1), X_train[:,1:], dim=0)
        # loss_eigen = (F_phi - self.keff * A_phi) ** 2




        total_PDE = loss_f+loss_norm+loss_keff
        
        return total_PDE


    # def Dirichlet_BC(self,X_bc):
    #     phi = self.model.forward(X_bc)[:,0:1]
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
                    print('Check again boundary condition')
                    residu = torch.zeros_like(X_bc[idim][id])
                list_residu_BC.append(residu)
        residu_BC = torch.vstack(list_residu_BC)
        return residu_BC
    
    
    # def mass_regularization(self,X):
    #     zeta = self.model.forward(X)

        
    #     N = zeta.shape.numel()
    #     # R = torch.exp(-self.lamda+0.5) + self.lamda**2 + torch.pow(torch.mean(phi)-(1/N_f)*self.C,2)
    #     R =   torch.pow(torch.mean(zeta)-(1/N)*self.C,2)
    #     return R
        











