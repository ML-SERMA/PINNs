import torch
import numpy as np
from .pdeBase import pdeBase
import matplotlib.pyplot as plt
from ..Tools.operator import operator
import copy

class acceleration_mixed_eigenvalue_one_group(pdeBase):
    def __init__(self,domain,model,params_pde,params_solver,device):
        super().__init__(domain,model,device)
        self.params_pde = params_pde
        self.params_solver = params_solver
        self.keff = torch.tensor([1.0]).to(device)
        self.phi = torch.tensor([1.0]).to(device)
        self.Sgr = torch.tensor([1.0]).to(device)
        # self.fluxError= torch.tensor([1.0]).to(device)

        self.prevSgr = torch.tensor([1.0]).to(device)
        self.Xh = torch.tensor([1.0]).to(device)
        self.Fh = torch.tensor([1.0]).to(device)

        
        self.it = 0
        self.ite = 0
        
        self.rtol = torch.tensor([0.00001]).to(device)
        # self.rtol_min = torch.tensor([0.002]).to(device)
        self.resKeff = []
        self.resFlux = []
        self.accKeff = []
        self.evoKeff = []
        self.evoNit  = []
        self.resInnerSolver = []
        self.stopping = False
        self.do_outer = True

        self.N_inner = self.params_solver['num_inner_iters']
        self.anderson = params_solver.get('anderson',False)
        self.momentum = params_solver.get('momentum',False)
        self.verbose  = params_solver.get('verbose',0)

        if params_solver['keff_ref'] is not None:
            self.keff_ref = torch.tensor([params_solver['keff_ref']]).to(device)
        else:
            self.keff_ref = None

        
        
    def residual_PDE(self,X):
        D = self.params_pde['func_D'](X).to(self.device)
        sigma_a = self.params_pde['func_Sigma_a'](X).to(self.device)
        sigma_f = self.params_pde['func_Sigma_f'](X).to(self.device)

        if self.ite == 0:
            self.Sgr = sigma_f*torch.ones_like(X[:,0:1]).to(self.device)
            # for momentum -initialization
            if self.momentum:
                self.prevSgr = copy.copy(self.Sgr)

            # for anderson acceleration
            if self.anderson:
                self.Xh = torch.empty((self.Sgr.shape[0], 0)).to(self.device) 
                self.Fh = torch.empty((self.Sgr.shape[0], 0)).to(self.device) 

        # source problem
        zeta = self.model.forward(X)
        phi = zeta[:,0:1]
        p   = zeta[:,1:]


        eq_c = 1.0/D*p + operator.grad(phi,X)
        eq_f = operator.div(p,X)+ sigma_a*phi-self.Sgr
        eq = torch.hstack((eq_c,eq_f))
        bs = torch.hstack((torch.zeros_like(eq_c),self.Sgr))

        resInnerSolver = torch.linalg.vector_norm(eq)/torch.linalg.vector_norm(bs)
        
        self.it = self.it + 1
        # when do_outer = False, we continue to train the final source problem with 40000 step
        if self.do_outer == False :
            self.stopping = True

        if (self.it % self.N_inner == 0 or resInnerSolver <= self.rtol) and self.do_outer: 

            # update new source and keff
            S = sigma_f*phi
            keff = torch.abs( torch.sum(S*S)/torch.sum(S*self.Sgr))
            Sgr = 1.0/keff*S

            # error 
            resFlux = torch.linalg.vector_norm(phi - self.phi)/torch.linalg.vector_norm(self.phi)
            resKeff = torch.abs(keff-self.keff)/torch.abs(self.keff)
            
            # fluxError = torch.linalg.vector_norm(phi - self.phi)
            # tau = fluxError/self.fluxError

            # change inner iteration
            # self.N_inner = self.get_number_inner_iteration(loss_source)

            # self.rtol = torch.maximum(0.5*resInnerSolver,self.rtol_min)
            
            
            if resFlux <= 1.e-5 and resKeff <= 1.e-6:
                self.do_outer = False
                

            # print error
            self.ite = self.ite + 1
            if self.verbose >= 1:
                print(f"Iter= {self.ite},\t keff = {keff.cpu().detach().item()},\t resKeff = {resKeff.cpu().detach().item()},\t resFlux = {resFlux.cpu().detach().item()}, \t resInnerSolver = {resInnerSolver.cpu().detach().item()}")
                # print(f" N_inner = {self.it}, \t r_tol = {self.rtol.cpu().detach().item()}")
            
            self.resKeff.append(resKeff.cpu().detach().item())
            self.resFlux.append(resFlux.cpu().detach().item())
            self.evoKeff.append(keff.cpu().detach().item())
            self.evoNit.append(self.it)
            self.resInnerSolver.append(resInnerSolver.cpu().detach().item())

            if self.keff_ref is not None:
                accKeff = torch.abs(keff-self.keff_ref)/torch.abs(self.keff_ref)
                self.accKeff.append(accKeff.cpu().detach().item())
            
            # # Add momentum
            if self.momentum:
                beta1 = self.params_solver.get('beta1',0.)
                beta2 = self.params_solver.get('beta2',0.)
                if self.verbose == 2:
                    print(f"Apply momentum  with beta_1 = {beta1} and beta_2 = {beta2}")
                Sgr_new = Sgr + beta1*(Sgr-self.Sgr) + beta2*(Sgr-self.prevSgr)
                # update
                self.prevSgr = copy.copy(self.Sgr)

            elif self.anderson:
                
                beta = self.params_solver.get('beta',0.5)
                m=self.params_solver.get('m',4)

                if self.verbose == 2:
                    print(f"Apply Anderson with beta = {beta} and m = {m}")
                
                alpha = minimze_residual(self.Sgr,Sgr,self.Xh,self.Fh,m=m)
                # print(f"==>> alpha: {alpha}")
                X = torch.hstack([self.Xh[:,-m:],self.Sgr])
                F = torch.hstack([self.Fh[:,-m:],Sgr])
                Sgr_new = (1-beta)*(X @ alpha) + beta*(F @ alpha)
                # update history
                self.Xh = torch.hstack([self.Xh, self.Sgr])
                self.Fh = torch.hstack([self.Fh, Sgr])
                # Forget the pass
                if self.Xh.shape[1]>=20:
                    self.Xh  = self.Xh[:,-20:]
                    self.Fh  = self.Fh[:,-20:]


            else:
                Sgr_new = Sgr

            
            # Update
            self.Sgr = copy.copy(Sgr_new)
            self.phi = copy.copy(phi)
            self.keff = copy.copy(keff)
            self.it = 0
            # self.fluxError = copy.copy(fluxError)
            
        return eq


    def Dirichlet_BC(self,X_bc):
        phi = self.model.forward(X_bc)[:,0:1]
        phi_bc = torch.zeros_like(phi)
        residu = phi-phi_bc
        return residu
    
    # def get_number_inner_iteration(self,error,N_0=2000,E_0=1.e-3,factor_N=2,factor_E=2):
    #     N_inner = N_0
    #     if error <=E_0/factor_E**4:
    #         N_inner = int(N_0/factor_N**5)
    #     elif error <=E_0/factor_E**3:
    #         N_inner = int(N_0/factor_N**4)
    #     elif error <=E_0/factor_E**2:
    #         N_inner = int(N_0/factor_N**3)
    #     elif error <=E_0/factor_E:
    #         N_inner = int(N_0/factor_N**2)
    #     elif error <=E_0:
    #         N_inner = int(N_0/factor_N)
        
    #     return N_inner




# def delete_last_n_columns(a, n):
#     n_rows = a.size()[0]
#     n_cols = a.size()[1]
#     assert(n<n_cols)
#     first_cols = n_cols - n
#     mask = torch.arange(0,first_cols)
#     b = torch.index_select(a,1,mask) # Retain first few columns; delete last_n columns
#     return b   


def minimze_residual(prevS,S,Xh,Fh,m):
        Rg = Fh[:,-m:]-Xh[:,-m:]
        Gt = S -prevS
        matLG = torch.repeat_interleave(Gt,Rg.shape[1],axis=1) -Rg
        gamma= torch.linalg.lstsq(matLG,Gt).solution
        alpha = torch.vstack([gamma,1-torch.sum(gamma)])
        return alpha







