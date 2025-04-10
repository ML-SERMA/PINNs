import torch
import numpy as np
from .pdeBase import pdeBase
import matplotlib.pyplot as plt
from ..Tools.operator import operator
import copy


class mixed_eigenvalue_one_group(pdeBase):
    def __init__(self,domain,model,params_pde,params_solver,device):
        super().__init__(domain,model,device)
        self.params_pde = params_pde
        self.params_solver = params_solver
        self.keff = torch.tensor([1.0]).to(device)
        self.phi = torch.tensor([1.0]).to(device)
        self.Sgr = torch.tensor([1.0]).to(device)
        self.innerSolver= torch.tensor([1.0]).to(device)

        self.prevSgr = torch.tensor([1.0]).to(device)
        self.Xh = torch.tensor([1.0]).to(device)
        self.Fh = torch.tensor([1.0]).to(device)

        
        self.it = 0 # inner iteration
        self.ite = 0
        
        self.rtol = torch.tensor([0.00001]).to(device)
        # self.rtol_min = torch.tensor([0.002]).to(device)
        self.resKeff = []
        self.resFlux = []
        self.resIS   = []
        self.accKeff = []
        self.evoKeff = []
        self.evoNit  = []
        self.evoInnerSolver = []
        
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
                self.Xh = torch.empty((self.Sgr.shape[0], 0)).requires_grad_(False).to(self.device) 
                self.Fh = torch.empty((self.Sgr.shape[0], 0)).requires_grad_(False).to(self.device) 

        # source problem
        zeta = self.model.forward(X)
        phi = zeta[:,0:1]
        p   = zeta[:,1:]


        eq_c = 1.0/D*p + operator.grad(phi,X)
        eq_f = operator.div(p,X)+ sigma_a*phi-self.Sgr
        eq = torch.hstack((eq_c,eq_f))

        # bs = torch.hstack((torch.zeros_like(eq_c),self.Sgr))
        # innerSolver = torch.linalg.vector_norm(eq)/torch.linalg.vector_norm(bs)
        
        self.it = self.it + 1

        if self.do_outer == False :
            # stop algorithm 
            self.stopping = True
            
        return eq
    

    def update_source(self,X):
        D = self.params_pde['func_D'](X).to(self.device)
        sigma_a = self.params_pde['func_Sigma_a'](X).to(self.device)
        sigma_f = self.params_pde['func_Sigma_f'](X).to(self.device)
        zeta = self.model.forward(X)
        phi = zeta[:,0:1]
        p   = zeta[:,1:]

        eq_c = 1.0/D*p + operator.grad(phi,X)
        eq_f = operator.div(p,X)+ sigma_a*phi-self.Sgr
        eq = torch.hstack((eq_c,eq_f))
        bs = torch.hstack((torch.zeros_like(eq_c),self.Sgr))
        innerSolver = torch.linalg.vector_norm(eq)/torch.linalg.vector_norm(bs)


        # update new source and keff
        S = sigma_f*phi
        keff =  torch.sum(S*S)/torch.sum(S*self.Sgr)
        Sgr = 1.0/keff*S

        # error 
        resFlux = torch.linalg.vector_norm(phi - self.phi)/torch.linalg.vector_norm(self.phi)
        resKeff = torch.abs(keff-self.keff)/torch.abs(self.keff)
        resIS   = torch.abs(innerSolver-self.innerSolver)/torch.abs(self.innerSolver)
        
        # fluxError = torch.linalg.vector_norm(phi - self.phi)
        # tau = fluxError/self.fluxError
        # self.N_inner = self.get_number_inner_iteration(loss_source)
        # self.rtol = torch.maximum(0.5*innerSolver,self.rtol_min)
        
        
        if resFlux <= 1.e-5 and resKeff <= 1.e-6:
            # Stop outer iteration
            self.do_outer = False
            

        # print error
        self.ite = self.ite + 1
        if self.verbose >= 1:
            print(f"Iter= {self.ite},\t keff = {keff.cpu().detach().item()},\t resKeff = {resKeff.cpu().detach().item()},\t resFlux = {resFlux.cpu().detach().item()}")
            print(f" \t innerSolver = {innerSolver.cpu().detach().item()}, \t resIS ={resIS.cpu().detach().item()}")

        
        self.resKeff.append(resKeff.cpu().detach().item())
        self.resFlux.append(resFlux.cpu().detach().item())
        self.resIS.append(resIS.cpu().detach().item())
        self.evoKeff.append(keff.cpu().detach().item())
        self.evoNit.append(self.it)
        self.evoInnerSolver.append(innerSolver.cpu().detach().item())

        if self.keff_ref is not None:
            accKeff = torch.abs(keff-self.keff_ref)/torch.abs(self.keff_ref)
            self.accKeff.append(accKeff.cpu().detach().item())

        # # Add momentum
        if self.momentum:
            beta1 = self.params_solver.get('beta1',0.)
            beta2 = self.params_solver.get('beta2',0.)
            if self.verbose == 2:
                print(f"Apply momentum  with beta_1 = {beta1} and beta_2 = {beta2}")
            with torch.no_grad():
                Sgr_new = Sgr + beta1*(Sgr-self.Sgr) + beta2*(Sgr-self.prevSgr)
                self.prevSgr = copy.copy(self.Sgr)

        elif self.anderson:
            
            beta = self.params_solver.get('beta',0.5)
            m=self.params_solver.get('m',4)

            if self.verbose == 2:
                print(f"Apply Anderson with beta = {beta} and m = {m}")
            
            # alpha = minimze_residual(self.Sgr,Sgr,self.Xh,self.Fh,m=m)
            # X = torch.hstack([self.Xh[:,-m:],self.Sgr])
            # F = torch.hstack([self.Fh[:,-m:],Sgr])
            # Sgr_new = (1-beta)*(X @ alpha) + beta*(F @ alpha)
            # self.Xh = torch.hstack([self.Xh[:,-m:], self.Sgr])
            # self.Fh = torch.hstack([self.Fh[:,-m:], Sgr])

            with torch.no_grad():
                alpha = self.solve_minimze_residual(Sgr,m=m)
                self.Xh = torch.hstack([self.Xh[:,-m:], self.Sgr])
                self.Fh = torch.hstack([self.Fh[:,-m:], Sgr])
                Sgr_new = (1-beta)*(self.Xh @ alpha) + beta*(self.Fh @ alpha)


            # print('memory Xh',self.Xh.element_size() * self.Xh.nelement())
            # print('memory Fh',self.Xh.element_size() * self.Xh.nelement())
        else:
            with torch.no_grad():
                Sgr_new = Sgr

        # Update
        self.Sgr = copy.copy(Sgr_new)
        self.phi = copy.copy(phi)
        self.keff = copy.copy(keff)
        self.innerSolver = copy.copy(innerSolver)
        self.it = 0

        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # print(torch.cuda.memory_allocated()/1024**2)
            # print(torch.cuda.memory_cached()/1024**2)

        
    def solve_minimze_residual(self,Sgr,m):
        Rg = self.Fh[:,-m:]-self.Xh[:,-m:]
        Gt = Sgr -self.Sgr
        matLG = torch.repeat_interleave(Gt,Rg.shape[1],axis=1) -Rg
        gamma= torch.linalg.lstsq(matLG,Gt).solution
        alpha = torch.vstack([gamma,1-torch.sum(gamma)])

        return alpha

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
                



    
# def minimze_residual(prevS,S,Xh,Fh,m):
#         Rg = Fh[:,-m:]-Xh[:,-m:]
#         Gt = S -prevS
#         matLG = torch.repeat_interleave(Gt,Rg.shape[1],axis=1) -Rg
#         gamma= torch.linalg.lstsq(matLG,Gt).solution
#         alpha = torch.vstack([gamma,1-torch.sum(gamma)])
#         return alpha









































# class mixed_eigenvalue_one_group(pdeBase):
#     def __init__(self,domain,model,params_pde,params_solver,device):
#         super().__init__(domain,model,device)
#         self.params_pde = params_pde
#         self.params_solver = params_solver
#         self.keff = torch.tensor([1.0]).to(device)
#         self.phi = torch.tensor([1.0]).to(device)
#         self.prevS = torch.tensor([1.0]).to(device)
#         self.S = torch.tensor([1.0]).to(device)
        
#         self.it = 0
#         self.ite = 0
        
#         self.resKeff = []
#         self.resFlux = []
#         self.accKeff = []
#         self.evoKeff = []
#         if params_solver['keff_ref'] is not None:
#             self.keff_ref = torch.tensor([params_solver['keff_ref']]).to(device)
#         else:
#             self.keff_ref = None

        
#     def residual_PDE(self,X):
#         D = self.params_pde['func_D'](X).to(self.device)
#         sigma_a = self.params_pde['func_Sigma_a'](X).to(self.device)
#         sigma_f = self.params_pde['func_Sigma_f'](X).to(self.device)

#         if self.ite == 0:
#             self.S = sigma_f*torch.ones_like(X[:,0:1]).to(self.device)

#         # source problem
#         zeta = self.model.forward(X)
#         phi = zeta[:,0:1]
#         p   = zeta[:,1:]


#         eq_c = 1.0/D*p + operator.grad(phi,X)
#         eq_f = operator.div(p,X)+ sigma_a*phi-1.0/self.keff*self.S
        

#         eq = torch.hstack((eq_c,eq_f))
#         loss_source = torch.mean((eq)**2)
        
#         self.it = self.it + 1
#         # or loss_source < 1.e-4
        
#         if self.it % self.params_solver['num_inner_iters'] == 0 or loss_source <= 1.e-9: 

#             # update new source and keff
#             S = sigma_f*phi
#             keff = self.keff*torch.sum(S*S)/torch.sum(S*self.S)

#             # error 
#             resFlux = torch.linalg.norm(phi - self.phi)/torch.linalg.norm(self.phi)
#             resKeff = torch.abs(keff-self.keff)/torch.abs(self.keff)
            
#             # print error
#             self.ite = self.ite + 1
#             print(f"Iter= {self.ite},\t keff = {keff.cpu().detach().item()},\t resKeff = {resKeff.cpu().detach().item()},\t resFlux = {resFlux.cpu().detach().item()}, \t loss_source = {loss_source.cpu().detach().item()}")
            
#             self.resKeff.append(resKeff.cpu().detach().item())
#             self.resFlux.append(resFlux.cpu().detach().item())
#             self.evoKeff.append(keff.cpu().detach().item())

#             if self.keff_ref is not None:
#                 accKeff = torch.abs(keff-self.keff_ref)/torch.abs(self.keff_ref)
#                 self.accKeff.append(accKeff.cpu().detach().item())
            
#             # # Add momentum
#             if self.params_solver['momentum'] :
#                 print('Apply momentum !')
#                 S_new = S + self.params_solver['beta1']*(S-self.S) + self.params_solver['beta2']*(S-self.prevS)
#             else:
#                 S_new = S

#             self.prevS = copy.copy(self.S)
#             self.S = copy.copy(S_new)

#             self.phi = copy.copy(phi)
#             self.keff = copy.copy(keff)
            
#         return eq


#     def Dirichlet_BC(self,X_bc):
#         phi = self.model.forward(X_bc)[:,0:1]
#         phi_bc = torch.zeros_like(phi)
#         residu = phi-phi_bc
#         return residu
        











