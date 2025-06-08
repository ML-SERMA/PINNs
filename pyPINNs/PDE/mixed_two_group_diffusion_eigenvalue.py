import torch
import numpy as np
from .pdeBase import pdeBase
import matplotlib.pyplot as plt
from ..Tools.operator import operator
import copy




class mixed_two_group_diffusion_eigenvalue(pdeBase):
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
        

        D1 = self.params_pde['func_D1'](X).to(self.device)
        D2 = self.params_pde['func_D2'](X).to(self.device)
        Sigma_a1  = self.params_pde['func_Sigma_a1'](X).to(self.device)
        Sigma_a2  = self.params_pde['func_Sigma_a2'](X).to(self.device)
        Sigma_s12 = self.params_pde['func_Sigma_s12'](X).to(self.device)
        Sigma_f1 = self.params_pde['func_Sigma_f1'](X).to(self.device)
        Sigma_f2 = self.params_pde['func_Sigma_f2'](X).to(self.device)
        Sigma_r1 = Sigma_a1+Sigma_s12


        if self.ite == 0:
            # self.Sgr = (Sigma_f1+Sigma_f2)*torch.ones_like(X[:,0:1]).to(self.device)
            self.Sgr = torch.ones_like(X[:,0:1]).to(self.device)
            # for momentum -initialization
            if self.momentum:
                self.prevSgr = copy.copy(self.Sgr)
            # for anderson acceleration
            if self.anderson:
                self.Xh = torch.empty((self.Sgr.shape[0], 0)).requires_grad_(False).to(self.device) 
                self.Fh = torch.empty((self.Sgr.shape[0], 0)).requires_grad_(False).to(self.device) 

        # source problem
        # We have to ensure all variables are 2D tensor!!!
        zeta = self.model.forward(X)
        phi1 = zeta[:,0:1]
        p1   = zeta[:,1:3]
        phi2 = zeta[:,3:4]
        p2   = zeta[:,4:6]

        eq_g1_f = operator.div(p1,X)+ Sigma_r1*phi1 -self.Sgr
        eq_g1_c = 1.0/D1*p1 + operator.grad(phi1,X)
        
        eq_g2_f =  operator.div(p2,X)+ Sigma_a2*phi2 -Sigma_s12*phi1
        eq_g2_c = 1.0/D2*p2 + operator.grad(phi2,X)
        
        eq = torch.hstack((eq_g1_f,eq_g1_c,eq_g2_f,eq_g2_c))


        self.it = self.it + 1
        if self.do_outer == False :
            # stop algorithm 
            self.stopping = True
            
        return eq
    

    def update_source(self,X):
        D1 = self.params_pde['func_D1'](X).to(self.device)
        D2 = self.params_pde['func_D2'](X).to(self.device)
        Sigma_a1  = self.params_pde['func_Sigma_a1'](X).to(self.device)
        Sigma_a2  = self.params_pde['func_Sigma_a2'](X).to(self.device)
        Sigma_s12 = self.params_pde['func_Sigma_s12'](X).to(self.device)
        Sigma_f1 = self.params_pde['func_Sigma_f1'](X).to(self.device)
        Sigma_f2 = self.params_pde['func_Sigma_f2'](X).to(self.device)
        Sigma_r1 = Sigma_a1+Sigma_s12


        zeta = self.model.forward(X)
        phi1 = zeta[:,0:1]
        p1   = zeta[:,1:3]
        phi2 = zeta[:,3:4]
        p2   = zeta[:,4:6]

        eq_g1_f = operator.div(p1,X)+ Sigma_r1*phi1 -self.Sgr
        eq_g1_c = 1.0/D1*p1 + operator.grad(phi1,X)
        eq_g2_f =  operator.div(p2,X)+ Sigma_a2*phi2 -Sigma_s12*phi1
        eq_g2_c = 1.0/D2*p2 + operator.grad(phi2,X)
        eq = torch.hstack((eq_g1_c,eq_g1_f,eq_g2_c,eq_g2_f))


        bs = torch.hstack((torch.zeros_like(eq_g1_c),self.Sgr,torch.zeros_like(eq_g2_c),torch.zeros_like(eq_g2_f)))
        innerSolver = torch.linalg.vector_norm(eq)/torch.linalg.vector_norm(bs)


        # update new source and keff
        S = Sigma_f1*phi1 + Sigma_f2*phi2
        keff =  torch.sum(S*S)/torch.sum(S*self.Sgr)
        Sgr = 1.0/keff*S

        # error 
        phi = torch.hstack((phi1,phi2))
        resFlux = torch.linalg.vector_norm(phi - self.phi)/torch.linalg.vector_norm(self.phi)
        resKeff = torch.abs(keff-self.keff)/torch.abs(self.keff)
        resIS   = torch.abs(innerSolver-self.innerSolver)/torch.abs(self.innerSolver)
        
    
        
        
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

            with torch.no_grad():
                alpha = self.solve_minimze_residual(Sgr,m=m)
                self.Xh = torch.hstack([self.Xh[:,-m:], self.Sgr])
                self.Fh = torch.hstack([self.Fh[:,-m:], Sgr])
                Sgr_new = (1-beta)*(self.Xh @ alpha) + beta*(self.Fh @ alpha)
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
                