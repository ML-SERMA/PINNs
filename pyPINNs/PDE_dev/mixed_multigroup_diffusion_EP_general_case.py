import torch
import numpy as np
from .pdeBase import pdeBase
from .mixed_multigroup_diffusion import mixed_multigroup_diffusion
from ..Tools.operator import operator
from ..Tools.logger import Logger
import copy

class mixed_multigroup_diffusion_EP(mixed_multigroup_diffusion):
    def __init__(self, domain, model, params_pde, params_solver, device):
        """
        Args:
            domain: domain object for PDE spatial operators
            model: neural network model predicting [phi, J]
            params_pde: dict of PDE params functions (D, Sigma_a, Sigma_s, Sigma_f)
            params_solver: dict of solver params (num_inner_iters, momentum, anderson, betas, etc)
            device: torch device

        """
        super().__init__(domain, model, params_pde, device)
        self.domain = domain
        self.model = model
        self.params_pde = params_pde
        self.params_solver = params_solver
        self.device = device
        self.initialized = False
        self.initialized_test = False
        self.scaling_loss = params_solver.get('scaling_loss',True)
        

    def _initialize(self,N):
        # N = X.shape[0]: the number of points
        self.it = 0  # inner iteration count
        self.ite = 0 # outer iteration count
        self.do_outer = True
        self.stopping = False

        phi =  torch.ones(N, self.G, device=self.device)  # [N_train, G]
        self.Sgr_train = torch.sum(self.NuSigma_f*phi,dim=1, keepdim=True)  # [N_train, 1]

        # self.phi =  torch.ones(N, self.G, device=self.device)  # [N, G]
        # self.Sgr = torch.sum(self.NuSigma_f*self.phi,dim=1, keepdim=True)  # [N, 1]
        # self.prevSgr = torch.sum(self.NuSigma_f*self.phi,dim=1, keepdim=True)  # [N, 1]
        # self.Xh = torch.ones(N, 1, device=self.device) # [N,1]
        # self.Fh = torch.ones(N, 1, device=self.device)  # [N,1]


        self.keff = torch.tensor([1.0], device=self.device)
        self.keff_ref = torch.tensor([self.params_solver['keff_ref']], device=self.device) if self.params_solver['keff_ref'] else None
        self.innerSolver = torch.tensor([1.0], device=self.device)
        self.N_inner = self.params_solver.get('num_inner_iters', 10)
        self.anderson = self.params_solver.get('anderson', False)
        self.momentum = self.params_solver.get('momentum', False)
        self.verbose = self.params_solver.get('verbose', 0)
        self.keff_method = self.params_solver.get('keff_method','scaled_rayleigh')

        # beta params for momentum
        self.beta1 = self.params_solver.get('beta1', 0.0)
        self.beta2 = self.params_solver.get('beta2', 0.0)

        # parameters for Anderson acceleration
        self.beta_anderson = self.params_solver.get('beta', 0.5)
        self.m_anderson = self.params_solver.get('m', 4)

        #logger
        filepath = self.params_solver.get('filepath',"train.csv")
        save_dir = self.params_solver.get('save_dir','')
        header_main = self.params_solver.get('header_main',['Iter','keff', 'resKeff', 'resFlux','innerSolver','resIS'])
        verbose = self.params_solver.get('verbose',1)
        header_extra = ['accKeff'] if self.params_solver['keff_ref'] else None
        self.logger = Logger(filepath=filepath,mode='w+',save_dir=save_dir,
                            header_main=header_main,header_extra=header_extra,verbose=verbose)

        self.initialized = True # it is important to reset this parameter

    def _initialize_test(self,N):
        self.phi =  torch.ones(N, self.G, device=self.device)  # [N_test, G]
        self.Sgr = torch.sum(self.NuSigma_f_test*self.phi,dim=1, keepdim=True)  # [N_test, 1]
        self.prevSgr = torch.sum(self.NuSigma_f_test*self.phi,dim=1, keepdim=True)  # [N_test, 1]
        self.Xh = torch.ones(N, 1, device=self.device) # [N_test,1]
        self.Fh = torch.ones(N, 1, device=self.device)  # [N_test,1]
        self.initialized_test = True


    def update_cross_sections(self,X):
        self.get_cross_sections(X)
        # update Sgr
        zeta = self.model.forward(X)
        phi_list= self.unpack_solution(zeta,self.output_format)[0]
        phi = torch.cat(phi_list, dim=1)  # [N, G]
        fission_production = self.NuSigma_f * phi # [N, G]
        S = torch.sum(fission_production,dim=1, keepdim=True)  # [N, 1]: 
        # S = self.chi.view(1,-1)*total_fission  # [1,G]*[N, 1] =[N,G]
        self.Sgr = S / (self.keff + 1e-12) # [N,1]
        # reset history with new X
        self.prevSgr = S / (self.keff + 1e-12) # [N,1]
        # self.Xh = phi.view(-1,1)  # [N*G,1]
        # self.Fh = phi.view(-1,1)  # [N*G,1]
        self.Xh = self.prevSgr.detach().clone()  # [N,1]
        self.Fh = self.Sgr.detach().clone()  # [N,1]


    def residual_PDE(self, X):
        """Compute PDE residual for multigroup neutron diffusion.

        Args:
            X: spatial points tensor [N, dim]

        Returns:
            residual: tensor of shape [N, G*(1+d)] with residuals for each group:
                    flux residual [N,1], and current residual [N,d]
        """
        if not self.initialized:
            self.get_cross_sections(X) # get cross sections value from X_train
            self._initialize(X.shape[0])
            
        # Evaluate model predictions
        
        zeta = self.model(X)
        phi_list, p_list = self.unpack_solution(zeta,self.output_format)  # unpack φ and p
        phi = torch.cat(phi_list, dim=1)      # [N, G]
        p = torch.stack(p_list, dim=1)        # [N, G, d]


        ## Method 1: take a loop on g
        # residuals = []
        # for g in range(self.G):
        #     # Compute scattering source
        #     source_fission = self.Sgr[:, g].view(-1, 1)
        #     source_scatter = torch.sum(self.Sigma_s[:, g, :] * torch.cat(phi_list, dim=1), dim=1).view(-1,1)  # [N,1]
        #     # Residual for flux balance and Fick's law  for the current
        #     res_flux = operator.div(p_list[g], X) + self.Sigma_r[:, g].view(-1, 1) * phi_list[g] -source_scatter - source_fission   # [N,1]
        #     res_current = p_list[g] / self.D[:, g].view(-1, 1) + operator.grad(phi_list[g], X)  # [N,d]
        #     residuals.append(torch.hstack([res_flux, res_current])) # list of [N, 1+d]
        # residuals = torch.stack(residuals,dim=1)  # [N, G,(1+d)]


        ## perform calculate in bloc form
    
        # Te matrix: T_e(g, g') = Sigma_r if g=g' else -Sigma_s[g',g]
        # Te = -self.Sigma_s.clone()           # [N, G, G]
        # Te[:, torch.arange(self.G), torch.arange(self.G)] = self.Sigma_r   

        diag_mask = torch.eye(self.G, device=X.device).unsqueeze(0)  # [1,G,G]
        Te = -self.Sigma_s + diag_mask * (self.Sigma_r.unsqueeze(-1))   
    
        # Compute divergence of current: div(p_g)
        div_p = torch.stack([operator.div(p[:, g, :], X) for g in range(self.G)], dim=1)   # [N, G, 1]

        # removal_term = torch.einsum('nij,nj->ni', Te, phi)
        flux_constrain = div_p + torch.bmm(Te, phi.unsqueeze(-1)) - self.Sgr_train.unsqueeze(1) * self.chi[None, :, None]    # [N, G, 1]
        # Compute gradient constraints: 1/D * p_g + ∇phi_g
        grad_constraints = torch.stack([(p[:, g, :] / self.D[:, g:g+1]) + operator.grad(phi[:, g:g+1], X) for g in range(self.G)], dim=1)  # [N, G, d]
        
        if self.scaling_loss:
            dTe_inv_sqrt = 1.0 / torch.sqrt(self.Sigma_r + 1e-12)   # [N, G]
            flux_constrain_scaled = flux_constrain * dTe_inv_sqrt.unsqueeze(-1)  # [N, G, 1]
            grad_constraints_scaled = grad_constraints * torch.sqrt(self.D + 1e-12).unsqueeze(-1)
            residuals = torch.cat([flux_constrain_scaled, grad_constraints_scaled], dim=-1)  # [N, G, 1+d]
        else:
            residuals = torch.cat([flux_constrain, grad_constraints], dim=-1)  # [N, G, 1+d]

        self.it = self.it + 1
        # stop algorithm 
        if self.do_outer == False :
            self.stopping = True

        return residuals

    def solve_minimize_residual(self, Sgr, m):
        """Solve for alpha coefficients in Anderson acceleration."""
        Rg = self.Fh[:, -m:] - self.Xh[:, -m:]
        Gt = Sgr.view(-1,1) - self.Sgr.view(-1,1)
        matLG = torch.repeat_interleave(Gt, Rg.shape[1], dim=1) - Rg
        gamma = torch.linalg.lstsq(matLG, Gt).solution
        alpha = torch.vstack([gamma, 1 - torch.sum(gamma)])
        return alpha

    def apply_acceleration(self, Sgr):
        """Apply momentum or Anderson acceleration on the fission source."""
        if self.momentum:
            if self.verbose == 2:
                print(f"Apply momentum with beta1={self.beta1}, beta2={self.beta2}")
            with torch.no_grad():
                S_updated = Sgr + self.beta1 * (Sgr - self.Sgr) + self.beta2 * (Sgr - self.prevSgr)
                self.prevSgr = self.Sgr.detach().clone()
        elif self.anderson:
            if self.verbose == 2:
                print(f"Apply Anderson with beta={self.beta_anderson}, m={self.m_anderson}")
            with torch.no_grad():
                alpha = self.solve_minimize_residual(Sgr, m=self.m_anderson)
                self.Xh = torch.hstack([self.Xh[:, -self.m_anderson:], self.Sgr.view(-1,1)])
                self.Fh = torch.hstack([self.Fh[:, -self.m_anderson:], Sgr.view(-1,1)])
                if self.ite <=5:
                    print("Skip some first iterations for Anderson method !")
                    S_updated = Sgr.detach().clone()
                else:
                    S_temp = (1 - self.beta_anderson) * (self.Xh @ alpha) + self.beta_anderson * (self.Fh @ alpha)
                    S_updated = S_temp.view(-1,1)# (N,1)
        else:
            S_updated = Sgr.detach().clone()

        return S_updated

    def check_stopping(self, res_flux, res_keff):
        tol_flux = self.params_solver.get('tol_flux', 1e-5)
        tol_keff = self.params_solver.get('tol_keff', 1e-6)
        return res_flux <= tol_flux and res_keff <= tol_keff
    
    def update_Sgr_train(self,X): # X=X_train and using this function after update source to ensure that self.keff is updated
        with torch.no_grad():
            zeta = self.model.forward(X)
            phi_list,J_list = self.unpack_solution(zeta,self.output_format)
            phi = torch.cat(phi_list, dim=1)  # [N, G]
            # Fission production rate in group g
            fission_production = self.NuSigma_f * phi # [N, G]
            # Total fission neutron yield (integrated over all groups)
            S = torch.sum(fission_production,dim=1, keepdim=True)  # [N, 1]: 
            self.Sgr_train = S / (self.keff + 1e-12)

    def get_cross_sections_test(self,X):
        if self.params_pde.get('XsEvaluater',None):
            print(" get cross sections from XsEvaluater")
            xsEvaluater = self.params_pde['XsEvaluater']
            self.D_test = xsEvaluater.diffusion(X)
            self.Sigma_r_test = xsEvaluater.sigma_r(X)
            self.Sigma_s_test = xsEvaluater.sigma_s(X)
            
            self.NuSigma_f_test = xsEvaluater.nusigma_f(X)
            self.chi_test = xsEvaluater.fission_spectrum()
            
            self.Sf_test = xsEvaluater.source(X)
        else:
            print(" get cross sections from direct functions")
            self.D_test = self.params_pde['func_D'](X).to(self.device)              # [N, G]
            self.Sigma_r_test = self.params_pde['func_Sigma_r'](X).to(self.device)  # [N, G]

            if self.params_pde.get('func_Sigma_s',None):
                self.Sigma_s_test = self.params_pde['func_Sigma_s'](X).to(self.device)  # [N, G, G]
            else:
                self.Sigma_s_test = torch.zeros(X.shape[0],1,1,device=self.device)
            # source problem
            if self.params_pde.get('func_Sf',None):
                self.Sf_test = self.params_pde['func_Sf'](X).to(self.device) # [N, G]  
            # eigenvalue problem
            if self.params_pde.get('func_NuSigma_f',None):
                self.NuSigma_f_test = self.params_pde['func_NuSigma_f'](X).to(self.device) # [N, G]
            self.chi_test = self.params_pde.get('chi',None).to(self.device) # [G]


    def residual_PDE_test(self, X):
        """Compute PDE residual for multigroup neutron diffusion.

        Args:
            X: spatial points tensor [N, dim]

        Returns:
            residual: tensor of shape [N, G*(1+d)] with residuals for each group:
                    flux residual [N,1], and current residual [N,d]
        """
        zeta = self.model(X)
        phi_list, p_list = self.unpack_solution(zeta,self.output_format)  # unpack φ and p
        phi = torch.cat(phi_list, dim=1)      # [N, G]
        p = torch.stack(p_list, dim=1)        # [N, G, d]

        # Te matrix: T_e(g, g') = Sigma_r if g=g' else -Sigma_s[g',g]
        # Te = -self.Sigma_s_test.clone()           # [N, G, G]
        # Te[:, torch.arange(self.G), torch.arange(self.G)] = self.Sigma_r_test      
        diag_mask = torch.eye(self.G, device=X.device).unsqueeze(0)  # [1,G,G]
        Te = -self.Sigma_s_test + diag_mask * (self.Sigma_r_test.unsqueeze(-1))
    
        # Compute divergence of current: div(p_g)
        div_p = torch.stack([operator.div(p[:, g, :], X) for g in range(self.G)], dim=1)   # [N, G, 1]

        # removal_term = torch.einsum('nij,nj->ni', Te, phi)
        flux_constrain = div_p + torch.bmm(Te, phi.unsqueeze(-1)) - self.Sgr.unsqueeze(1) * self.chi_test[None, :, None]    # [N, G, 1]
        # Compute gradient constraints: 1/D * p_g + ∇phi_g
        grad_constraints = torch.stack([(p[:, g, :] / self.D_test[:, g:g+1]) + operator.grad(phi[:, g:g+1], X) for g in range(self.G)], dim=1)  # [N, G, d]
        
        if self.scaling_loss:
            dTe_inv_sqrt = 1.0 / torch.sqrt(self.Sigma_r_test + 1e-12)   # [N, G]
            flux_constrain_scaled = flux_constrain * dTe_inv_sqrt.unsqueeze(-1)  # [N, G, 1]
            grad_constraints_scaled = grad_constraints * torch.sqrt(self.D_test + 1e-12).unsqueeze(-1)
            residuals = torch.cat([flux_constrain_scaled, grad_constraints_scaled], dim=-1)  # [N, G, 1+d]
        else:
            residuals = torch.cat([flux_constrain, grad_constraints], dim=-1)  # [N, G, 1+d]

        return residuals

    def update_source(self, X):
        """Update fission source and eigenvalue keff, and monitor convergence."""
        if not self.initialized_test: 
            self.get_cross_sections_test(X) # get cross sections value from X_test
            self._initialize_test(X.shape[0])

        # Model output

        zeta = self.model.forward(X)
        phi_list,_= self.unpack_solution(zeta,self.output_format)
        # Compute residual for inner solver norm
        eq_residual = self.residual_PDE_test(X)  # [N, G,(1+d)]
        bs_norm = torch.linalg.vector_norm(self.Sgr) + 1e-12  # avoid div by zero
        innerSolver = torch.linalg.vector_norm(eq_residual) / bs_norm

        phi = torch.cat(phi_list, dim=1)  # [N, G]



        # Fission production rate in group g
        fission_production = self.NuSigma_f_test * phi # [N, G]
        # Total fission neutron yield (integrated over all groups)
        S = torch.sum(fission_production,dim=1, keepdim=True)  # [N, 1]: 
        # S = self.chi.view(1,-1)*total_fission  # [1,G]*[N, 1] =[N,G]
        # --- Compute keff using selected method ---
        S_old = self.Sgr * self.keff  # recover old unnormalized source


        if self.keff_method == "scaled_rayleigh":
            numerator = torch.sum(S * S)
            denominator = torch.sum(S * self.Sgr)
            keff = numerator / (denominator + 1e-12)  # avoid div zero
        elif self.keff_method == "rayleigh":
            numerator = torch.sum(S * S)
            denominator = torch.sum(S * S_old)
            keff = numerator / (denominator + 1e-12)
        elif self.keff_method == "projection":  
            numerator = torch.sum(S * S_old)
            denominator = torch.sum(S_old * S_old)
            keff = numerator / (denominator + 1e-12)
        elif self.keff_method == "power_method":
            numerator = torch.linalg.vector_norm(S)
            denominator  = torch.linalg.vector_norm(S_old) 
            keff = numerator / (denominator + 1e-12)

        elif self.keff_method == "scaled_norm":  
            # least squares method
            numerator = torch.sum(S * S)
            denominator = torch.sum(self.Sgr * self.Sgr)
            keff = torch.sqrt(numerator / (denominator + 1e-12))
            
        else:
            print('The method is not implemented!')



        # Normalize new source
        Sgr = S / (keff + 1e-12)

        # Compute convergence errors
        phi_prev = torch.cat(self.phi, dim=1) if isinstance(self.phi, list) else self.phi
        resFlux = torch.linalg.vector_norm(phi - phi_prev) / (torch.linalg.vector_norm(phi_prev) + 1e-12)
        resKeff = torch.abs(keff - self.keff) / (torch.abs(self.keff) + 1e-12)
        resIS = torch.abs(innerSolver - self.innerSolver) / (torch.abs(self.innerSolver) + 1e-12)

        # Update iteration and check stopping criteria
        self.ite += 1
        if self.check_stopping(resFlux,resKeff):
            self.do_outer = False


        if self.keff_ref is not None:
            accKeff = torch.abs(keff-self.keff_ref)/torch.abs(self.keff_ref)
            # self.accKeff.append(accKeff.cpu().detach().item())
            self.logger.log(infos_main=[self.ite,keff.item(),resKeff.item(),resFlux.item(),innerSolver.item(),resIS.item()],
                            infos_extra=[accKeff.item()])
        else:
            self.logger.log(infos_main=[self.ite,keff.item(),resKeff.item(),resFlux.item(),innerSolver.item(),resIS.item()])

        # acceleration method 
        Sgr_new = self.apply_acceleration(Sgr)
        # Update
        self.Sgr = Sgr_new.detach().clone()
        self.phi = phi.detach().clone()
        self.keff = keff.detach().clone()
        self.innerSolver = innerSolver.detach().clone()
        self.it = 0
        # clear cuda
        self.clear_cuda()





    
