import torch
import numpy as np
from .pdeBase import pdeBase
from .mixed_multigroup_diffusion import mixed_multigroup_diffusion
from ..Tools.operator import operator
from ..Tools.logger import Logger
import copy

class mixed_multigroup_diffusion_eigenvalue(mixed_multigroup_diffusion):
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
        

    def _initialize(self,N):
        # N = X.shape[0]: the number of points
        self.it = 0  # inner iteration count
        self.ite = 0 # outer iteration count
        self.do_outer = True
        self.stopping = False

        self.phi =  torch.ones(N, self.G, device=self.device)  # [N, G]
        self.Sgr = self.chi.view(1,-1)*torch.sum(self.NuSigma_f*self.phi,dim=1, keepdim=True)  # [N, G]
        self.prevSgr = self.chi.view(1,-1)*torch.sum(self.NuSigma_f*self.phi,dim=1, keepdim=True)  # [N, G]
        self.Xh = torch.ones(N, self.G, device=self.device).view(-1,1)  # [N*G,1]
        self.Fh = torch.ones(N, self.G, device=self.device).view(-1,1)  # [N*G,1]

        self.keff = torch.tensor([1.0], device=self.device)
        self.keff_ref = torch.tensor([self.params_solver['keff_ref']], device=self.device) if self.params_solver['keff_ref'] else None
        self.innerSolver = torch.tensor([1.0], device=self.device)
        self.N_inner = self.params_solver.get('num_inner_iters', 10)
        self.anderson = self.params_solver.get('anderson', False)
        self.momentum = self.params_solver.get('momentum', False)
        self.verbose = self.params_solver.get('verbose', 0)
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

    

    def residual_PDE(self, X):
        """Compute PDE residual for multigroup neutron diffusion.

        Args:
            X: spatial points tensor [N, dim]

        Returns:
            residual: tensor of shape [N, G*(1+d)] with residuals for each group:
                    flux residual [N,1], and current residual [N,d]
        """
        if not self.initialized:
            self.get_cross_sections(X) # get cross sections value 
            self._initialize(X.shape[0])
            
        # Evaluate model predictions
        zeta = self.model(X)
        phi_list, p_list = self.unpack_solution(zeta)  # unpack φ and p
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
        Te = -self.Sigma_s.clone()           # [N, G, G]
        Te[:, torch.arange(self.G), torch.arange(self.G)] = self.Sigma_r      
    
        # Compute divergence of current: div(p_g)
        div_p = torch.stack([operator.div(p[:, g, :], X) for g in range(self.G)], dim=1)   # [N, G, 1]
        # removal_term = torch.einsum('nij,nj->ni', Te, phi)
        flux_constrain = div_p + torch.bmm(Te, phi.unsqueeze(-1)) - self.Sgr.unsqueeze(-1)     # [N, G, 1]
        # Compute gradient constraints: 1/D * p_g + ∇phi_g
        grad_constraints = torch.stack([(p[:, g, :] / self.D[:, g:g+1]) + operator.grad(phi[:, g:g+1], X) for g in range(self.G)], dim=1)  # [N, G, d]
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
                S_temp = (1 - self.beta_anderson) * (self.Xh @ alpha) + self.beta_anderson * (self.Fh @ alpha)
                S_updated = S_temp.view(-1,self.G)
        else:
            S_updated = Sgr.detach().clone()

        return S_updated

    def check_stopping(self, res_flux, res_keff):
        tol_flux = self.params_solver.get('tol_flux', 1e-5)
        tol_keff = self.params_solver.get('tol_keff', 1e-6)
        return res_flux <= tol_flux and res_keff <= tol_keff
    
    def update_source(self, X):
        """Update fission source and eigenvalue keff, and monitor convergence."""
        
        # Model output
        zeta = self.model.forward(X)
        phi_list = self.unpack_solution(zeta,self.output_format)[0]
        # Compute residual for inner solver norm
        eq_residual = self.residual_PDE(X)  # [N, G*(1+d)]
        bs_norm = torch.linalg.vector_norm(self.Sgr) + 1e-12  # avoid div by zero
        innerSolver = torch.linalg.vector_norm(eq_residual) / bs_norm
        
        # Compute fission source S = Sigma_f * phi
        phi = torch.cat(phi_list, dim=1)  # [N, G]
        fission_production = (self.NuSigma_f * phi).sum(dim=1, keepdim=True)  # [N, 1]
        S = self.chi.view(1,-1)*fission_production  # [1,G]*[N, G] 
        # Compute new keff (Rayleigh quotient)
        numerator = torch.sum(S * S)
        denominator = torch.sum(S * self.Sgr)
        keff = numerator / (denominator + 1e-12)  # avoid div zero
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





    
