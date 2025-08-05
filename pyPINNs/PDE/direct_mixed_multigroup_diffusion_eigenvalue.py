import torch
import numpy as np
from .pdeBase import pdeBase
from .mixed_multigroup_diffusion import mixed_multigroup_diffusion
from ..Tools.operator import operator
from ..Tools.logger import Logger
import copy

class direct_mixed_multigroup_diffusion_eigenvalue(mixed_multigroup_diffusion):
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
        # self.keff = torch.nn.Parameter(torch.tensor(1.0).float().to(self.device),requires_grad=True)
        self.keff_param = torch.nn.Parameter(torch.tensor(0.0, device=self.device), requires_grad=True)
        

    def _initialize(self,N):
        # N = X.shape[0]: the number of points
        self.it = 0  # inner iteration count
        self.ite = 0 # outer iteration count
        self.do_outer = True
        self.stopping = False
        
        
        self.phi =  torch.ones(N, self.G, device=self.device)  # [N, G]
        self.keff_prev = torch.tensor(1.0).float().to(self.device)
        self.keff_ref = torch.tensor([self.params_solver['keff_ref']], device=self.device) if self.params_solver['keff_ref'] else None
        self.innerSolver = torch.tensor([1.0], device=self.device)
        self.N_inner = self.params_solver.get('num_inner_iters', 10)
        self.verbose = self.params_solver.get('verbose', 0)
        
        #logger
        filepath = self.params_solver.get('filepath',"train.csv")
        save_dir = self.params_solver.get('save_dir','')
        header_main = self.params_solver.get('header_main',['Iter','keff', 'resKeff', 'resFlux','innerSolver','resIS'])
        verbose = self.params_solver.get('verbose',1)
        header_extra = ['accKeff'] if self.params_solver['keff_ref'] else None
        self.logger = Logger(filepath=filepath,mode='w+',save_dir=save_dir,
                            header_main=header_main,header_extra=header_extra,verbose=verbose)

        self.initialized = True # it is important to reset this parameter

    def update_cross_sections(self,X):
        self.get_cross_sections(X)
        


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
        phi_list, p_list = self.unpack_solution(zeta,self.output_format)  # unpack φ and p
        phi = torch.cat(phi_list, dim=1)      # [N, G]
        p = torch.stack(p_list, dim=1)        # [N, G, d]

        self.keff = torch.exp(self.keff_param)

        Sgr = (1.0/self.keff)*self.chi.view(1,-1)*torch.sum(self.NuSigma_f * phi,dim=1, keepdim=True)   # [1,G]*[N, 1] =[N,G]

        # Te matrix: T_e(g, g') = Sigma_r if g=g' else -Sigma_s[g',g]
        Te = -self.Sigma_s.clone()           # [N, G, G]
        Te[:, torch.arange(self.G), torch.arange(self.G)] = self.Sigma_r      
    
        # Compute divergence of current: div(p_g)
        div_p = torch.stack([operator.div(p[:, g, :], X) for g in range(self.G)], dim=1)   # [N, G, 1]
        # removal_term = torch.einsum('nij,nj->ni', Te, phi)
        flux_constrain = div_p + torch.bmm(Te, phi.unsqueeze(-1)) - Sgr.unsqueeze(-1)     # [N, G, 1]
        # Compute gradient constraints: 1/D * p_g + ∇phi_g
        grad_constraints = torch.stack([(p[:, g, :] / self.D[:, g:g+1]) + operator.grad(phi[:, g:g+1], X) for g in range(self.G)], dim=1)  # [N, G, d]
        residuals = torch.cat([flux_constrain, grad_constraints], dim=-1)  # [N, G, 1+d]

        # if self.it % 20 ==0:
        #     if self.verbose >= 1:
        #         print(f"Iter= {self.it}, \t keff = {self.keff.cpu().detach().item()}")
                
                
        self.it = self.it + 1
        return residuals
    
    
    def update_source(self,X):
        #! X can be different from X_train
        # Model output
        zeta = self.model(X)
        phi_list,p_list = self.unpack_solution(zeta,self.output_format)
        phi = torch.cat(phi_list, dim=1)      # [N, G]
        # p = torch.stack(p_list, dim=1)        # [N, G, d]

        self.keff = torch.exp(self.keff_param)

        Sgr = (1.0/self.keff)*self.chi.view(1,-1)*torch.sum(self.NuSigma_f * phi,dim=1, keepdim=True)   # [1,G]*[N, 1] =[N,G]
        
        
        # Compute residual for inner solver norm
        eq_residual = self.residual_PDE(X)  # [N, G,(1+d)]
        bs_norm = torch.linalg.vector_norm(Sgr) + 1e-12  # avoid div by zero
        innerSolver = torch.linalg.vector_norm(eq_residual) / bs_norm
        
        
        # Compute convergence errors
        phi_prev = torch.cat(self.phi, dim=1) if isinstance(self.phi, list) else self.phi
        resFlux = torch.linalg.vector_norm(phi - phi_prev) / (torch.linalg.vector_norm(phi_prev) + 1e-12)
        resKeff = torch.abs(self.keff - self.keff_prev) / (torch.abs(self.keff) + 1e-12)
        resIS = torch.abs(innerSolver - self.innerSolver) / (torch.abs(self.innerSolver) + 1e-12)

        # Update iteration and check stopping criteria
        self.ite += 1
        # if self.check_stopping(resFlux,resKeff):
        #     self.do_outer = False


        if self.keff_ref is not None:
            accKeff = torch.abs(self.keff-self.keff_ref)/torch.abs(self.keff_ref)

            self.logger.log(infos_main=[self.ite,self.keff.item(),resKeff.item(),resFlux.item(),innerSolver.item(),resIS.item()],
                            infos_extra=[accKeff.item()])
        else:
            self.logger.log(infos_main=[self.ite,self.keff.item(),resKeff.item(),resFlux.item(),innerSolver.item(),resIS.item()])

        # save old values
        self.phi = phi.detach().clone()
        self.innerSolver = innerSolver.detach().clone()
        self.keff_prev   = self.keff
        # clear cuda
        self.clear_cuda()
        
    def mass_regularization(self,X_train):
        zeta = self.model(X_train)
        phi_list = self.unpack_solution(zeta,self.output_format)[0]  
        phi = torch.cat(phi_list, dim=1)      # [N, G]
        loss_norm = (1 - torch.mean(phi**2))**2

        self.keff = torch.exp(self.keff_param)
        # Flux positivity constraint (penalizing negative flux)
        loss_positivity = torch.mean(torch.relu(-phi))
        # Regularization on k_eff
        loss_keff = 1.0 / self.keff + torch.exp(self.keff-1)
        # loss_log = torch.log(self.keff)
        # loss_log = torch.log(1 + self.keff ** 2)

        # loss_keff = (1.0 / self.keff)* torch.log(1.0 / self.keff) - (1.0 / self.keff) + 1.0 # KL

        
        # Entropy-based stabilization: k_eff log(k_eff)
        # loss_entropy = self.keff * torch.log(self.keff + 1e-6)

        total_regularization = loss_norm +loss_positivity+loss_keff
        return total_regularization


    





    
