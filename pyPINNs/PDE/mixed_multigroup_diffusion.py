import torch
import numpy as np
from .pdeBase import pdeBase
import matplotlib.pyplot as plt
from ..Tools.operator import operator

class mixed_multigroup_diffusion(pdeBase):
    def __init__(self,domain,model,params_pde,device):
        super().__init__(domain,model,device)
        self.params_pde = params_pde
        self.G = params_pde.get('num_groups',1)
        self.output_format = params_pde.get('output_format',"interleave")

    def unpack_solution(self, zeta,output_format="interleave"):
        """Unpack neural network output zeta into phi_list and p_list depending on output_format.

        Returns:
            phi_list: list of tensors [N,1], length G
            p_list: list of tensors [N,d], length G 
        """

        d = self.domain.input_dim
        G = self.G
        if output_format == "interleave":
            phi_list = [zeta[:, (d+1)*g : (d+1)*g + 1] for g in range(G)]
            p_list   = [zeta[:, (d+1)*g + 1 : (d+1)*(g + 1)] for g in range(G)]
                
        else:  # block format: 
            phi_list = [zeta[:, g:g+1] for g in range(G)]           #  G elements of [N, 1]
            p_list = [zeta[:, G + g*d: G + (g+1)*d]   for g in range(G)] #  G element of [N, d]
        return phi_list, p_list
    
    def phi_predict(self,X_test):
        zeta = self.model.forward(X_test)
        phi_list = self.unpack_solution(zeta,output_format=self.output_format)[0]
        return phi_list
    
    def my_currents_predict(self, Xc_test):
        """
        Predict current components J_{g,d} using unpack_solution.

        Args:
            Xc_test (list of torch.Tensor): List of length ndim, where each element is a [N, input_dim]
                tensor with test points specific to direction d.

        Returns:
            List[List[Tensor]]: p_pred[g][d] = J_g^d at points Xc_test[d], shape [N, 1]
        """
        ndim = self.domain.input_dim
        G = self.G
        output_format = self.output_format  # either 'interleaved' or 'block'

        p_pred = []

        for g in range(G):
            group_currents = []
            for d in range(ndim):
                z_d = self.model.forward(Xc_test[d])
                _, current_list = self.unpack_solution(z_d, output_format=output_format)
                j_gd = current_list[g][:, d : d + 1]  # select the d-th component of J_g
                group_currents.append(j_gd)
            p_pred.append(group_currents)

        return p_pred  # shape: [G][ndim], each [N, 1]

    
    def currents_predict(self,Xc_test):
        ndim = self.domain.input_dim
        ngroup = self.G
        p_pred = [[self.model.forward(Xc_test[idim])[:,igroup*(ndim+1) + (idim+1): igroup*(ndim+1) +(idim+2)] for idim in range(ndim)] for igroup in range(ngroup)]
        return p_pred
    
    

    
    # def get_phi_test(self,X_test,phi_test,normalization=False):

    #     ngroup = self.G
    #     phi_pred = self.phi_predict(X_test) 
    #     mass_phi_pred = torch.sum(torch.vstack(phi_pred))
    #     mass_phi_test = torch.sum(torch.vstack(phi_test))

    #     if normalization:
    #         phi_pred = [phi_pred[igroup]/mass_phi_pred for igroup in range(ngroup)]
    #         phi_test = [phi_test[igroup]/mass_phi_test for igroup in range(ngroup)]
    #     phi_AE = [torch.abs(phi_test[igroup]-phi_pred[igroup]) for igroup in range(ngroup)]
    #     phi_REL_L2 = torch.linalg.vector_norm(torch.vstack(phi_test)-torch.vstack(phi_pred))/torch.linalg.vector_norm(torch.vstack(phi_test)) 
    #     return phi_REL_L2, phi_AE, phi_pred, phi_test,mass_phi_pred, mass_phi_test
    
    # def get_currents_test(self,Xc_test,p_test,normalization=False,mass_pred=1.0,mass_test=1.0):
    
    #     p_pred = self.currents_predict(Xc_test)
    #     ngroup = self.G
    #     if normalization:
    #         p_pred = [ [ p_pred[igroup][idim]/mass_pred for idim in range(self.domain.input_dim)] for igroup in range(ngroup)]
    #         p_test = [ [ p_test[igroup][idim]/mass_test for idim in range(self.domain.input_dim)] for igroup in range(ngroup)]

    #     p_AE    = [[torch.abs(p_test[igroup][idim]-p_pred[igroup][idim]) for idim in range(self.domain.input_dim)] for igroup in range(ngroup)]# list
    #     p_AE_g  = [ torch.vstack(p_AE[igroup])    for igroup in range(ngroup)]
    #     p_test_g = [torch.vstack(p_test[igroup])    for igroup in range(ngroup)]
    #     p_REL_L2 =  torch.linalg.vector_norm(torch.vstack(p_AE_g))/torch.linalg.vector_norm(torch.vstack(p_test_g)) 
    
    #     return p_REL_L2, p_AE, p_pred, p_test
    
    

    def get_phi_test(self, X_test, phi_test, normalization=False):
        ngroup = self.G
        phi_pred = self.phi_predict(X_test)

        # Stack once (global view)
        phi_pred_stack = torch.cat(phi_pred, dim=0)
        phi_test_stack = torch.cat(phi_test, dim=0)

        # Mass 
        mass_phi_pred = torch.sum(phi_pred_stack)
        mass_phi_test = torch.sum(phi_test_stack)

        eps = 1e-14
        if normalization:
            phi_pred = [p / mass_phi_pred for p in phi_pred]
            phi_test = [p / mass_phi_test for p in phi_test]
            # Stack once (global view)
            phi_pred_stack = torch.cat(phi_pred, dim=0)
            phi_test_stack = torch.cat(phi_test, dim=0)



        # Per-group absolute error
        phi_AE = [
            torch.abs(phi_test[g] - phi_pred[g])
            for g in range(ngroup)
        ]

        # Per-group relative L2  (LIST)
        phi_REL_L2 = [
            torch.linalg.vector_norm(phi_test[g] - phi_pred[g])/(torch.linalg.vector_norm(phi_test[g]) + eps)
            for g in range(ngroup)
        ]

        # Global relative L2  (SCALAR)
        phi_REL_L2_global = (
            torch.linalg.vector_norm(phi_test_stack - phi_pred_stack)/(torch.linalg.vector_norm(phi_test_stack) + eps)
        )

        return (
            phi_REL_L2_global,    # scalar
            phi_REL_L2,           # list (per group)
            phi_AE,               # list (per group)
            phi_pred,
            phi_test,
            mass_phi_pred,
            mass_phi_test,
        )



    def get_currents_test(self,Xc_test,p_test,normalization=False,mass_pred=1.0,mass_test=1.0):

        p_pred = self.currents_predict(Xc_test)
        ngroup = self.G
        ndim = self.domain.input_dim
        eps = 1e-14

        # Optional normalization
        if normalization:
            p_pred = [
                [p_pred[g][d] / (mass_pred + eps) for d in range(ndim)]
                for g in range(ngroup)
            ]
            p_test = [
                [p_test[g][d] / (mass_test + eps) for d in range(ndim)]
                for g in range(ngroup)
            ]

        # Absolute error per group per dimension
        p_AE = [
            [torch.abs(p_test[g][d] - p_pred[g][d]) for d in range(ndim)]
            for g in range(ngroup)
        ]

        # Stack each group (all dimensions together)
        p_pred_g = [
            torch.cat([p_pred[g][d].reshape(-1) for d in range(ndim)], dim=0)
            for g in range(ngroup)
        ]

        p_test_g = [
            torch.cat([p_test[g][d].reshape(-1) for d in range(ndim)], dim=0)
            for g in range(ngroup)
        ]

        p_AE_g = [
            torch.cat([p_AE[g][d].reshape(-1) for d in range(ndim)], dim=0)
            for g in range(ngroup)
        ]

        # Per-group REL_L2
        p_REL_L2 = [
            torch.linalg.vector_norm(p_AE_g[g])
            / (torch.linalg.vector_norm(p_test_g[g]) + eps)
            for g in range(ngroup)
        ]

        # Global stacking (all groups)
        p_pred_stack = torch.cat(p_pred_g, dim=0)
        p_test_stack = torch.cat(p_test_g, dim=0)

        # Global REL_L2
        p_REL_L2_global = (
            torch.linalg.vector_norm(p_test_stack - p_pred_stack)
            / (torch.linalg.vector_norm(p_test_stack) + eps)
        )

        return (
            p_REL_L2_global,          # global
            p_REL_L2,                 # per group
            p_AE,                     # per group per dimension
            p_pred,                   # per group per dimension
            p_test,                   # per group per dimension
        )







    def get_cross_sections(self,X):
        if self.params_pde.get('XsEvaluater',None):
            print(" get cross sections from XsEvaluater")
            xsEvaluater = self.params_pde['XsEvaluater']
            self.D = xsEvaluater.diffusion(X)
            self.Sigma_r = xsEvaluater.sigma_r(X)
            self.Sigma_s = xsEvaluater.sigma_s(X)
            
            self.NuSigma_f = xsEvaluater.nusigma_f(X)
            self.chi = xsEvaluater.fission_spectrum()
            
            self.Sf = xsEvaluater.source(X)
        else:
            print(" get cross sections from direct functions")
            self.D = self.params_pde['func_D'](X).to(self.device)              # [N, G]
            self.Sigma_r = self.params_pde['func_Sigma_r'](X).to(self.device)  # [N, G]

            if self.params_pde.get('func_Sigma_s',None):
                self.Sigma_s = self.params_pde['func_Sigma_s'](X).to(self.device)  # [N, G, G]
            else:
                self.Sigma_s = torch.zeros(X.shape[0],1,1,device=self.device)
            # source problem
            if self.params_pde.get('func_Sf',None):
                self.Sf = self.params_pde['func_Sf'](X).to(self.device) # [N, G]  
            # eigenvalue problem
            if self.params_pde.get('func_NuSigma_f',None):
                self.NuSigma_f = self.params_pde['func_NuSigma_f'](X).to(self.device) # [N, G]
            self.chi = self.params_pde.get('chi',None).to(self.device) # [G]

    # def update_training_points(self, new_X):
    #     self.get_cross_sections(new_X)

    def residual_PDE(self, X):
        """
        Calculate the residual of the multigroup neutron diffusion  with output shape [N, G]
        """
        """
        Multigroup neutron diffusion residual using Te operator:
            T_e(g, g') = Σ_r if g = g'
                        -Σ_s(g'→g) if g ≠ g'
        """
        NotImplemented("The function Dirichlet_BC is not implemented") 
    
    
    def residual_BC(self, X_bc):
        """
        Compute boundary residuals and return them in shape [N_total, G],
        where N_total is the total number of boundary points across all faces.

        Parameters:
            X_bc: list of shape [ndim][2], each entry is a tensor of shape [N_face, d]

        Returns:
            residu_BC: tensor of shape [N_total, G]
        """

        # nunk = ndim + 1  # unknowns per group: φ + p
        list_residu_BC_groupwise = [[] for _ in range(self.G)]

        for idim in range(self.domain.input_dim):
            for id in range(2):  # 0: min, 1: max
                X_face = X_bc[idim][id]
                zeta = self.model.forward(X_face)  # shape [N_face, G * nunk]
                phi_list, p_list = self.unpack_solution(zeta,output_format=self.output_format)

                for g in range(self.G):
                    phi = phi_list[g]
                    p   = p_list[g]
                    bc_type = self.domain.boundary_conditions[idim][id]
                    if bc_type == 'ZERO_FLUX':
                        residu = phi  # φ = 0

                    elif bc_type == 'REFLECTION':
                        normal_sign = (-1) ** (id + 1)
                        pn = p[:, idim:idim+1] * normal_sign
                        residu = pn  # p · n = 0

                    elif bc_type == 'VACUUM':
                        normal_sign = (-1) ** (id + 1)
                        pn = p[:, idim:idim+1] * normal_sign
                        residu = -pn + 0.5 * phi  # Robin BC

                    else:
                        residu = torch.zeros_like(phi)  # unknown BC
                    list_residu_BC_groupwise[g].append(residu)

        # Stack residuals per group: each has shape [N_total, 1]
        group_residuals = [torch.vstack(res_list) for res_list in list_residu_BC_groupwise]

        # Concatenate along last dimension → shape [N_total, G]
        residu_BC = torch.cat(group_residuals, dim=1)
        return residu_BC  # shape [N_total, G]
    

    def loss_PDE(self,X_train):
        residu = self.residual_PDE(X_train)
        loss_f = torch.mean(torch.sum(residu**2, dim=tuple(range(1, residu.ndim))))
        return loss_f
    
    def loss_BC(self,X_bc):
        residu = self.residual_BC(X_bc)
        loss_bc = torch.mean(torch.sum(residu**2, dim=tuple(range(1, residu.ndim))))
        return loss_bc
    
