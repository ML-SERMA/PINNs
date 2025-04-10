import torch
import numpy as np
from .pdeBase import pdeBase
import matplotlib.pyplot as plt
from ..Tools.operator import operator

class PC_Center_primal_diffusion_one_group(pdeBase):
    def __init__(self,domain,model,params_pde,device):
        super().__init__(domain,model,device)
        self.params_pde = params_pde
        
    
    def residual_PDE(self,X):
        D = self.params_pde['func_D'](X).to(self.device)
        sigma_a = self.params_pde['func_Sigma_a'](X).to(self.device)
        s = self.params_pde['func_Source'](X).to(self.device)
        z = self.model.forward(X).to(self.device)
        
        x=X[:,0:1]
        y=X[:,1:2]
        phi1 = z[:,0:1]
        phi2 = z[:,1:2]
        phi = torch.where((x >= 1.0/3) & (x <= 2.0/3) & (y >= 1.0/3) & (y <= 2.0/3), phi1, phi2) # Center
        # phi =   ((x >= 1.0/3) & (x <= 2.0/3) & (y >= 1.0/3) & (y <= 2.0/3))*phi1 + \
        #         ((x >= 0) & (x <= 1) & (y >= 0) & (y < 1.0/3))*phi2 +\
        #         ((x >=0) & (x <= 1) & (y > 2.0/3) & (y <= 1.0))*phi2 +\
        #         ((x >=0) & (x < 1.0/3) & (y >= 1.0/3) & (y <= 2.0/3))*phi2 +\
        #         ((x >2.0/3) & (x <= 1) & (y >= 1.0/3) & (y <= 2.0/3))*phi2

        # phi = torch.where(((x>=0.5)&(y<=0.5)) + ((x<=0.5)&(y>=0.5)), phi1, phi2) # Dauge
                
        eq = -operator.div(D*operator.grad(phi,X),X) +sigma_a*phi -s
        return eq


    
    def residual_BC(self, X_bc):
        # define residual of each boundary
        ndim = len(X_bc)
        list_residu_BC = []
        for idim in range(ndim):
            for id in range(2):
                if self.domain.boundary_conditions[idim][id] == 'ZERO_FLUX':
                    z = self.model.forward(X_bc[idim][id])
                    x=X_bc[idim][id][:,0:1]
                    y=X_bc[idim][id][:,1:2]
                    phi1 = z[:,0:1]
                    phi2 = z[:,1:2]
                    phi = torch.where((x >= 1.0/3) & (x <= 2.0/3) & (y >= 1.0/3) & (y <= 2.0/3), phi1, phi2) 
                    # phi =   ((x >= 1.0/3) & (x <= 2.0/3) & (y >= 1.0/3) & (y <= 2.0/3))*phi1 + \
                    #         ((x >= 0) & (x <= 1) & (y >= 0) & (y < 1.0/3))*phi2 +\
                    #         ((x >=0) & (x <= 1) & (y > 2.0/3) & (y <= 1.0))*phi2 +\
                    #         ((x >=0) & (x < 1.0/3) & (y >= 1.0/3) & (y <= 2.0/3))*phi2 +\
                    #         ((x >2.0/3) & (x <= 1) & (y >= 1.0/3) & (y <= 2.0/3))*phi2

                    # phi = torch.where(((x>=0.5)&(y<=0.5)) + ((x<=0.5)&(y>=0.5)), phi1, phi2) # Dauge

                    g_D = torch.zeros_like(phi)
                    residu = phi-g_D
                else:
                    # print('Check again boundary condition')
                    residu = torch.zeros_like(X_bc[idim][id])[:,0:1]
                list_residu_BC.append(residu)
        residu_BC = torch.vstack(list_residu_BC)
        return residu_BC
    
    def interface_jump(self,X_interface):
        ndim = len(X_interface)
        list_jump_phi = []
        list_jump_grad = []
        for idim in range(ndim):
            z = self.model.forward(X_interface[idim])
            phi1 = z[:,0:1]
            phi2 = z[:,1:2]
            grad1 = operator.grad(phi1,X_interface[idim])
            grad2 = operator.grad(phi2,X_interface[idim])
            jump_grad_ij = 2.0*grad1[:,idim]-1.0*grad2[:,idim]
            jump_phi_ij  = phi1-phi2

            list_jump_phi.append(jump_phi_ij)
            list_jump_grad.append(jump_grad_ij)
        jump_phi = torch.vstack(list_jump_phi)
        jump_grad = torch.vstack(list_jump_grad)

        loss_jump_phi  = torch.mean(jump_phi**2)
        loss_jump_grad = torch.mean(jump_grad**2)

        error_interface = loss_jump_phi + loss_jump_grad
        return error_interface


    def phi_predict(self,X_test,ngroup=1):
        # print('Subclass')
        if ngroup ==1:
            x=X_test[:,0:1]
            y=X_test[:,1:2]
            z = self.model.forward(X_test)
            phi1 = z[:,0:1]
            phi2 = z[:,1:2]
            phi = torch.where((x >= 1.0/3) & (x <= 2.0/3) & (y >= 1.0/3) & (y <= 2.0/3), phi1, phi2) 
            # phi = torch.where(((x>=0.5)&(y<=0.5)) + ((x<=0.5)&(y>=0.5)), phi1, phi2) # Dauge
        else:
            # ndim = self.domain.input_dim
            print('Not implemented')

        return phi
    
    

    # def get_phi_test(self,X_test,phi_test,normalization=False,ngroup=None):
    #     if ngroup is None:
    #         phi_pred = self.phi_predict(X_test)
    #         mass_phi = torch.sum(phi_pred)
    #         if normalization:
    #             phi_pred = phi_pred/mass_phi
        
    #         phi_AE = torch.abs(phi_test-phi_pred)
    #         phi_REL_L2 = torch.linalg.vector_norm(phi_test-phi_pred)/torch.linalg.vector_norm(phi_test)
            
    #     else:
    #         print('Not implemented')
           
    #     return phi_REL_L2, phi_AE, phi_pred, phi_test,mass_phi
    

#===================================================================================================================
class PC_Dauge_primal_diffusion_one_group(pdeBase):
    def __init__(self,domain,model,params_pde,device):
        super().__init__(domain,model,device)
        self.params_pde = params_pde
        
    
    def residual_PDE(self,X):
        D = self.params_pde['func_D'](X).to(self.device)
        sigma_a = self.params_pde['func_Sigma_a'](X).to(self.device)
        s = self.params_pde['func_Source'](X).to(self.device)
        z = self.model.forward(X).to(self.device)
        
        x=X[:,0:1]
        y=X[:,1:2]
        phi1 = z[:,0:1]
        phi2 = z[:,1:2]
        # phi = torch.where((x >= 1.0/3) & (x <= 2.0/3) & (y >= 1.0/3) & (y <= 2.0/3), phi1, phi2) # Center

        phi = torch.where(((x>=0.5)&(y<=0.5)) + ((x<=0.5)&(y>=0.5)), phi1, phi2) # Dauge
                
    
        eq = -operator.div(D*operator.grad(phi,X),X) +sigma_a*phi -s
        return eq


    
    def residual_BC(self, X_bc):
        # define residual of each boundary
        ndim = len(X_bc)
        list_residu_BC = []
        for idim in range(ndim):
            for id in range(2):
                if self.domain.boundary_conditions[idim][id] == 'ZERO_FLUX':
                    z = self.model.forward(X_bc[idim][id])
                    x=X_bc[idim][id][:,0:1]
                    y=X_bc[idim][id][:,1:2]
                    phi1 = z[:,0:1]
                    phi2 = z[:,1:2]
                    # phi = torch.where((x >= 1.0/3) & (x <= 2.0/3) & (y >= 1.0/3) & (y <= 2.0/3), phi1, phi2) 
                    phi = torch.where(((x>=0.5)&(y<=0.5)) + ((x<=0.5)&(y>=0.5)), phi1, phi2) # Dauge

                    g_D = torch.zeros_like(phi)
                    residu = phi-g_D
                else:
                    # print('Check again boundary condition')
                    residu = torch.zeros_like(X_bc[idim][id])[:,0:1]
                list_residu_BC.append(residu)
        residu_BC = torch.vstack(list_residu_BC)
        return residu_BC
    
    def interface_jump(self,X_interface):
        ndim = len(X_interface)
        list_jump_phi = []
        list_jump_grad = []
        for idim in range(ndim):
            z = self.model.forward(X_interface[idim])
            phi1 = z[:,0:1]
            phi2 = z[:,1:2]
            grad1 = operator.grad(phi1,X_interface[idim])
            grad2 = operator.grad(phi2,X_interface[idim])
            jump_grad_ij = 2.0*grad1[:,idim]-1.0*grad2[:,idim]
            jump_phi_ij  = phi1-phi2

            list_jump_phi.append(jump_phi_ij)
            list_jump_grad.append(jump_grad_ij)
        jump_phi = torch.vstack(list_jump_phi)
        jump_grad = torch.vstack(list_jump_grad)

        loss_jump_phi  = torch.mean(jump_phi**2)
        loss_jump_grad = torch.mean(jump_grad**2)

        error_interface = loss_jump_phi + loss_jump_grad
        return error_interface


    def phi_predict(self,X_test,ngroup=1):
        # print('Subclass')
        if ngroup ==1:
            x=X_test[:,0:1]
            y=X_test[:,1:2]
            z = self.model.forward(X_test)
            phi1 = z[:,0:1]
            phi2 = z[:,1:2]
            # phi = torch.where((x >= 1.0/3) & (x <= 2.0/3) & (y >= 1.0/3) & (y <= 2.0/3), phi1, phi2) 
            phi = torch.where(((x>=0.5)&(y<=0.5)) + ((x<=0.5)&(y>=0.5)), phi1, phi2) # Dauge
        else:
            # ndim = self.domain.input_dim
            print('Not implemented')

        return phi
    
    

    # def get_phi_test(self,X_test,phi_test,normalization=False,ngroup=None):
    #     if ngroup is None:
    #         phi_pred = self.phi_predict(X_test)
    #         mass_phi = torch.sum(phi_pred)
    #         if normalization:
    #             phi_pred = phi_pred/mass_phi
        
    #         phi_AE = torch.abs(phi_test-phi_pred)
    #         phi_REL_L2 = torch.linalg.vector_norm(phi_test-phi_pred)/torch.linalg.vector_norm(phi_test)
            
    #     else:
    #         print('Not implemented')
           
    #     return phi_REL_L2, phi_AE, phi_pred, phi_test,mass_phi