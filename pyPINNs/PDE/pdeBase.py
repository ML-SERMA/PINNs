import torch
import numpy as np
import time
import seaborn as sns
import matplotlib.pyplot as plt


class pdeBase():
    def __init__(self,domain,model,device):
        super().__init__() 
        self.device = device
        # bounds
        self.domain = domain
        # self.lb = self.domain.min_values.float().to(device)
        # self.ub = self.domain.max_values.float().to(device)
        
        self.best_train = np.inf
        self.best_test  = np.inf
        self.optimizer = None
        self.model = model.to(self.device)

        ## loss function
        self.loss_function = torch.nn.MSELoss(reduction ='mean')
        self.iter = 1
        # ADAPTIVE 
        self.beta = None
        self.adaptive_constant = None
        self.stopping = False
        self.do_outer = False

    def Dirichlet_BC(self,X):
        """
        Raises:NotImplemented: Indicates that the function needs to be implemented in a subclass.
        """
        NotImplemented("The function Dirichlet_BC is not implemented") 

    def residual_BC(self,X_bc):
        """
        Raises:NotImplemented: Indicates that the function needs to be implemented in a subclass.
        """
        NotImplemented("The function Dirichlet_BC is not implemented")  
        
    def loss_BC(self,X_bc):

        residu_bc = self.residual_BC(X_bc)
        loss_bc = torch.mean((residu_bc)**2)
        return loss_bc

    def loss_EXP(self,X_p,u_p):
        loss_exp = self.loss_function(self.model.forward(X_p)[:,0:1], u_p)
        return loss_exp    

    def list_of_derivatives(self, u, x, order=1):
        derivate_old = torch.autograd.grad(u, x,torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        derivate     = derivate_old
        for i in range(order-1):
            derivate_old = torch.autograd.grad(derivate_old, x, torch.ones_like(u), retain_graph=True, create_graph=True)[0]
            derivate     = torch.cat( (derivate, derivate_old), dim=1 )
        return derivate
    
    def residual_PDE(self,X):
        """
        Raises:NotImplemented: Indicates that the function needs to be implemented in a subclass.
        """
        NotImplemented("The function residual_PDE is not implemented")

    def loss_PDE(self,X_train):
        residu = self.residual_PDE(X_train)
        # loss_f = torch.mean((residu)**2)
        # loss_Field = torch.mean(torch.pow(residu,2),dim=0)
        loss_f = torch.mean(torch.sum(torch.pow(residu,2),dim=1,keepdim=True))
        return loss_f
        
    def classical_loss(self,X_train,X_bc):
        loss_bc  = self.loss_BC(X_bc)
        loss_f   = self.loss_PDE(X_train)
        loss_val = loss_bc + loss_f
        return loss_val, loss_bc, loss_f
    
    def compile(self,optimizer,scheduler=None):
        self.optimizer = optimizer
        self.scheduler = scheduler

    def fit(self,data,n_step=100, log_every=100, LossFile='Loss.dat',save_dir = '',**kwargs):
        # Classical training method 
        tin=time.time()

        self.adaptive_constant = 1.0
        def closure():
            self.optimizer.zero_grad()
            loss_bc  = self.loss_BC(data.X_bc)
            loss_f   = self.loss_PDE(data.X_train)
            loss = self.adaptive_constant*(loss_bc) + loss_f
            loss.backward(retain_graph=True)
            
            closure.loss_bc = loss_bc
            closure.loss_f  = loss_f
            return loss   
        
        with open(save_dir+LossFile, 'w') as f:
            f.write(f"Iter Loss Loss_BC Loss_PDE\n")
            for it in range(n_step):
                loss    = self.optimizer.step(closure)
                if self.scheduler is not None:
                    self.scheduler.step()

                loss_bc = closure.loss_bc
                loss_f  = closure.loss_f
                # Test the model and log information 
                if self.iter % log_every == 0:
                    f.write(f"{self.iter:6d} {loss.item():12f} {loss_bc.item():12f} {loss_f.item():12f}\n")
                    if data.phi_test is not None:
                        error_vec = self.get_phi_test(data.X_test,data.phi_test)[0]
                        print(f"{self.iter:6d}: Loss = {loss.item():>12.5f}; Error = {error_vec.item():<12.5f}")
                        # save model according to test set
                        if error_vec.item() < self.best_test:
                            torch.save(self.model.state_dict(),save_dir+ "model_test.pt")
                            self.best_test = error_vec.item()
                    else:
                        print(f"{self.iter:6d}: Loss = {loss.item():>12.5f}; ")
                    
                # Save the model if it improves according to the loss function
                if loss.item() < self.best_train:
                    torch.save(self.model.state_dict(),save_dir+ "model.pt")
                    self.best_train = loss.item()
                self.iter += 1

        tout=time.time()
        print("Elapsed: ",tout-tin," seconds")
        
    #============================================================================================================
    def full_predict(self,X_test):
        pred = self.model.forward(X_test)
        return pred
    def clear_cuda(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    
    # def phi_predict(self,X_test,ngroup=1):
    #     if ngroup ==1:
    #         phi_pred = self.model.forward(X_test)[:,0:1]
    #     else:
    #         ndim = self.domain.input_dim
    #         phi_pred = [self.model.forward(X_test)[:,igroup*(ndim+1):igroup*(ndim+1)+1]  for igroup in range(ngroup)]
    #     return phi_pred
    
    # def currents_predict(self,Xc_test,ngroup=1):
    #     if ngroup ==1:
    #         p_pred = [self.model.forward(Xc_test[idim])[:,idim+1:idim+2] for idim in range(self.domain.input_dim)]
    #     else:
    #         ndim = self.domain.input_dim
    #         p_pred = [[self.model.forward(Xc_test[idim])[:,igroup*(ndim+1) + (idim+1): igroup*(ndim+1) +(idim+2)] for idim in range(ndim)] for igroup in range(ngroup)]
    #     return p_pred

    # def get_phi_test(self,X_test,phi_test,normalization=False,ngroup=None):
    #     if ngroup is None:
    #         phi_pred = self.phi_predict(X_test)
    #         mass_phi_pred = torch.sum(phi_pred)
    #         mass_phi_test = torch.sum(phi_test)
    #         if normalization:
    #             phi_pred = phi_pred/mass_phi_pred
    #             phi_test = phi_test/mass_phi_test
        
    #         phi_AE = torch.abs(phi_test-phi_pred)
    #         phi_REL_L2 = torch.linalg.vector_norm(phi_test-phi_pred)/torch.linalg.vector_norm(phi_test)
            
    #     else:
    #         phi_pred = self.phi_predict(X_test,ngroup) # mass_phi = [torch.sum(phi_pred[igroup]) for igroup in range(ngroup)]
    #         mass_phi_pred = torch.sum(torch.vstack(phi_pred))
    #         mass_phi_test = torch.sum(torch.vstack(phi_test))
    #         if normalization:
    #             phi_pred = [phi_pred[igroup]/mass_phi_pred for igroup in range(ngroup)]
    #             phi_test = [phi_test[igroup]/mass_phi_test for igroup in range(ngroup)]
        
    #         phi_AE = [torch.abs(phi_test[igroup]-phi_pred[igroup]) for igroup in range(ngroup)]
    #         phi_REL_L2 = torch.linalg.vector_norm(torch.vstack(phi_test)-torch.vstack(phi_pred))/torch.linalg.vector_norm(torch.vstack(phi_test)) 
 

    #     return phi_REL_L2, phi_AE, phi_pred, phi_test,mass_phi_pred, mass_phi_test
    
    # def get_currents_test(self,Xc_test,p_test,normalization=False,mass_pred=1.0,mass_test=1.0,ngroup=None):
    #     if ngroup is None:
    #         p_pred = self.currents_predict(Xc_test)
    #         if normalization:
    #             p_pred = [ p_pred[idim]/mass_pred for idim in range(self.domain.input_dim)]
    #             p_test = [ p_test[idim]/mass_test for idim in range(self.domain.input_dim)]

    #         p_AE    = [torch.abs(p_test[idim]-p_pred[idim]) for idim in range(self.domain.input_dim)] # list
    #         p_REL_L2 =  torch.linalg.vector_norm(torch.vstack(p_AE))/torch.linalg.vector_norm(torch.vstack(p_test)) # scalar

    #     else:
    #         p_pred = self.currents_predict(Xc_test,ngroup)
    #         if normalization:
    #             p_pred = [ [ p_pred[igroup][idim]/mass_pred for idim in range(self.domain.input_dim)] for igroup in range(ngroup)]
    #             p_test = [ [ p_test[igroup][idim]/mass_test for idim in range(self.domain.input_dim)] for igroup in range(ngroup)]

    #         p_AE    = [[torch.abs(p_test[igroup][idim]-p_pred[igroup][idim]) for idim in range(self.domain.input_dim)] for igroup in range(ngroup)]# list

    #         p_AE_g  = [ torch.vstack(p_AE[igroup])    for igroup in range(ngroup)]
    #         p_test_g = [torch.vstack(p_test[igroup])    for igroup in range(ngroup)]
    #         p_REL_L2 =  torch.linalg.vector_norm(torch.vstack(p_AE_g))/torch.linalg.vector_norm(torch.vstack(p_test_g)) 
    #         # p_REL_L2 =  [torch.linalg.vector_norm(torch.vstack(p_AE[igroup]))/torch.linalg.vector_norm(torch.vstack(p_test[igroup])) for igroup in range(ngroup)] 
    #         # print(f"==>> p_REL_L2: {p_REL_L2}")

    #     return p_REL_L2, p_AE, p_pred, p_test
    
    #===========================================================================================================================
    
    
    
    # def adaptive_sampling(self,X,method='RAR-G'):
    #     '''
    #     method:
    #             Residual based adaptive refinement with greedy (RAR-G)
    #             Residual based adaptive distribution (RAD)
    #             Residual based adaptive distribution with greedy (RAD-G)
    #             Evolutionary sampling (EVO):

    #     '''
    #     if method =='RAR':
    #         # X_test = generator_points(n_samples=100,n_dim=2,random_seed=2024,type_of_points='random')
    #         # X_test = X_test*(self.ub-self.lb) + self.lb

    #         X_test = self.domain.add_collocation_points(n_collocation=4000,random_seed=None)
    #         X_test = X_test.requires_grad_().float().to(self.device)
    #         residu = self.residual_PDE(X_test)
    #         residu = torch.pow(residu,2).to(self.device)
    #         residu = torch.sum(residu,1) # 1D tesnor
    #         _, indices = torch.topk(residu,k=1024,dim=0,sorted=False)
    #         X_new = X_test[indices]
    #         X_new = X_new.clone().detach().requires_grad_(True)
    #         X = torch.vstack((X,X_new))
    #         print(f"==>> X-RAD: {X.shape}")

    #     elif method == 'RAD':

    #         greedyNum = 256
    #         X_test = self.domain.add_collocation_points(n_collocation=1024,random_seed=None)
    #         X_test = X_test.requires_grad_().float().to(self.device)
    #         residu = self.residual_PDE(X_test)
    #         residu = torch.pow(residu,2).to(self.device)
    #         residu = torch.sum(residu,1) # 1D tensor
    #         distribution = residu/torch.sum(residu)
    #         indices = torch.multinomial(distribution,greedyNum)
    #         X_new = X_test[indices]
    #         X_new = X_new.clone().detach().requires_grad_(True)
    #         X = torch.vstack((X,X_new))
    #         print(f"==>> X-RAD: {X.shape}")

    #     elif method == 'EVO':
    #         residu = self.residual_PDE(X)
    #         residu = torch.pow(residu,2).to(self.device)
    #         residu = torch.sum(residu,1) # 1D tensor
    #         threshold = torch.mean(residu)
    #         indices = torch.nonzero((residu>=threshold)*1,as_tuple=True)[0]
    #         # myindices = torch.stack(list(torch.nonzero((residu>=threshold)*1,as_tuple=True)[0]),dim=0)
    #         X_evo = X[indices]
    #         X_evo = X_evo.clone().detach().requires_grad_(True)

    #         n_rand = X.shape[0]-X_evo.shape[0]
    #         X_rand = self.domain.add_collocation_points(n_collocation=n_rand,random_seed=None)
    #         X_rand = X_rand.requires_grad_().float().to(self.device) 

    #         X = torch.vstack((X_evo,X_rand))
    #         X = X.clone().detach().requires_grad_(True)
    #         print(f"==>> X-EVO: {X.shape}")
            
    #     else:
    #         print('Please check again the method for adaptive sampling !')
    #     return X
    
   