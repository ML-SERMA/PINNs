import numpy as np
import torch
from torch.utils.data import DataLoader
import pickle
from ..Mesh.CartesianMesh import CartesianMesh
from ..Tools.quadrature import quadrature


class DataSet:
    def __init__(self,domain,n_collocation,n_boundary,n_test,random_seed,device):
        self.domain = domain
        self.n_collocation = n_collocation
        self.n_boundary = n_boundary
        self.n_test = n_test
        self.random_seed = random_seed
        self.device = device
        self.data_collocation = None
        self.data_boundary = None
        self.X_interface = None
        self.phi_test = None
        self.p_test = None
        self.generate_collocation_points(self.n_collocation,self.random_seed,self.device)
        self.generate_boundary_points(self.n_boundary,self.random_seed,self.device)
        self.generate_test_points(self.n_test,self.device)
        self.generate_currents_points(self.n_test, self.device)
        


    def generate_collocation_points(self,n_collocation,random_seed,device):
        X_train = self.domain.add_collocation_points(n_collocation=n_collocation,random_seed=random_seed)
        self.X_train = X_train.requires_grad_().float().to(device) 

        
    def generate_boundary_points(self,n_boundary,random_seed,device):
        X_bc= self.domain.add_boundary_points(n_boundary=n_boundary,random_seed=random_seed)
        self.X_bc = [ [X_bc[idim][id].requires_grad_().float().to(device) for id in range(2)] for idim in range(len(X_bc))]
        # self.X_bc_all   = X_bc_all.requires_grad_().float().to(device)


    def generate_interface_points(self,n_face,test):
        X_interface = self.domain.add_interface_points(n_face,test,self.random_seed)  
        self.X_interface = [X_interface[idim].requires_grad_().float().to(self.device)  for idim in range(len(X_interface))]
        print(f"==>> X_interface: {X_interface[0].shape,X_interface[1].shape}")

    
    def generate_test_points(self,n_test,device):
        # params_mesh = params_mesh = {'ndim':2, 'xmin':[0, 0], 'xmax':[100, 100], 'nmail':[n_test, n_test]}
        # mesh = CartesianMesh(**params_mesh)
        # X, Y     = np.meshgrid(np.array(mesh.xmid[0]), np.array(mesh.xmid[1]))
        # X_test   = np.hstack((X.flatten()[:,None], Y.flatten()[:,None]))  
        # self.X_test   = torch.tensor(X_test , requires_grad=False).float().to(device) # Test points
        
        X_test = self.domain.add_test_points(n_test=n_test)
        self.X_test = X_test.requires_grad_(True).float().to(device)
        
    def generate_currents_points(self,n_test,device):
        Xc_test = self.domain.add_currents_points(n_test=n_test)
        self.Xc_test = [Xc_test[idim].requires_grad_(False).float().to(device) for idim in range(self.domain.input_dim)]

    def generate_phi_test(self,f_exact=None,load_dir=None,multigroup=False):
        if multigroup:
            print(' Multigroup case')
            if load_dir is not None:
                print('Load reference solution for multigroup!')
                phi_test   = pickle.load(open(load_dir,"rb"))[0] # for one subdomain
                ngroup = phi_test.shape[0]
                self.phi_test   = [torch.tensor(phi_test[ig].reshape((-1,1)), requires_grad=False).float().to(self.device) for ig in range(ngroup)]
                # print(f"==>> phi_test: {self.phi_test[0].shape,self.phi_test[1].shape}")
            else:
                print('Set default value for phi_test')
                self.phi_test = None

        else:
            print(' Monogroup case')
            if load_dir is not None:
                print('Load reference solution for monogroup!')
                phi_test   = pickle.load(open(load_dir,"rb"))
                # self.phi_test   = torch.tensor(phi_test, requires_grad=False).float().to(self.device)
                # Load monogroup like multigroup
                self.phi_test   = [torch.tensor(phi_test, requires_grad=False).float().to(self.device)]
            elif f_exact is not None:
                print('Evaluate exact solution!')
                self.phi_test = f_exact(self.X_test)
            else:
                print('Set default value for phi_test')
                self.phi_test = None
            
    def generate_p_test(self,load_dir=None,multigroup=False):
        if multigroup:
            print(' Multigroup case')
            if load_dir is not None:
                print('Load reference currents!')
                p_test   = pickle.load(open(load_dir,"rb"))[0] # for one  subdomain
                ngroup = len(p_test)
                self.p_test   = [ [torch.tensor(p_test[ig][idim].reshape((-1,1)), requires_grad=False).float().to(self.device) 
                                for idim in range(self.domain.input_dim)] for ig in range(ngroup)]
            else:
                print('Set default value for p_test')
                self.p_test = None   
        else:
            print(' Monogroup case')
            if load_dir is not None:
                print('Load reference currents!')
                p_test   = pickle.load(open(load_dir,"rb"))
                # self.p_test   = [torch.tensor(p_test[idim], requires_grad=False).float().to(self.device) for idim in range(self.domain.input_dim)]
                #  Load monogroup like multigroup
                self.p_test   = [[torch.tensor(p_test[idim], requires_grad=False).float().to(self.device) for idim in range(self.domain.input_dim)]]
            else:
                print('Set default value for p_test')
                self.p_test = None       
            

        
    def prepare_DataLoader(self,batch_size_collocation=None,batch_size_boundary=None): ## Do not use this function for the moment!
        if batch_size_collocation is None:
            self.batch_size_collocation = self.X_train.shape[0]
        else:
            self.batch_size_collocation = batch_size_collocation
        if batch_size_boundary is None:
            self.batch_size_boundary = self.X_bc.shape[0]
        else:
            self.batch_size_boundary = batch_size_boundary

        self.data_collocation = DataLoader(torch.utils.data.TensorDataset(self.X_train), batch_size=self.batch_size_collocation, shuffle=True)
        self.data_boundary = DataLoader(torch.utils.data.TensorDataset(self.X_bc), batch_size=self.batch_size_boundary, shuffle=True)


    def update_random_collocation_points(self,random_seed=None):
        self.generate_collocation_points(n_collocation=self.n_collocation,random_seed=random_seed,device=self.device)
        self.generate_boundary_points(n_boundary=self.n_boundary,random_seed=random_seed,device=self.device)

    def generate_exp_points(self):

        self.X_exp = self.X_test.clone().detach().requires_grad_(True)
        self.phi_exp = self.phi_test

    def generate_angular_collocation_points(self,n_mu=4,n_phi=4,dim_angular=2,mu_rule ='legendre',phi_rule='legendre'):
        self.s_train,_ = quadrature.angular_quadrature(n_mu=n_mu,n_phi=n_phi,dim_angular=dim_angular,
                                                    mu_rule = mu_rule,phi_rule=phi_rule,device=self.device)
        
    def generate_angular_boundary_points(self,n_mu=4,n_phi=4,dim_angular=2,mu_rule ='legendre',phi_rule='legendre'):
        self.s_bc,_ = quadrature.angular_quadrature(n_mu=n_mu,n_phi=n_phi,dim_angular=dim_angular,
                                                    mu_rule = mu_rule,phi_rule=phi_rule,device=self.device)   
    def combine_space_angular_points(self,mode='auto'):
        '''
        This function is used to combine space points and angular points for the multigroup neutron transport equation

        '''
        assert self.X_train.shape[1] == self.s_train.shape[1], "Mismatch in spatial/angular dimensions"
        if mode == 'paired' and self.X_train.shape[0] == self.s_train.shape[0]:
            print(f"==>>  mode train: {mode}")
            xs_train = torch.cat([self.X_train, self.s_train], dim=1)
        else:
            # Repeat the angular directions for each input point
            x_in = self.X_train.repeat_interleave(self.s_train.shape[0], dim=0) # [N × Q_train, d]
            s_in = self.s_train.repeat(self.n_collocation, 1)                # [N × Q_train, d]
            xs_train = torch.cat([x_in, s_in], dim=1)      # [N × Q_train, 2d] 

        self.X_train =  xs_train.requires_grad_().float().to(self.device) 
        # for boundary points
        XS_bc = []
        for dim in range(len(self.X_bc)):
            XS_bc_dim = []
            for side in range(2):  # 0: min side, 1: max side
                x_b = self.X_bc[dim][side]  # [N, d]
                if mode == 'paired' and x_b.shape[0] == self.s_bc.shape[0]:
                    print(f"==>>  mode bc: {mode}")
                    xs_bc = torch.cat([x_b, self.s_bc], dim=1)
                else:
                    # Repeat directions and spatial points
                    x_in = x_b.repeat_interleave(self.s_bc.shape[0], dim=0)       # [N*Q, d]
                    s_in = self.s_bc.repeat(x_b.shape[0], 1)                   # [N*Q, d]
                    xs_bc = torch.cat([x_in, s_in], dim=1)  # [N*Q, 2d]
                XS_bc_dim.append(xs_bc)    
            XS_bc.append(XS_bc_dim)

        self.X_bc = [ [XS_bc[idim][id].requires_grad_().float().to(self.device) for id in range(2)] for idim in range(len(XS_bc))]
        
        






    

    

