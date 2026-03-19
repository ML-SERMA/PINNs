import torch
from ..activations import get_activation
from ..layers.FourierFeature import FourierFeatureMapping
from ..layers.RBF_layer import RBF_Layer
from ..layers.RBF_kernels import gaussian
from ..layers.AdaptiveLinear import AdaptiveLinear
from ..layers.FA_layer import LinearFA
import copy


def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)


class FCN(torch.nn.Module):
    def __init__(self,layers,activation,device,hard_BC =None):
        super().__init__() 
        self.device = device
        self.hard_BC = hard_BC
        self.layers = layers
        ## activation function
        self.activation = get_activation(activation)
        self.linears = torch.nn.ModuleList([torch.nn.Linear(self.layers[i], self.layers[i+1]) for i in range(len(self.layers)-1)])
        ## Xavier Normal Initialization
        for i in range(len(self.layers)-1):
            torch.nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            # set biases to zero
            torch.nn.init.zeros_(self.linears[i].bias.data)
    ## forward pass
    def forward(self,x):
        if torch.is_tensor(x) != True:         
            x = torch.from_numpy(x) 
        

        a = x.float()
        for i in range(len(self.layers)-2):
            z = self.linears[i](a)
            a = self.activation(z)
        a = self.linears[-1](a)
        if self.hard_BC is not None:
            a = a*x[:,0:1]*(1-x[:,0:1])*x[:,1:2]*(1-x[:,1:2])
        return a


class FCN_FF(torch.nn.Module):
    def __init__(self,layers,activation,device,fourier_mapping_size=None,hard_BC =None):
        super().__init__() 
        self.device = device
        self.fourier_mapping_size = fourier_mapping_size
        self.hard_BC = hard_BC
        if fourier_mapping_size is not None:
            self.fourier_mapping = FourierFeatureMapping(layers[0], fourier_mapping_size=fourier_mapping_size, scale=1)
            input_dim = 2 * fourier_mapping_size
        else:
            input_dim = layers[0]
        
        self.layers =  [input_dim] + layers[1:]
        ## activation function
        self.activation = get_activation(activation)
        self.linears = torch.nn.ModuleList([torch.nn.Linear(self.layers[i], self.layers[i+1]) for i in range(len(self.layers)-1)])
        ## Xavier Normal Initialization
        for i in range(len(self.layers)-1):
            torch.nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            # set biases to zero
            torch.nn.init.zeros_(self.linears[i].bias.data)
    ## forward pass
    def forward(self,x):
        if torch.is_tensor(x) != True:         
            x = torch.from_numpy(x) 
        if self.hard_BC == '2G_C5G7_N':
            a = x.float()/64.26 # normalization
        else:
            a = x.float()
        # fourier feature
        if self.fourier_mapping_size is not None:
            a = self.fourier_mapping(a)
        
        for i in range(len(self.layers)-2):
            z = self.linears[i](a)
            a = self.activation(z)
        a = self.linears[-1](a)
        
        # zero_flux boundary condition
        if self.hard_BC is not None:
            if self.hard_BC == 'primal_L1':
                a = a*x[:,0:1]*(1-x[:,0:1])*x[:,1:2]*(1-x[:,1:2])
            elif self.hard_BC == 'primal_L1_PC':
                xt = x[:,0:1]*(1-x[:,0:1])
                yt = x[:,1:2]*(1-x[:,1:2])
                xyt = xt*yt
                at = torch.hstack((xyt,xyt))
                a = a*at

            elif self.hard_BC == 'primal_L1_D':
                xt = x[:,0:1]*(1-x[:,0:1])
                yt = x[:,1:2]*(1-x[:,1:2])
                xyt = xt*yt
                at = torch.hstack((xyt,torch.ones(x.shape[0],1).to(self.device)))
                a = a*at
                
            elif self.hard_BC == 'mixed_L1':
                xt = x[:,0:1]*(1-x[:,0:1])
                yt = x[:,1:2]*(1-x[:,1:2])
                xyt = xt*yt
                at = torch.hstack((xyt,torch.ones(x.shape[0],2).to(self.device)))
                a = a*at
            elif self.hard_BC == 'mixed_L100':
                xt = x[:,0:1]*(100.-x[:,0:1])/(100*100)
                yt = x[:,1:2]*(100.-x[:,1:2])/(100*100)
                xyt = xt*yt
                at = torch.hstack((xyt,torch.ones(x.shape[0],2).to(self.device)))
                a = a*at
            elif self.hard_BC == 'mixed_vacuum_96_86':
                xt = x[:,0:1]*(96.-x[:,0:1])/(96*96)
                yt = x[:,1:2]*(86.-x[:,1:2])/(86*86)
                phi = a[:,0:1]
                Gx = (2.0/96.0*x[:,0:1]-1.0)*(0.5*phi)
                Gy = (2.0/86.0*x[:,1:2]-1.0)*(0.5*phi)
                at = torch.hstack([torch.ones(x.shape[0],1).to(self.device),xt,yt])
                G  = torch.hstack([torch.zeros(x.shape[0],1).to(self.device),Gx,Gy])              
                a = a*at+G
            elif self.hard_BC == 'two_group_mixed_zero_L7':
                xt = x[:,0:1]*(7.-x[:,0:1])/(7*7)
                yt = x[:,1:2]*(7.-x[:,1:2])/(7*7)
                xyt = xt*yt
                at = torch.hstack((xyt,torch.ones(x.shape[0],2).to(self.device),xyt,torch.ones(x.shape[0],2).to(self.device)))
                a= a*at
            elif self.hard_BC == '2G_C5G7' or self.hard_BC == '2G_C5G7_N':
                xy_phi = (64.26-x[:,0:1])/64.26*(64.26-x[:,1:2])/64.26
                xy_qx   = x[:,0:1]/64.26
                xy_qy   = x[:,1:2]/64.26
                at = torch.hstack((xy_phi,xy_qx,xy_qy,xy_phi,xy_qx,xy_qy))
                a= a*at
                
            elif self.hard_BC == '2G_C5G7_L1':
                xy_phi = (1.0-x[:,0:1])*(1.0-x[:,1:2])
                xy_qx   = x[:,0:1]
                xy_qy   = x[:,1:2]
                at = torch.hstack((xy_phi,xy_qx,xy_qy,xy_phi,xy_qx,xy_qy))
                a= a*at
            elif self.hard_BC == 'TWIGL_2D':
                xy_phi = (80.0-x[:,0:1])/80.0*(80.0-x[:,1:2])/80.0
                xy_qx   = x[:,0:1]/80.0
                xy_qy   = x[:,1:2]/80.0
                at = torch.hstack((xy_phi,xy_qx,xy_qy,xy_phi,xy_qx,xy_qy))
                a= a*at
            elif self.hard_BC == 'TWIGL_3D':
                xyz_phi = (80.0-x[:,0:1])/80.0*(80.0-x[:,1:2])/80.0*(80.0-x[:,2:3])/80.0
                xyz_qx   = x[:,0:1]/80.0
                xyz_qy   = x[:,1:2]/80.0
                xyz_qz   = x[:,2:3]/80.0
                
                at = torch.hstack((xyz_phi,xyz_qx,xyz_qy,xyz_qz,xyz_phi,xyz_qx,xyz_qy,xyz_qz))
                a= a*at    
            elif self.hard_BC == 'curl_mixed_L1':
                xt = x[:,0:1]*(1-x[:,0:1])
                yt = x[:,1:2]*(1-x[:,1:2])
                xyt = xt*yt
                at = torch.hstack((xyt,torch.ones(x.shape[0],4).to(self.device)))
                a = a*at

            elif self.hard_BC == 'mixed_L1_LD':
                xt = x[:,0:1]
                yt = x[:,1:2]
                xyt = xt*yt
                at = torch.hstack((xyt,torch.ones(x.shape[0],2).to(self.device)))
                a = a*at
            elif self.hard_BC == 'mixed_L1_LU':
                xt = x[:,0:1]
                yt = (1-x[:,1:2])
                xyt = xt*yt
                at = torch.hstack((xyt,torch.ones(x.shape[0],2).to(self.device)))
                a = a*at
            elif self.hard_BC == 'mixed_L1_RD':
                xt = (1-x[:,0:1])
                yt = x[:,1:2]
                xyt = xt*yt
                at = torch.hstack((xyt,torch.ones(x.shape[0],2).to(self.device)))
                a = a*at
            elif self.hard_BC == 'mixed_L1_RU':
                xt = (1-x[:,0:1])
                yt = (1-x[:,1:2])
                xyt = xt*yt
                at = torch.hstack((xyt,torch.ones(x.shape[0],2).to(self.device)))
                a = a*at
            elif self.hard_BC == 'NTE_L1_G1':
                
                xt = (x[:,0:1]**2+ torch.abs(x[:,2:3]) - x[:,2:3])*((1-x[:,0:1])**2+ torch.abs(x[:,2:3]) + x[:,2:3])
                yt = (x[:,1:2]**2+ torch.abs(x[:,3:4]) - x[:,3:4])*((1-x[:,1:2])**2+ torch.abs(x[:,3:4]) + x[:,3:4])
                xyt = xt*yt
                a = a*xyt
            elif self.hard_BC == 'NTE_L1_G2':
                
                xt = (x[:,0:1]**2+ torch.abs(x[:,2:3]) - x[:,2:3])*((1-x[:,0:1])**2+ torch.abs(x[:,2:3]) + x[:,2:3])
                yt = (x[:,1:2]**2+ torch.abs(x[:,3:4]) - x[:,3:4])*((1-x[:,1:2])**2+ torch.abs(x[:,3:4]) + x[:,3:4])
                xyt = xt*yt
                at = torch.hstack((xyt,xyt))
                a = a*at
            elif self.hard_BC == 'Takeda1':
                # xt = x[:,0:1]*(25.0-x[:,0:1])/(25*25)
                # yt = x[:,1:2]*(25.0-x[:,1:2])/(25*25)
                # zt = x[:,2:3]*(25.0-x[:,2:3])/(25*25)
                # phi1 = a[:,0:1]
                # phi2 = a[:,4:5]
                # # Gx1 = (x[:,0:1]/25.0)*(2.0/25.0*x[:,0:1]-1.0)*(0.5*phi1)
                # # Gy1 = (x[:,1:2]/25.0)*(2.0/25.0*x[:,1:2]-1.0)*(0.5*phi1)
                # # Gz1 = (x[:,2:3]/25.0)*(2.0/25.0*x[:,2:3]-1.0)*(0.5*phi1)
                # # Gx2 = (x[:,0:1]/25.0)*(2.0/25.0*x[:,0:1]-1.0)*(0.5*phi2)
                # # Gy2 = (x[:,1:2]/25.0)*(2.0/25.0*x[:,1:2]-1.0)*(0.5*phi2)
                # # Gz2 = (x[:,2:3]/25.0)*(2.0/25.0*x[:,2:3]-1.0)*(0.5*phi2)
                # Gx1 = (x[:,0:1]/25.0)*(0.5*phi1)
                # Gy1 = (x[:,1:2]/25.0)*(0.5*phi1)
                # Gz1 = (x[:,2:3]/25.0)*(0.5*phi1)
                # Gx2 = (x[:,0:1]/25.0)*(0.5*phi2)
                # Gy2 = (x[:,1:2]/25.0)*(0.5*phi2)
                # Gz2 = (x[:,2:3]/25.0)*(0.5*phi2)
                # at = torch.hstack([torch.ones(x.shape[0],1).to(self.device),xt,yt,zt,torch.ones(x.shape[0],1).to(self.device),xt,yt,zt])
                # G  = torch.hstack([torch.zeros(x.shape[0],1).to(self.device),Gx1,Gy1,Gz1,torch.zeros(x.shape[0],1).to(self.device),Gx2,Gy2,Gz2])              
                # a = a*at+G

                L = 25.0  # Domain size
                # Split coordinates
                x0 = x[:, 0:1]
                x1 = x[:, 1:2]
                x2 = x[:, 2:3]

                # Interior masking functions
                xt = x0 * (L - x0) / (L * L)
                yt = x1 * (L - x1) / (L * L)
                zt = x2 * (L - x2) / (L * L)

                # Fields
                phi1 = a[:, 0:1]
                phi2 = a[:, 4:5]

                # Robin-like terms (0.5 * phi * x / L)
                grad_weight = 0.5 / L
                G1 = grad_weight * torch.cat([x0, x1, x2], dim=1) * phi1  # [N, 3]
                G2 = grad_weight * torch.cat([x0, x1, x2], dim=1) * phi2  # [N, 3]

                # Build modulation tensor : [N, 8]
                ones = torch.ones_like(phi1)
                at = torch.cat([ones, xt, yt, zt, ones, xt, yt, zt], dim=1)

                # Boundary term : [N, 8]
                zeros = torch.zeros_like(phi1)
                G = torch.cat([zeros, G1, zeros, G2], dim=1)

                # Final result
                a = a * at + G
            else:
                print('Type of tranformation is not implemented !')
        return a




class FCN_RBF(torch.nn.Module):
    def __init__(self,layers,activation,device,fourier_mapping_size=None,hard_BC =None):
        super().__init__() 
        self.device = device
        self.fourier_mapping_size = fourier_mapping_size
        self.hard_BC = hard_BC
        if fourier_mapping_size is not None:
            self.rbf_layer = RBF_Layer(ub=0,lb=0,in_features=2,out_features=fourier_mapping_size,basis_func=gaussian,order_pol=1)
            input_dim = fourier_mapping_size + 2
        else:
            input_dim = layers[0]
        
        self.layers =  [input_dim] + layers[1:]
        ## activation function
        self.activation = get_activation(activation)
        self.linears = torch.nn.ModuleList([torch.nn.Linear(self.layers[i], self.layers[i+1]) for i in range(len(self.layers)-1)])
        ## Xavier Normal Initialization
        for i in range(len(self.layers)-1):
            torch.nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            # set biases to zero
            torch.nn.init.zeros_(self.linears[i].bias.data)
    ## forward pass
    def forward(self,x,lb=None,ub=None):
        if torch.is_tensor(x) != True:         
            x = torch.from_numpy(x) 
        # bounds
        if lb is not None and ub is not None:
            self.lb = lb.float().to(self.device)
            self.ub = ub.float().to(self.device)
            # scale
            x = (x - self.lb)/(self.ub - self.lb)
        a = x.float()
        # fourier feature
        if self.fourier_mapping_size is not None:
            a = self.rbf_layer(a)
            # print(f"==>> a: {a.shape}")
        
        for i in range(len(self.layers)-2):
            z = self.linears[i](a)
            a = self.activation(z)
        a = self.linears[-1](a)
        # zero_flux boundary condition
        if self.hard_BC is not None:
            a = a*x[:,0:1]*(1-x[:,0:1])*x[:,1:2]*(1-x[:,1:2])
        return a
    



    

class FCN_FF_LLAF(torch.nn.Module):
    def __init__(self,layers,activation,device,fourier_mapping_size=None,hard_BC =None):
        super().__init__() 
        self.device = device
        self.fourier_mapping_size = fourier_mapping_size
        self.hard_BC = hard_BC

        if fourier_mapping_size is not None:
            self.fourier_mapping = FourierFeatureMapping(layers[0], fourier_mapping_size=fourier_mapping_size, scale=1)
            input_dim = 2 * fourier_mapping_size
        else:
            input_dim = layers[0]
        
        self.layers =  [input_dim] + layers[1:]
        ## activation function
        self.activation = get_activation(activation)
        # self.linears = torch.nn.ModuleList([torch.nn.Linear(self.layers[i], self.layers[i+1]) for i in range(len(self.layers)-1)])
        self.linears = torch.nn.ModuleList([AdaptiveLinear(self.layers[i], self.layers[i+1],bias=True, adaptive_rate=1.0, adaptive_rate_scaler=1) for i in range(len(self.layers)-1)])
        ## Xavier Normal Initialization
        for i in range(len(self.layers)-1):
            torch.nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            # set biases to zero
            torch.nn.init.zeros_(self.linears[i].bias.data)
    ## forward pass
    def forward(self,x):
        if torch.is_tensor(x) != True:         
            x = torch.from_numpy(x) 
        a = x.float()
        # fourier feature
        if self.fourier_mapping_size is not None:
            a = self.fourier_mapping(a)
        
        for i in range(len(self.layers)-2):
            z = self.linears[i](a)
            a = self.activation(z)
        a = self.linears[-1](a)
        
        # zero_flux boundary condition
        if self.hard_BC is not None:

            xt = x[:,0:1]*(1-x[:,0:1])
            yt = x[:,1:2]*(1-x[:,1:2])
            xyt = xt*yt
            at = torch.hstack((xyt,torch.ones(x.shape[0],2).to(self.device)))
            a = a*at
        return a




# Direct Feedback 
class FCN_FA(torch.nn.Module):
    def __init__(self,layers,activation,device,fourier_mapping_size=None,hard_BC =None):
        super().__init__() 
        self.device = device
        self.fourier_mapping_size = fourier_mapping_size
        self.hard_BC = hard_BC

        if fourier_mapping_size is not None:
            self.fourier_mapping = FourierFeatureMapping(layers[0], fourier_mapping_size=fourier_mapping_size, scale=1)
            input_dim = 2 * fourier_mapping_size
        else:
            input_dim = layers[0]
        
        self.layers =  [input_dim] + layers[1:]
        ## activation function
        self.activation = get_activation(activation)

        self.linears = torch.nn.ModuleList([LinearFA(self.layers[i], self.layers[i+1]) for i in range(len(self.layers)-1)])

        ## Xavier Normal Initialization
        for i in range(len(self.layers)-1):
            torch.nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            # set biases to zero
            torch.nn.init.zeros_(self.linears[i].bias.data)
    ## forward pass
    def forward(self,x):
        if torch.is_tensor(x) != True:         
            x = torch.from_numpy(x) 

        a = x.float()
        # fourier feature
        if self.fourier_mapping_size is not None:
            a = self.fourier_mapping(a)
        
        for i in range(len(self.layers)-2):
            z = self.linears[i](a)
            a = self.activation(z)
        a = self.linears[-1](a)
        
        # zero_flux boundary condition
        if self.hard_BC is not None:
            a = a*x[:,0:1]*(1-x[:,0:1])*x[:,1:2]*(1-x[:,1:2])
        return a


class FCN_RegionBaseActivation(torch.nn.Module):
    def __init__(self,layers,activation,device,fourier_mapping_size=None,hard_BC =None):
        super().__init__() 
        self.device = device
        self.fourier_mapping_size = fourier_mapping_size
        self.hard_BC = hard_BC
        if fourier_mapping_size is not None:
            self.fourier_mapping = FourierFeatureMapping(layers[0], fourier_mapping_size=fourier_mapping_size, scale=1)
            input_dim = 2 * fourier_mapping_size
        else:
            input_dim = layers[0]
        
        self.layers =  [input_dim] + layers[1:]
        ## activation function
        self.activation = get_activation(activation)
        self.linears = torch.nn.ModuleList([torch.nn.Linear(self.layers[i], self.layers[i+1]) for i in range(len(self.layers)-1)])
        ## Xavier Normal Initialization
        for i in range(len(self.layers)-1):
            torch.nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            # set biases to zero
            torch.nn.init.zeros_(self.linears[i].bias.data)
    ## forward pass
    def forward(self,x):
        if torch.is_tensor(x) != True:         
            x = torch.from_numpy(x) 
        x_input = x.clone()  # Keep a copy of the original input
        a = x.float()
        # fourier feature
        if self.fourier_mapping_size is not None:
            a = self.fourier_mapping(a)
        
        for i in range(len(self.layers)-2):
            z = self.linears[i](a)
            a = self.activation(z,x_input)
        a = self.linears[-1](a)
        
        # zero_flux boundary condition
        if self.hard_BC is not None:
            if self.hard_BC == 'primal_L1':
                a = a*x[:,0:1]*(1-x[:,0:1])*x[:,1:2]*(1-x[:,1:2])
            elif self.hard_BC == 'mixed_L1':
                xt = x[:,0:1]*(1-x[:,0:1])
                yt = x[:,1:2]*(1-x[:,1:2])
                xyt = xt*yt
                at = torch.hstack((xyt,torch.ones(x.shape[0],2).to(self.device)))
                a = a*at
            elif self.hard_BC == 'mixed_L100':
                xt = x[:,0:1]*(100.-x[:,0:1])/(100*100)
                yt = x[:,1:2]*(100.-x[:,1:2])/(100*100)
                xyt = xt*yt
                at = torch.hstack((xyt,torch.ones(x.shape[0],2).to(self.device)))
                a = a*at
            elif self.hard_BC == 'mixed_vacuum_96_86':
                xt = x[:,0:1]*(96.-x[:,0:1])/(96*96)
                yt = x[:,1:2]*(86.-x[:,1:2])/(86*86)
                phi = a[:,0:1]
                Gx = (2.0/96.0*x[:,0:1]-1.0)*(0.5*phi)
                Gy = (2.0/86.0*x[:,1:2]-1.0)*(0.5*phi)
                at = torch.hstack([torch.ones(x.shape[0],1).to(self.device),xt,yt])
                G  = torch.hstack([torch.zeros(x.shape[0],1).to(self.device),Gx,Gy])              
                a = a*at+G
            elif self.hard_BC == 'two_group_mixed_zero_L7':
                xt = x[:,0:1]*(7.-x[:,0:1])/(7*7)
                yt = x[:,1:2]*(7.-x[:,1:2])/(7*7)
                xyt = xt*yt
                at = torch.hstack((xyt,torch.ones(x.shape[0],2).to(self.device),xyt,torch.ones(x.shape[0],2).to(self.device)))
                a= a*at
            elif self.hard_BC == 'curl_mixed_L1':
                xt = x[:,0:1]*(1-x[:,0:1])
                yt = x[:,1:2]*(1-x[:,1:2])
                xyt = xt*yt
                at = torch.hstack((xyt,torch.ones(x.shape[0],4).to(self.device)))
                a = a*at
            else:
                print('Type of tranformation is not implemented !')
        return a