
import torch
from ..activations import get_activation
from ..layers.ChebyKANLayer import ChebyKANLayer




class ChebyKAN(torch.nn.Module):
    def __init__(self,layer_dims,activation,degree,device,hard_BC =None):
        super().__init__() 
        self.device = device
        self.hard_BC = hard_BC
        self.layer_dims = layer_dims
        self.degree = degree
        ## activation function
        self.activation = get_activation(activation)
        self.linears = torch.nn.ModuleList([ChebyKANLayer(self.layer_dims[i], self.layer_dims[i+1],self.degree) for i in range(len(self.layer_dims)-1)])
        
    ## forward pass
    def forward(self,x):
        if torch.is_tensor(x) != True:         
            x = torch.from_numpy(x) 
        
        a = x.float()
        for i in range(len(self.layer_dims)-2):
            z = self.linears[i](a)
            a = self.activation(z)
        a = self.linears[-1](a)

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
            else:
                print('Type of tranformation is not implemented !')
        return a