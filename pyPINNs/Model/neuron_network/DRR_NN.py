import torch
import torch.nn as nn
from ..layers.ResidualBlock import ResBasicBlock
import copy


class DeepRitzResNN(nn.Module):
    """
    drrnn -- Deep Ritz Residual Neural Network

    Implements a network with the architecture used in the
    deep ritz method paper

    Parameters:
        in_N  -- input dimension
        out_N -- output dimension
        width     -- width of layers that form blocks
        depth -- number of blocks to be stacked
        phi   -- the activation function
    """

    def __init__(self, in_N, width, depth, out_N,activation ='Tanh',hard_BC = False):
        super(DeepRitzResNN, self).__init__()
        # set parameters
        self.in_N = in_N
        self.width = width
        self.out_N = out_N
        self.depth = depth
        self.activation = activation
        self.hard_BC = hard_BC

        # list for holding all the blocks
        self.stacks = nn.ModuleList()

        # add first layer to list
        self.stacks.append(nn.Linear(in_N, width))

        # add middle blocks to list
        for i in range(depth):
            self.stacks.append(ResBasicBlock(width, width, width,activation))

        # add output linear layer
        self.stacks.append(nn.Linear(width, out_N))

    def forward(self, x):
        if torch.is_tensor(x) != True:         
            x = torch.from_numpy(x) 
        a = x.float()
        # first layer
        for i in range(len(self.stacks)):
            a = self.stacks[i](a)
        # if self.hard_BC:
        #     xt =  x[:,0:1]
        #     yt  = x[:,1:2]
        #     at  = copy.copy(a[:,0:1])
        #     a[:,0:1] = at*xt*(1-xt)*yt*(1-yt)
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
            elif self.hard_BC == '2G_C5G7':
                xy_phi = (64.26-x[:,0:1])/64.26*(64.26-x[:,1:2])/64.26
                xy_qx   = x[:,0:1]/64.26
                xy_qy   = x[:,1:2]/64.26
                at = torch.hstack((xy_phi,xy_qx,xy_qy,xy_phi,xy_qx,xy_qy))
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