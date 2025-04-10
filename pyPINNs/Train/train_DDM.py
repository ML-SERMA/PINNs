import torch
import time
from tqdm import tqdm
import os
from ..Tools.visualization import visualization
from ..Tools.basic_utils import getDomainIndex, fromIndexToDomain
from .train_model import train

def compute_l1_loss(w):
    return torch.abs(w).sum()

def compute_l2_loss(w):
    return torch.square(w).sum()


def update_boundary(gridDDM,subdo_pde,subdo_data):
    with torch.no_grad():
        ndim = len(gridDDM)
        nDD = len(subdo_pde)
        for iDD in range(nDD):
            index_I = getDomainIndex(ideg=iDD,sdeg=gridDDM)
            subdo_pde[iDD].g_bc = []
            for idim in range(ndim):
                g_bc_idim = []
                for id in range(2): # left or right
                    g_ij = None
                    index_J = index_I
                    if subdo_data[iDD].domain.boundary_conditions[idim][id] == 'INTERFACE':
                        n = (-1)**(id+1) # square domain
                        index_J[idim] = index_J[idim] + n 
                        # Find neighbor domain
                        jDD = fromIndexToDomain(gridDDM,index_J)
                        
                        #=====================================
                        zeta_i =  subdo_pde[iDD].model.forward(subdo_data[iDD].X_bc[idim][id])
                        zeta_j  = subdo_pde[jDD].model.forward(subdo_data[iDD].X_bc[idim][id])
                        
                        phi_ij  = 0.5*(zeta_i[:,0:1]+zeta_j[:,0:1])
                        p_ij    = 0.5*(zeta_i[:,1:]+zeta_j[:,1:])*n
                        
                        # phi_ij  = zeta_j[:,0:1]
                        # p_ij    = zeta_j[:,1:]*n + 2*zeta_j[:,0:1]
                        
                        g_ij = torch.hstack((phi_ij, p_ij))
                    g_bc_idim.append(g_ij)
                subdo_pde[iDD].g_bc.append(g_bc_idim)

    





def train_DDM(gridDDM,subdo_pde,subdo_data,params_train,n_step=2,**kwargs):

    mode = kwargs.get('mode','w+')
    nsubdo = len(subdo_pde)

    
    tin=time.time()

    
        
    for it in range(n_step):
        print(f'it = {it} ')
        for iDD in range(nsubdo):
            train(subdo_pde[iDD],subdo_data[iDD],**params_train[iDD])
        update_boundary(gridDDM=gridDDM,subdo_pde=subdo_pde,subdo_data=subdo_data)
    
    
    tout=time.time()
    print("Elapsed: ",tout-tin," seconds")