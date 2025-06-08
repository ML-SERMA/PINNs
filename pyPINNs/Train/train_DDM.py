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


def update_boundary_old(gridDDM,subdo_pde,subdo_data):
    with torch.no_grad():
        ndim = len(gridDDM)
        nDD = len(subdo_pde)
        for iDD in range(nDD):
            index_I = getDomainIndex(ideg=iDD,sdeg=gridDDM)
            print(f"==>> index_I: {index_I}")
            subdo_pde[iDD].g_bc = []
            for idim in range(ndim):
                g_bc_idim = []
                for id in range(2): # left or right
                    g_ij = None
                    index_J = index_I.copy()
                    if subdo_data[iDD].domain.boundary_conditions[idim][id] == 'INTERFACE':
                        n = (-1)**(id+1) # square domain
                        index_J[idim] = index_J[idim] + n 
                        print(f"==>> index_J: {index_J}")
                        # Find neighbor domain
                        jDD = fromIndexToDomain(gridDDM,index_J)
                        print(f"==>> jDD: {jDD}")
                        print(f"==>> index_I again: {index_I}")
                        #=====================================
                        zeta_i =  subdo_pde[iDD].model.forward(subdo_data[iDD].X_bc[idim][id])
                        zeta_j  = subdo_pde[jDD].model.forward(subdo_data[iDD].X_bc[idim][id])
                        
                        # phi_ij  = 0.5*(zeta_i[:,0:1]+zeta_j[:,0:1])
                        # p_ij    = 0.5*(zeta_i[:,1:]+zeta_j[:,1:])*n
                        
                        # phi_ij  = zeta_j[:,0:1]
                        # p_ij    = zeta_j[:,1:]*n + 2*zeta_j[:,0:1]
                        alpha = 0.5
                        beta  = 0.5
                        phi_ij  = alpha*zeta_i[:,0:1]   + (1-alpha)*zeta_j[:,0:1]
                        p_ij    = beta*(zeta_i[:,1:])*n + (1-beta)*(zeta_j[:,1:])*n
                        
                        g_ij = torch.hstack((phi_ij, p_ij))
                    g_bc_idim.append(g_ij)
                subdo_pde[iDD].g_bc.append(g_bc_idim)
            input('Enter')

    
def update_boundary(gridDDM, subdo_pde, subdo_data):
    """
    Updates boundary conditions for each subdomain in Domain Decomposition Method (DDM).

    Args:
        gridDDM: List of grid configurations for the domain decomposition.
        subdo_pde: List of PDE solvers (models) for each subdomain.
        subdo_data: List of data for each subdomain, including boundary conditions.

    The function checks for interface conditions between neighboring subdomains and updates
    the boundary conditions using the interface values.
    """
    with torch.no_grad():
        # Number of spatial dimensions (ndim) and subdomains (nDD)
        ndim = len(gridDDM)
        nDD = len(subdo_pde)

        # Loop over all subdomains
        for iDD in range(nDD):
            # Get the index of the current subdomain (index_I)
            index_I = getDomainIndex(ideg=iDD, sdeg=gridDDM)

            # Initialize an empty list to store boundary conditions for the current subdomain
            subdo_pde[iDD].g_bc = []

            # Loop over each spatial dimension (idim)
            for idim in range(ndim):
                g_bc_idim = []

                # Loop over the two boundary sides (left and right)
                for id in range(2):
                    boundary_condition = subdo_data[iDD].domain.boundary_conditions[idim][id]

                    if boundary_condition == 'INTERFACE':
                        # Interface boundary condition: Find the neighbor subdomain
                        n = (-1)**(id+1)  # Directional factor based on left or right boundary

                        # Adjust the index to locate the neighboring subdomain
                        index_J = index_I.copy()  # Create a copy of index_I
                        index_J[idim] = index_J[idim] + n

                        # Find the neighboring subdomain using the index
                        jDD = fromIndexToDomain(gridDDM, index_J)

                        # Compute the solution for the current and neighboring subdomains at the boundary
                        zeta_i = subdo_pde[iDD].model.forward(subdo_data[iDD].X_bc[idim][id])
                        zeta_j = subdo_pde[jDD].model.forward(subdo_data[iDD].X_bc[idim][id])

                        # Interpolate phi and current (p) values at the interface
                        alpha = 0.5  # Weight for zeta_i
                        beta = 0.5   # Weight for zeta_j

                        # Compute the interface values for phi and current
                        phi_ij = alpha * zeta_i[:, 0:1] + (1 - alpha) * zeta_j[:, 0:1]
                        p_ij = beta * (zeta_i[:, 1:])*n + (1 - beta) * (zeta_j[:, 1:])*n

                        # Combine phi_ij and p_ij into a single tensor for boundary condition
                        g_ij = torch.hstack((phi_ij, p_ij))

                    else:
                        # For non-interface conditions
                        g_ij = None  # Mark for later selective boundary loss

                    # Append the boundary condition for this spatial dimension and boundary side
                    g_bc_idim.append(g_ij)
                # After processing both sides, append the boundary conditions for the current dimension
                subdo_pde[iDD].g_bc.append(g_bc_idim)




# def train_DDM(gridDDM,subdo_pde,subdo_data,params_train,n_step=2,**kwargs):

#     mode = kwargs.get('mode','w+')
#     nsubdo = len(subdo_pde)
#     tin=time.time()
#     for it in range(n_step):
#         print(f'it = {it} ')
#         for iDD in range(nsubdo):
#             train(subdo_pde[iDD],subdo_data[iDD],**params_train[iDD])
#         update_boundary(gridDDM=gridDDM,subdo_pde=subdo_pde,subdo_data=subdo_data)
#     tout=time.time()
#     print("Elapsed: ",tout-tin," seconds")


def train_DDM(gridDDM, subdo_pde, subdo_data, params_train, n_step=2, **kwargs):
    """
    Train DDM PINNs with specified subdomains, updating boundary conditions between each step.

    Parameters:
    - gridDDM: grid defining the DDM setup.
    - subdo_pde: list of PDE solvers for each subdomain.
    - subdo_data: list of data objects for each subdomain.
    - params_train: training parameters for each subdomain.
    - n_step: number of training steps.
    - kwargs: additional keyword arguments, e.g., mode ('w+', default).

    Returns:
    - None (training is done in place)
    """
    mode = kwargs.get('mode', 'w+')  # Default mode is 'w+' for file handling
    nsubdo = len(subdo_pde)  # Number of subdomains
    # shared_interfaces = build_shared_interface_points(gridDDM, subdo_data)
    # Track the time of training
    # tin = time.time()

    # Main training loop
    for it in range(n_step):
        # print(f'Iteration {it + 1}/{n_step}')
        
        # Train each subdomain
        for iDD in range(nsubdo):
            # print(f"Training subdomain {iDD + 1}/{nsubdo}")
            train(subdo_pde[iDD], subdo_data[iDD], **params_train[iDD])

        # Update the boundary conditions after each training step
        update_boundary(gridDDM=gridDDM, subdo_pde=subdo_pde, subdo_data=subdo_data)

    # Time the total training duration
    # tout = time.time()
    # print("Training completed.")
    # print(f"Elapsed time: {tout - tin:.2f} seconds")



def build_shared_interface_points(gridDDM, subdo_data):
    ndim = len(gridDDM)
    nDD = len(subdo_data)
    shared_interfaces = {}  # {(iDD, jDD, idim, id): tensor}
    
    for iDD in range(nDD):
        index_i = getDomainIndex(iDD, gridDDM)
        for idim in range(ndim):
            for id in range(2):
                if subdo_data[iDD].domain.boundary_conditions[idim][id] == 'INTERFACE':
                    n = (-1)**(id+1)
                    index_j = index_i.copy()
                    index_j[idim] += n
                    jDD = fromIndexToDomain(gridDDM, index_j)
                    
                    key = tuple(sorted((iDD, jDD)) + [idim, id])
                    if key not in shared_interfaces:
                        # Choose one side's points, or interpolate between
                        X_shared = subdo_data[iDD].X_bc[idim][id]
                        shared_interfaces[key] = X_shared
    return shared_interfaces

def update_boundary_symmetric(gridDDM, subdo_pde, subdo_data, shared_interfaces):
    ndim = len(gridDDM)
    nDD = len(subdo_pde)

    with torch.no_grad():
        for iDD in range(nDD):
            index_I = getDomainIndex(iDD, gridDDM)
            subdo_pde[iDD].g_bc = []

            for idim in range(ndim):
                g_bc_idim = []
                for id in range(2):
                    if subdo_data[iDD].domain.boundary_conditions[idim][id] == 'INTERFACE':
                        n = (-1)**(id+1)
                        index_J = index_I.copy()
                        index_J[idim] += n
                        jDD = fromIndexToDomain(gridDDM, index_J)

                        key = tuple(sorted((iDD, jDD)) + [idim, id])
                        X_shared = shared_interfaces[key].to(subdo_data[iDD].X_bc[idim][id].device)

                        u_i = subdo_pde[iDD].model(X_shared)
                        u_j = subdo_pde[jDD].model(X_shared)

                        alpha, beta = 0.5, 0.5
                        phi_ij = alpha * u_i[:, 0:1] + (1 - alpha) * u_j[:, 0:1]
                        p_ij   = beta * u_i[:, 1:]*n   + (1 - beta) * u_j[:, 1:]*n

                        g_bc = torch.hstack((phi_ij, p_ij))
                    else:
                        g_bc = None
                    g_bc_idim.append(g_bc)
                subdo_pde[iDD].g_bc.append(g_bc_idim)

def interface_residual(iDD, jDD, X_shared, subdo_pde):
    with torch.no_grad():
        u_i = subdo_pde[iDD].model(X_shared)
        u_j = subdo_pde[jDD].model(X_shared)
        return torch.norm(u_i - u_j, dim=1).mean()
    


def update_boundary_multigroup(gridDDM, subdo_pde, subdo_data, group_order="group_mixed"):
    with torch.no_grad():
        ndim = len(gridDDM)
        nDD = len(subdo_pde)
        ngroup = subdo_pde[0].ngroup  # assume all subdomains have same number of groups
        d = subdo_data[0].X_bc[0][0].shape[1]  # spatial dimension

        for iDD in range(nDD):
            index_I = getDomainIndex(ideg=iDD, sdeg=gridDDM)
            subdo_pde[iDD].g_bc = []

            for idim in range(ndim):
                g_bc_idim = []
                for id in range(2):  # left or right
                    bc_type = subdo_data[iDD].domain.boundary_conditions[idim][id]

                    if bc_type == 'INTERFACE':
                        index_J = index_I.copy()
                        n = (-1) ** (id + 1)
                        index_J[idim] += n
                        jDD = fromIndexToDomain(gridDDM, index_J)

                        zeta_i = subdo_pde[iDD].model(subdo_data[iDD].X_bc[idim][id])
                        zeta_j = subdo_pde[jDD].model(subdo_data[iDD].X_bc[idim][id])

                        g_list = []
                        for g in range(ngroup):
                            if group_order == "group_mixed":
                                phi_i = zeta_i[:, g * (1 + d):g * (1 + d) + 1]
                                J_i = zeta_i[:, g * (1 + d) + 1: g * (1 + d) + 1 + d]
                                phi_j = zeta_j[:, g * (1 + d):g * (1 + d) + 1]
                                J_j = zeta_j[:, g * (1 + d) + 1: g * (1 + d) + 1 + d]
                            elif group_order == "group_first":
                                phi_i = zeta_i[:, g:g + 1]
                                phi_j = zeta_j[:, g:g + 1]
                                J_i = zeta_i[:, ngroup * d + g * d: ngroup * d + (g + 1) * d]
                                J_j = zeta_j[:, ngroup * d + g * d: ngroup * d + (g + 1) * d]
                            else:
                                raise ValueError("Invalid group_order.")

                            # Interface averaging (can be adjusted)
                            alpha, beta = 0.0, 0.0
                            phi_ij = alpha * phi_i + (1 - alpha) * phi_j
                            J_ij = beta * J_i + (1 - beta) * J_j
                            J_ij_n = J_ij[:, idim:idim + 1] * n
                            g_list.append(torch.hstack((phi_ij, J_ij_n)))

                        g_ij = torch.cat(g_list, dim=1)  # [N, ngroup*2]
                    else:
                        g_ij = None  # use None so loss is computed only when needed

                    g_bc_idim.append(g_ij)
                subdo_pde[iDD].g_bc.append(g_bc_idim)

