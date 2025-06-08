import numpy as np
import time
import torch
def get_time(pde,data,n_step=100,k=2):
    listTimePDE = []
    listTimeBC  = []
    for it in range(n_step+k):
        # pde.optimizer.zero_grad()

        torch.cuda.synchronize()
        tin_f=time.time()
        loss_f   = pde.loss_PDE(data.X_train)
        torch.cuda.synchronize()
        tout_f=time.time()
        dt_f = tout_f-tin_f
        listTimePDE.append(dt_f)

        torch.cuda.synchronize()
        tin_bc=time.time()
        loss_bc  = pde.loss_BC(data.X_bc)
        torch.cuda.synchronize()
        tout_bc=time.time()
        dt_bc = tout_bc-tin_bc
        listTimeBC.append(dt_bc)

        # loss =  loss_f + loss_bc
        # loss.backward(retain_graph=True)
        # pde.optimizer.step()
        # pde.scheduler.step()
    meanTimePDE = np.mean(listTimePDE[k:])
    # print(f"==>> TimePDE: {listTimePDE}")
    meanTimeBC = np.mean(listTimeBC[k:])
    # print(f"==>> TimeBC: {listTimeBC}")
    return meanTimePDE, meanTimeBC

def get_time_gpu(pde,data,n_step=100,k=2):
    listTimePDE = []
    listTimeBC  = []
    for it in range(n_step+k):
        # pde.optimizer.zero_grad()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        loss_f   = pde.loss_PDE(data.X_train)
        end_event.record()
        torch.cuda.synchronize()
        dt_f = start_event.elapsed_time(end_event)
        listTimePDE.append(dt_f)
    meanTimePDE = np.mean(listTimePDE[k:])

    for it in range(n_step+k):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        loss_bc  = pde.loss_BC(data.X_bc)
        torch.cuda.synchronize()
        end_event.record()
        torch.cuda.synchronize()
        dt_bc = start_event.elapsed_time(end_event)
        listTimeBC.append(dt_bc)
    
    meanTimeBC = np.mean(listTimeBC[k:])

    return meanTimePDE, meanTimeBC

def find_cells(points: torch.Tensor, edges: list[torch.Tensor]) -> torch.Tensor:
    """
    Find which cell each point belongs to on a non-uniform Cartesian grid.

    Args:
        points: Tensor of shape [N, d], where d = len(edges)
        edges: List of 1D tensors of length [M0+1, M1+1, ..., Md+1],
            each representing cell boundaries along a dimension.

    Returns:
        Tensor of shape [N, d] — indices of the cell containing each point.
    """
    N, d = points.shape
    assert d == len(edges), "Number of edges must match point dimensions"

    indices = []
    for dim in range(d):
        coord = points[:, dim]
        edge = edges[dim]
        idx = torch.searchsorted(edge, coord, right=False) - 1
        idx = idx.clamp(0, len(edge) - 2)
        indices.append(idx)

    return torch.stack(indices, dim=1)  # shape [N, d]


def get_cross_sections(points: torch.Tensor,
                        edges: list[torch.Tensor],
                        materials: torch.Tensor,
                        sigma: torch.Tensor) -> torch.Tensor:
    """
    Return total cross sections σ_t at each point for all groups.

    Args:
        points: Tensor of shape [N, d]
        edges: List of d 1D tensors (non-uniform grid boundaries)
        materials: Tensor of shape [Nx, Ny] or [Nx, Ny, Nz] — material index per cell
        sigma_t: Tensor of shape [M, G] — cross section per material and group
        or sigma_s: Tensor of shape [M, G, G] — scattering cross sections per material

    Returns:
        Tensor of shape [N, G] — total cross section at each point
        Tensor of shape [N, G, G] — scattering cross sections at each point
    """
    # Step 1: locate cell for each point
    cell_indices = find_cells(points, edges)  # shape [N, d]

    # Step 2: lookup material index
    material_ids = materials[tuple(cell_indices.T)]  # shape [N]

    # Step 3: lookup σ_t for each material
    return sigma[material_ids]  # shape [N, G]
