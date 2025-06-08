import torch
import numpy as np
from scipy.interpolate import RegularGridInterpolator



class XsContainer:
    def __init__(self, num_materials, num_groups,device=torch.device('cpu')):
        self.M = num_materials
        self.G = num_groups
        
        # Initialize tensors with zeros 
        self.total = torch.zeros((self.M, self.G), dtype=torch.float32,device=device)
        self.uns3d = torch.zeros((self.M, self.G), dtype=torch.float32,device=device)
        self.diffusion = torch.zeros((self.M, self.G), dtype=torch.float32,device=device)
        self.nusigma_f = torch.zeros((self.M, self.G), dtype=torch.float32,device=device)
        self.sigma_r = torch.zeros((self.M, self.G), dtype=torch.float32,device=device)
        self.sigma_s = torch.zeros((self.M, self.G, self.G), dtype=torch.float32,device=device)
        self.fission_spectrum = torch.ones(self.G, dtype=torch.float32,device=device)
        self.device = device


    def setTotal(self, total_list, group_idx, material_idx=None):
        """
        total_list: list or tensor of length M or 1 (single material)
        group_idx: int, the group to set
        material_idx: int or None; if None set for all materials; else set for one material
        """
        values = torch.tensor(total_list, dtype=torch.float32,device=self.device)
        if material_idx is None:
            assert len(values) == self.M, "Length mismatch with number of materials"
            self.total[:, group_idx] = values
        else:
            self.total[material_idx, group_idx] = values.item() if len(values) == 1 else values[group_idx]

    def setDiffusion(self, diffusion_list, group_idx, material_idx=None):
        values = torch.tensor(diffusion_list, dtype=torch.float32,device=self.device)
        if material_idx is None:
            assert len(values) == self.M, "Length mismatch with number of materials"
            self.diffusion[:, group_idx] = values
        else:
            self.diffusion[material_idx, group_idx] = values.item() if len(values) == 1 else values[group_idx]
    
    def setUNS3D(self, uns3d_list, group_idx, material_idx=None):
        values = torch.tensor(uns3d_list, dtype=torch.float32,device=self.device)
        if material_idx is None:
            assert len(values) == self.M, "Length mismatch with number of materials"
            self.uns3d[:, group_idx] = values
        else:
            self.uns3d[material_idx, group_idx] = values.item() if len(values) == 1 else values[group_idx]

    def setNuSigf(self, nuSigf_list, group_idx, material_idx=None):
      
        values = torch.tensor(nuSigf_list, dtype=torch.float32,device=self.device)
        if material_idx is None:
            assert len(values) == self.M, "Length mismatch with number of materials"
            self.nusigma_f[:, group_idx] = values
        else:
            self.nusigma_f[material_idx, group_idx] = values.item() if len(values) == 1 else values[group_idx]

    def setRemoval(self, removal_list, group_idx, material_idx=None):
        values = torch.tensor(removal_list, dtype=torch.float32,device=self.device)
        if material_idx is None:
            assert len(values) == self.M, "Length mismatch with number of materials"
            self.sigma_r[:, group_idx] = values
        else:
            self.sigma_r[material_idx, group_idx] = values.item() if len(values) == 1 else values[group_idx]
       
       

    def setScattering(self, scatter_list, g_to, g_from, material_idx=None):
        """
        scatter_list: list or tensor length M or 1 (single material)
        g_to: group index scattering to
        g_from: group index scattering from
        material_idx: None or int
        """
        values = torch.tensor(scatter_list, dtype=torch.float32,device=self.device)
        if material_idx is None:
            assert len(values) == self.M, "Length mismatch with number of materials"
            self.sigma_s[:, g_to, g_from] = values
        else:
            self.sigma_s[material_idx, g_to, g_from] = values.item() if len(values) == 1 else values[g_to]


    def setFissionSpectrum(self,spectrum_list):
        self.fission_spectrum = torch.tensor(spectrum_list, dtype=torch.float32,device=self.device)




    def get_total(self, material_ids):
        # material_ids: LongTensor of shape (N,), returns total sigma (N, G)
        if self.total is None:
            raise ValueError("Total cross section not provided.")
        return self.total[material_ids]
    
    def get_uns3d(self, material_ids):
        # material_ids: LongTensor of shape (N,), returns total sigma (N, G)
        if self.uns3d is None:
            raise ValueError("uns3d cross section not provided.")
        return self.uns3d[material_ids]
    
    def get_diffusion(self, material_ids):
        # material_ids: LongTensor of shape (N,), returns total sigma (N, G)
        if self.diffusion is None:
            raise ValueError("D cross section not provided.")
        return self.diffusion[material_ids]

    def get_sigma_s(self, material_ids):
        # returns scattering matrix (N, G, G)
        if self.sigma_s is None:
            raise ValueError("Scattering cross section not provided.")
        return self.sigma_s[material_ids]
    
    def get_sigma_r(self, material_ids):
        # returns scattering matrix (N, G, G)
        if self.sigma_r is None:
            raise ValueError("Removal cross section not provided.")
        return self.sigma_r[material_ids]
    
    def get_nusigma_f(self, material_ids):
        # returns fission cross section (N, G)
        if self.nusigma_f is None:
            raise ValueError("nuFission cross section not provided.")
        return self.nusigma_f[material_ids]
    def get_fission_spectrum(self):
        return self.fission_spectrum




class XsEvaluator:
    def __init__(self, xs: XsContainer, geometry):
        self.xs = xs
        self.geometry = geometry

    def total(self, points):
        cell_indices = self.geometry.find_cells(points)  # (N, d)
        idx = tuple(cell_indices[:, i] for i in range(cell_indices.shape[1]))
        material_ids = self.geometry.material_map[idx]
        return self.xs.get_total(material_ids)
    
    def diffusion(self, points):
        cell_indices = self.geometry.find_cells(points)  # (N, d)
        idx = tuple(cell_indices[:, i] for i in range(cell_indices.shape[1]))
        material_ids = self.geometry.material_map[idx]
        return self.xs.get_diffusion(material_ids)
    
    def uns3d(self, points):
        cell_indices = self.geometry.find_cells(points)  # (N, d)
        idx = tuple(cell_indices[:, i] for i in range(cell_indices.shape[1]))
        material_ids = self.geometry.material_map[idx]
        return self.xs.get_uns3d(material_ids)
    
    def sigma_s(self, points):
        cell_indices = self.geometry.find_cells(points)
        idx = tuple(cell_indices[:, i] for i in range(cell_indices.shape[1]))
        material_ids = self.geometry.material_map[idx]
        return self.xs.get_sigma_s(material_ids)
    def sigma_r(self, points):
        cell_indices = self.geometry.find_cells(points)
        idx = tuple(cell_indices[:, i] for i in range(cell_indices.shape[1]))
        material_ids = self.geometry.material_map[idx]
        return self.xs.get_sigma_r(material_ids)
    def nusigma_f(self, points):
        cell_indices = self.geometry.find_cells(points)
        idx = tuple(cell_indices[:, i] for i in range(cell_indices.shape[1]))
        material_ids = self.geometry.material_map[idx]
        return self.xs.get_nusigma_f(material_ids)
    def fission_spectrum(self):
        return self.xs.get_fission_spectrum()


