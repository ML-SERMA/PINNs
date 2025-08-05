
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import plotly.graph_objects as go

class CartesianGeometry:
    def __init__(self, ndim=2, xmin=[0.0, 0.0], xmax=[1.0, 1.0], num_cells=[2, 2],device=torch.device('cpu')):
        self.ndim = ndim

        self.xmin = torch.tensor(xmin, dtype=torch.float32,device=device)
        self.xmax = torch.tensor(xmax, dtype=torch.float32,device=device)
        self.num_cells = torch.tensor(num_cells, dtype=torch.int32,device=device)
        self.device = device
        self.generate_mesh()

    @classmethod
    def from_edges(cls, edges: list, device = torch.device('cpu')):
        obj = cls.__new__(cls)
        obj.ndim = len(edges)
        obj.edges = [torch.tensor(e, dtype=torch.float32,device=device) for e in edges]
        obj.xmin = torch.tensor([e[0] for e in edges], dtype=torch.float32)
        obj.xmax = torch.tensor([e[-1] for e in edges], dtype=torch.float32)
        obj.num_cells = torch.tensor([len(e) - 1 for e in edges], dtype=torch.int32,device=device)
        obj.widths = [obj.edges[d][1:] - obj.edges[d][:-1] for d in range(obj.ndim)]
        obj.centers = [0.5 * (obj.edges[d][:-1] + obj.edges[d][1:]) for d in range(obj.ndim)]
        obj.nCell = torch.prod(obj.num_cells)
        obj.device = device
        return obj
    def generate_mesh(self):
        self.widths = [
            torch.full((self.num_cells[d],), (self.xmax[d] - self.xmin[d]) / self.num_cells[d], device=self.device)
            for d in range(self.ndim)
        ]
        self.edges = []
        for d in range(self.ndim):
            edge = torch.cat(
                [self.xmin[d].unsqueeze(0), self.xmin[d] + torch.cumsum(self.widths[d], dim=0)]
            )
            edge[-1] = self.xmax[d]  
            self.edges.append(edge)
        self.centers = [
            0.5 * (self.edges[d][:-1] + self.edges[d][1:])
            for d in range(self.ndim)
        ]
        self.nCell = torch.prod(self.num_cells)

    def refine_mesh(self, refinement_factors):
        '''
        refinement_factors = [[1, 2, 1, 1],  # for dim 0: subdivide second cell by 2, others remain
                            [2, 2, 1, 1]]  # for dim 1: subdivide first two cells by 2, others remain
        '''
        new_edges, new_widths, new_centers, new_num_cells = [], [], [], []

        for d in range(self.ndim):
            factors = refinement_factors[d]
            if not torch.is_tensor(factors):
                factors = torch.tensor(factors, dtype=torch.int32)
            
            edges = [self.edges[d][0].item()]
            widths = []

            for i, f in enumerate(factors):
                w = self.widths[d][i].item() / f.item()
                edges.extend(edges[-1] + w * torch.arange(1, f.item() + 1).tolist())
                widths.extend([w] * f.item())

            edges_t = torch.tensor(edges, dtype=torch.float32)
            widths_t = torch.tensor(widths, dtype=torch.float32)
            centers_t = 0.5 * (edges_t[:-1] + edges_t[1:])

            new_edges.append(edges_t)
            new_widths.append(widths_t)
            new_centers.append(centers_t)
            new_num_cells.append(len(centers_t))

        self.edges = new_edges
        self.widths = new_widths
        self.centers = new_centers
        self.num_cells = torch.tensor(new_num_cells, dtype=torch.int32)
        self.nCell = torch.prod(self.num_cells).item()

    def modify_mesh(self, edges):
        self.edges = [torch.tensor(ed, dtype=torch.float32,device=self.device) for ed in edges]
        self.ndim = len(edges)
        self.xmin = torch.tensor([e[0].item() for e in self.edges])
        self.xmax = torch.tensor([e[-1].item() for e in self.edges])
        self.num_cells = torch.tensor([len(e) - 1 for e in self.edges],device=self.device)
        self.widths = [self.edges[d][1:] - self.edges[d][:-1] for d in range(self.ndim)]
        self.centers = [0.5 * (self.edges[d][:-1] + self.edges[d][1:]) for d in range(self.ndim)]
        self.nCell = torch.prod(self.num_cells)


    def find_cells(self,points: torch.Tensor) -> torch.Tensor:
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
        assert d == len(self.edges), "Number of edges must match point dimensions"

        indices = []
        for dim in range(d):
            coord = points[:, dim]
            edge = self.edges[dim]
            idx = torch.searchsorted(edge, coord.contiguous(), right=False) - 1
            idx = idx.clamp(0, len(edge) - 2)
            indices.append(idx)

        return torch.stack(indices, dim=1)  # shape [N, d]

    
    def setRegionsIndex(self, regionsIndex):
        """
        Set the regions (materials) index array for each cell in the mesh.

        Parameters:
        -----------
        regionsIndex : torch.Tensor or list or numpy array
            An array of shape matching the cell grid (e.g., [Nx, Ny] or [Nx, Ny, Nz])
            that contains the material/region index assigned to each cell.
        """
        if not torch.is_tensor(regionsIndex):
            regionsIndex = torch.tensor(regionsIndex, dtype=torch.int32)
        self.regionsIndex = regionsIndex

    def setRegionsName(self,regionsName):
        self.regionsName = regionsName

    def construct_material_map(self):
        """
        Use self.regionsIndex (list of regions)
        and self.num_cells (tensor of number of cells per dimension)
        to build self.material_map, a tensor of material indices per cell.
        """
        shape = tuple(self.num_cells.tolist())  # convert tensor to tuple if needed
        material_map = torch.zeros(shape, dtype=torch.long, device=self.device)

        for region in self.regionsIndex:
            M = region[0]
            indices = region[1:]

            # Build slices for each dimension
            # Assuming indices are [start_dim1, end_dim1, start_dim2, end_dim2, ...]
            slices = tuple(slice(indices[2*i], indices[2*i+1]) for i in range(len(shape)))

            material_map[slices] = M

        self.material_map = material_map
        return material_map


    def plot_mesh(self):
        if self.ndim == 2:
            xs = self.edges[0]
            ys = self.edges[1]
            xv, yv = torch.meshgrid(xs, ys, indexing='ij')
            plt.plot(xv.flatten(), yv.flatten(), 'k.')
            plt.show()

    def view_grid(self):
        if self.ndim == 2:
            xs = self.edges[0].cpu()
            ys = self.edges[1].cpu()
            ax = plt.gca()
            for i in range(len(xs) - 1):
                for j in range(len(ys) - 1):
                    w = xs[i+1] - xs[i]
                    h = ys[j+1] - ys[j]
                    ax.add_patch(Rectangle((xs[i], ys[j]), w, h, fill=True, color='#008610', alpha=0.2))
            for x in xs:
                plt.plot([x, x], [ys[0], ys[-1]], color='black', alpha=0.3)
            for y in ys:
                plt.plot([xs[0], xs[-1]], [y, y], color='black', alpha=0.3)
            ax.set_aspect('equal')
            plt.show()

    def plot_materials(self, materials, material_names=None, colors=None):
        if self.ndim != 2:
            raise NotImplementedError("Only 2D visualization is supported for now.")

        xs = self.edges[0]
        ys = self.edges[1]
        Nx = len(xs) - 1
        Ny = len(ys) - 1

        unique_materials = np.unique(materials)
        if material_names is None:
            material_names = [f"Mat {i}" for i in unique_materials]
        if colors is None:
            colors = plt.cm.tab20(np.linspace(0, 1, len(unique_materials)))

        material_to_name = {i: name for i, name in zip(unique_materials, material_names)}
        material_to_color = {i: colors[k] for k, i in enumerate(unique_materials)}

        fig, ax = plt.subplots(figsize=(6, 6))
        for i in range(Nx):
            for j in range(Ny):
                x0 = xs[i]
                y0 = ys[j]
                width = xs[i+1] - xs[i]
                height = ys[j+1] - ys[j]

                mat_idx = materials[i, j]
                rect = Rectangle((x0, y0), width, height,
                                facecolor=material_to_color[mat_idx],
                                edgecolor='black', linewidth=0.5)
                ax.add_patch(rect)
                ax.text(x0 + width/2, y0 + height/2, material_to_name[mat_idx],
                        ha='center', va='center', fontsize=8, color='white')

        ax.set_xlim(xs[0], xs[-1])
        ax.set_ylim(ys[0], ys[-1])
        ax.set_aspect('equal')
        ax.set_title("Material Map")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.tight_layout()
        plt.show()
    


    def plot_material_map(self, material_map):
        # 1D coordinate arrays
        x = self.centers[0].cpu().numpy()
        y = self.centers[1].cpu().numpy()
        z = self.centers[2].cpu().numpy()

        # 3D meshgrid
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        M = material_map.cpu().numpy().astype(int)  # ensure it's integer for materials

        # Get discrete tab20 colormap from matplotlib
        cmap = plt.get_cmap('tab20', np.max(M) + 1)  # N distinct material colors
        colorscale = []

        for i in range(cmap.N):
            rgb = cmap(i)[:3]  # ignore alpha
            colorscale.append([i / (cmap.N - 1), f'rgb({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)})'])

        # Plotly volume plot
        fig = go.Figure(data=go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=M.flatten(),
            opacity=0.3,
            surface_count=np.unique(M).size,
            colorscale=colorscale,
            colorbar=dict(title='Material ID'),
            showscale=True
        ))

        fig.update_layout(
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
            ),
            title='3D Material Map'
        )

        fig.show()
    
        
    def plot_slice(self,map,z_index=None):
        # Choose a slice index in Z
        if z_index is None:
            z_index = map.shape[2] // 2  # middle slice
        mymap = map.cpu()
        slice_2d = mymap[:, :, z_index]  # shape (Nx, Ny)
        # Transpose to shape (Ny, Nx) for (Y, X) display
        # You may also flip the Y-axis if needed
        slice_2d = slice_2d.T  # now shape (Ny, Nx)
        plt.imshow(mymap[:, :, z_index], origin='upper', cmap='tab20')
        plt.title(f"Material Map Slice at z={z_index}")
        plt.colorbar(label="Material ID")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()
        
    def plot_slice_xy(material_map, z_index=None, cmap='tab20'):
        # Convert to numpy
        map_np = material_map.cpu().numpy()

        # Choose default slice
        if z_index is None:
            z_index = map_np.shape[2] // 2

        # Slice: get XY at fixed Z
        slice_xy = map_np[:, :, z_index].T  # transpose to (Ny, Nx)

        # Plot
        plt.figure(figsize=(6, 5))
        im = plt.imshow(slice_xy, origin='lower', cmap=cmap, interpolation='nearest')
        plt.title(f"Material Map Slice at z = {z_index}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.colorbar(im, label="Material ID")
        plt.tight_layout()
        plt.show()
        
    def plot_3d_voxel_centers_with_indices(mesh):
        x_centers = mesh.centers[0]
        y_centers = mesh.centers[1]
        z_centers = mesh.centers[2]

        # Create meshgrid of center coordinates
        X, Y, Z = np.meshgrid(x_centers, y_centers, z_centers, indexing='ij')

        # Flatten for plotting
        Xf = X.flatten()
        Yf = Y.flatten()
        Zf = Z.flatten()

        # Index for each voxel
        indices = np.arange(len(Xf))

        # Plot with text labels
        fig = go.Figure(data=go.Scatter3d(
            x=Xf, y=Yf, z=Zf,
            mode='text',
            text=[str(i) for i in indices],
            textfont=dict(size=10, color='black'),
            showlegend=False
        ))

        fig.update_layout(
            title="3D Mesh Cell Centers with Indices",
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data'
            ),
            margin=dict(l=0, r=0, b=0, t=30)
        )
        

    def plot_material_map_with_indices(self,material_map):
        # Extract centers
        x_centers = self.centers[0].cpu().numpy()
        y_centers = self.centers[1].cpu().numpy()
        z_centers = self.centers[2].cpu().numpy()

        # Ensure material_map is on CPU and numpy
        mat = material_map.cpu().numpy()

        # Create meshgrid of voxel centers
        X, Y, Z = np.meshgrid(x_centers, y_centers, z_centers, indexing='ij')
        Xf = X.flatten()
        Yf = Y.flatten()
        Zf = Z.flatten()
        mat_flat = mat.flatten()

        # Optional: pick a color per material ID
        unique_mats = np.unique(mat_flat)
        color_map = {m: i for i, m in enumerate(unique_mats)}  # Map material ID to integer color
        color_vals = [color_map[m] for m in mat_flat]

        fig = go.Figure()

        # Plot colored markers for each voxel
        fig.add_trace(go.Scatter3d(
            x=Xf, y=Yf, z=Zf,
            mode='markers+text',
            marker=dict(
                size=5,
                color=color_vals,
                colorscale='Viridis',
                colorbar=dict(title='Material ID'),
                showscale=True,
                opacity=0.8
            ),
            text=[str(m) for m in mat_flat],
            textposition="top center",
            textfont=dict(size=10, color='black'),
            name='Material ID'
        ))

        fig.update_layout(
            title='3D Material Map with Material ID at Voxel Centers',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data'
            ),
            margin=dict(l=0, r=0, b=0, t=30)
        )

        fig.show()
        


    def plot_material_map_with_indices_and_grid(self, material_map):
        # Convert to numpy
        x_centers = self.centers[0].cpu().numpy()
        y_centers = self.centers[1].cpu().numpy()
        z_centers = self.centers[2].cpu().numpy()

        x_edges = self.edges[0].cpu().numpy()
        y_edges = self.edges[1].cpu().numpy()
        z_edges = self.edges[2].cpu().numpy()

        mat = material_map.cpu().numpy()
        Nx, Ny, Nz = mat.shape

        # Meshgrid of centers
        Xc, Yc, Zc = np.meshgrid(x_centers, y_centers, z_centers, indexing='ij')
        Xf = Xc.flatten()
        Yf = Yc.flatten()
        Zf = Zc.flatten()
        mat_flat = mat.flatten()

        # Color mapping
        unique_mats = np.unique(mat_flat)
        color_map = {m: i for i, m in enumerate(unique_mats)}
        color_vals = [color_map[m] for m in mat_flat]

        fig = go.Figure()

        # Add colored center markers with text
        fig.add_trace(go.Scatter3d(
            x=Xf, y=Yf, z=Zf,
            mode='markers+text',
            marker=dict(
                size=5,
                color=color_vals,
                colorscale='Viridis',
                opacity=0.8,
                colorbar=dict(title='Material ID'),
            ),
            text=[str(m) for m in mat_flat],
            textposition="top center",
            textfont=dict(size=10, color='black'),
            name='Material ID'
        ))

        # Draw voxel edges (wireframe cubes)
        def draw_voxel_box(x0, x1, y0, y1, z0, z1):
            # 12 edges of the cube
            lines = [
                [(x0, y0, z0), (x1, y0, z0)],
                [(x1, y0, z0), (x1, y1, z0)],
                [(x1, y1, z0), (x0, y1, z0)],
                [(x0, y1, z0), (x0, y0, z0)],

                [(x0, y0, z1), (x1, y0, z1)],
                [(x1, y0, z1), (x1, y1, z1)],
                [(x1, y1, z1), (x0, y1, z1)],
                [(x0, y1, z1), (x0, y0, z1)],

                [(x0, y0, z0), (x0, y0, z1)],
                [(x1, y0, z0), (x1, y0, z1)],
                [(x1, y1, z0), (x1, y1, z1)],
                [(x0, y1, z0), (x0, y1, z1)],
            ]
            return lines

        # Optional: limit max number of cubes to avoid clutter
        max_boxes = 500  # change as needed
        count = 0

        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    if count >= max_boxes:
                        continue
                    x0, x1 = x_edges[i], x_edges[i+1]
                    y0, y1 = y_edges[j], y_edges[j+1]
                    z0, z1 = z_edges[k], z_edges[k+1]
                    edges = draw_voxel_box(x0, x1, y0, y1, z0, z1)
                    for pt1, pt2 in edges:
                        fig.add_trace(go.Scatter3d(
                            x=[pt1[0], pt2[0]],
                            y=[pt1[1], pt2[1]],
                            z=[pt1[2], pt2[2]],
                            mode='lines',
                            line=dict(color='black', width=1),
                            showlegend=False
                        ))
                    count += 1

        fig.update_layout(
            title='3D Material Map with Indices and Grid',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data',
            ),
            margin=dict(l=0, r=0, b=0, t=30)
        )

        fig.show()
        
    def plot_material_voxels(self, material_map):
        # Get edges
        x_edges = self.edges[0].cpu().numpy()
        y_edges = self.edges[1].cpu().numpy()
        z_edges = self.edges[2].cpu().numpy()

        # Convert material map
        mat = material_map.cpu().numpy().astype(int)
        Nx, Ny, Nz = mat.shape

        # Define a discrete color map
        material_ids = np.unique(mat)
        num_materials = len(material_ids)
        colormap = plt.get_cmap('tab20', num_materials)  # Discrete colormap
        norm = mcolors.Normalize(vmin=material_ids.min(), vmax=material_ids.max())
        colors = [mcolors.to_hex(colormap(norm(mid))) for mid in material_ids]
        mat_to_color = {mid: colors[i] for i, mid in enumerate(material_ids)}

        fig = go.Figure()

        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    m_id = mat[i, j, k]
                    x0, x1 = x_edges[i], x_edges[i+1]
                    y0, y1 = y_edges[j], y_edges[j+1]
                    z0, z1 = z_edges[k], z_edges[k+1]

                    fig.add_trace(go.Mesh3d(
                        x=[x0,x1,x1,x0,x0,x1,x1,x0],
                        y=[y0,y0,y1,y1,y0,y0,y1,y1],
                        z=[z0,z0,z0,z0,z1,z1,z1,z1],
                        i=[0,1,2,0,4,5,6,4,0,1,5,4],
                        j=[1,2,3,3,5,6,7,7,4,5,6,0],
                        k=[2,3,0,1,6,7,4,5,5,6,7,0],
                        opacity=0.8,
                        color=mat_to_color[m_id],
                        showscale=False,
                        flatshading=True,
                        name=f"Material {m_id}"
                    ))

        fig.update_layout(
            scene=dict(
                xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
                aspectmode='data'
            ),
            title='3D Material Map (Voxel Grid)',
            showlegend=False
        )
        fig.show()


