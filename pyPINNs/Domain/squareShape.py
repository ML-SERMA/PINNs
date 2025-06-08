import torch
import numpy as np
# import sobol_seq
from scipy.stats import qmc
from ..Mesh.CartesianMesh import CartesianMesh

# def RandomSampling(Ncollocation, Nbounds, bounds):
#     ## Collocation points, inside domain for training
#     x = np.random.uniform(bounds[0], bounds[1], Ncollocation)
#     y = np.random.uniform(bounds[2], bounds[3], Ncollocation)
#     X_f_train = np.hstack((x.flatten()[:,None], y.flatten()[:,None]))
    
#     ## Boundaries
#     ### boundaries up and down
#     x_bc_u = np.random.uniform(bounds[0], bounds[1], Nbounds)
#     y_bc_u = np.full(Nbounds, bounds[3])
#     x_bc_d = np.random.uniform(bounds[0], bounds[1], Nbounds)
#     y_bc_d = np.full(Nbounds, bounds[2])
#     ### boundaries left and right
#     y_bc_l = np.random.uniform(bounds[2], bounds[3], Nbounds)
#     x_bc_l = np.full(Nbounds, bounds[0])
#     y_bc_r = np.random.uniform(bounds[2], bounds[3], Nbounds)
#     x_bc_r = np.full(Nbounds, bounds[1])

#     X_star = np.hstack((np.vstack([x_bc_u,x_bc_d,x_bc_l,x_bc_r]).flatten()[:,None], np.vstack([y_bc_u,y_bc_d,y_bc_l,y_bc_r]).flatten()[:,None]))
#     return X_f_train, X_star

def generator_points(n_samples,n_dim,random_seed=None,type_of_points='random'):
    if type_of_points == 'random':
        if random_seed is not None:
            torch.random.manual_seed(random_seed)
            return torch.rand([n_samples, n_dim]).type(torch.FloatTensor)
        else:
            return torch.rand([n_samples, n_dim]).type(torch.FloatTensor)
        
    # elif type_of_points == "sobol":
    #     skip = random_seed
    #     data = np.full((n_samples, n_dim), np.nan)
    #     for j in range(n_samples):
    #         seed = j + skip
    #         data[j, :], next_seed = sobol_seq.i4_sobol(n_dim, seed)
    #     return torch.from_numpy(data).type(torch.FloatTensor)
    
    elif type_of_points == "sobol_torch":
        soboleng = torch.quasirandom.SobolEngine(dimension=n_dim,scramble=False, seed=random_seed)
        return soboleng.draw(n_samples)
    
    elif type_of_points == "LHS":
        sampler = qmc.LatinHypercube(d=n_dim,scramble=True, seed=random_seed)
        tmp = sampler.fast_forward(2)
        sample = sampler.random(n=n_samples)
        return torch.from_numpy(sample).type(torch.FloatTensor)
    elif type_of_points == "Halton":
        sampler = qmc.Halton(d=n_dim,scramble=True, seed=random_seed)
        tmp = sampler.fast_forward(2)
        sample = sampler.random(n=n_samples)
        return torch.from_numpy(sample).type(torch.FloatTensor)
    elif type_of_points == "Sobol":
        sampler = qmc.Sobol(d=n_dim,scramble=True, seed=random_seed)
        tmp = sampler.fast_forward(2)
        sample = sampler.random(n=n_samples)
        return torch.from_numpy(sample).type(torch.FloatTensor)
    else:
        print('This method is not implemented !')

class SquareDomain:
    def __init__(self,xmin,xmax,boundary_conditions = [['ZERO_FLUX', 'ZERO_FLUX'],['ZERO_FLUX', 'ZERO_FLUX']],type_of_points='random'):
        
        self.min_values = torch.tensor(xmin)
        self.max_values = torch.tensor(xmax)
        self.input_dim = self.min_values.shape[0]
        self.boundary_conditions = boundary_conditions
        self.type_of_points = type_of_points

    def add_collocation_points(self,n_collocation,random_seed=None):
        x_collocation = generator_points(n_collocation,self.input_dim,random_seed,self.type_of_points)
        x_collocation = x_collocation*(self.max_values-self.min_values) + self.min_values
        return x_collocation
        
    # def add_boundary_points(self,n_boundary,random_seed=None):
    #     x_b = []
    #     for idim in range(self.input_dim):
    #         for b_value in [self.min_values[idim], self.max_values[idim]]:
    #             x_points = generator_points(n_boundary,self.input_dim,random_seed,self.type_of_points)
    #             x_points = x_points*(self.max_values-self.min_values) + self.min_values
    #             x_points[:,idim] = b_value
    #             x_b.append(x_points)
    #     x_boundary = torch.vstack(x_b)
    #     return x_boundary

    def add_boundary_points(self,n_boundary,random_seed=None):
        x_boundary = []
        x_boundary_all = []
        for idim in range(self.input_dim):
            x_b = []
            for b_value in [self.min_values[idim], self.max_values[idim]]:
                x_points = generator_points(n_boundary,self.input_dim,random_seed,self.type_of_points)
                x_points = x_points*(self.max_values-self.min_values) + self.min_values
                x_points[:,idim] = b_value
                x_b.append(x_points)
                x_boundary_all.append(x_points)
            x_boundary.append(x_b)
        x_boundary_all = torch.vstack(x_boundary_all)
        return x_boundary
    


    

    




    def add_experiments_points(self,n_exp,dim,position,random_seed=None):

        x_points = generator_points(n_exp,self.input_dim,random_seed,self.type_of_points)
        x_points = x_points*(self.max_values-self.min_values) + self.min_values
        x_points[:,dim] = position
        return x_points
        
    def add_cross_points(self,n_cross,random_seed=None):
        x_cross = []
        xmid = 0.5*(self.max_values-self.min_values) + self.min_values
        for idim in range(self.input_dim):
            x_exp = self.add_experiments_points(n_exp=n_cross,dim=idim,position=xmid[idim],random_seed=random_seed)
            x_cross.append(x_exp)
        x_cross = torch.vstack(x_cross)
        return x_cross
    

    def add_interface_points(self,n_face,test='Dauge',random_seed=None):
        x_interface = []
        if test =='Dauge':
            xmid = 0.5*(self.max_values-self.min_values) + self.min_values
            for idim in range(self.input_dim):
                x_face = self.add_experiments_points(n_exp=n_face,dim=idim,position=xmid[idim],random_seed=random_seed)
                x_interface.append(x_face)

        elif test =='Center':
            b_min_values = 1.0/3*(self.max_values-self.min_values) + self.min_values
            b_max_values = 2.0/3*(self.max_values-self.min_values) + self.min_values
            for idim in range(self.input_dim):
                x_b = []
                for b_value in [b_min_values[idim], b_max_values[idim]]:
                    x_points = generator_points(n_face,self.input_dim,random_seed,self.type_of_points)
                    x_points = x_points*(b_max_values-b_min_values) + b_min_values
                    x_points[:,idim] = b_value
                    x_b.append(x_points)
                x_interface.append(torch.vstack(x_b))
                
        return x_interface

            





    def add_test_points(self,n_test):
        
        # params_mesh = params_mesh = {'ndim':2, 'xmin':[0, 0], 'xmax':[100, 100], 'nmail':[n_test, n_test]}
        if not isinstance(n_test, list):
            [n_test for idim in range(self.input_dim)]

        params_mesh = params_mesh = {'ndim':self.input_dim, 'xmin':self.min_values.tolist(), 'xmax':self.max_values.tolist(), 'nmail':n_test}
        mesh = CartesianMesh(**params_mesh)
        if self.input_dim ==2 :
            X, Y     = np.meshgrid(np.array(mesh.xmid[0]), np.array(mesh.xmid[1]))
            X_test   = np.hstack((X.flatten()[:,None], Y.flatten()[:,None]))  
            x_test = torch.tensor(X_test)
        elif self.input_dim == 3:
            # Create a 3D meshgrid from the midpoints along each axis
            X, Y, Z = np.meshgrid(
                np.array(mesh.xmid[0]),  # x-axis midpoints
                np.array(mesh.xmid[1]),  # y-axis midpoints
                np.array(mesh.xmid[2]),  # z-axis midpoints
                indexing='ij'  # ensures the axes are in (x, y, z) order
            )

            # Flatten and stack into a (N, 3) array where each row is a (x, y, z) point
            X_test = np.hstack((
                X.flatten()[:, None],
                Y.flatten()[:, None],
                Z.flatten()[:, None]
            ))

            # Convert to a PyTorch tensor
            x_test = torch.tensor(X_test, dtype=torch.float32)
        else:
            print('check again ndim')
        
        
        return x_test
    def add_currents_points(self,n_test):
        if not isinstance(n_test, list):
            n_test = [n_test for idim in range(self.input_dim)]

        params_mesh = params_mesh = {'ndim':self.input_dim, 'xmin':self.min_values.tolist(), 'xmax':self.max_values.tolist(), 'nmail':n_test}
        mesh = CartesianMesh(**params_mesh)
        if self.input_dim ==2:
            # construct mesh for currents
            Xp,Yp = np.meshgrid(np.array(mesh.xpos[0]), np.array(mesh.xmid[1]))
            xp_test = torch.tensor(np.hstack((Xp.flatten()[:,None], Yp.flatten()[:,None])))
            
            Xq,Yq = np.meshgrid(np.array(mesh.xmid[0]), np.array(mesh.xpos[1]))
            xq_test = torch.tensor(np.hstack((Xq.flatten()[:,None], Yq.flatten()[:,None])))
            
            xc_test=[xp_test,xq_test]
        elif self.input_dim == 3:
            # J_x: (xpos[0], xmid[1], xmid[2])
            Xp, Yp, Zp = np.meshgrid(mesh.xpos[0], mesh.xmid[1], mesh.xmid[2], indexing='ij')
            xp_test = torch.tensor(np.hstack((
                Xp.flatten()[:, None],
                Yp.flatten()[:, None],
                Zp.flatten()[:, None]
            )), dtype=torch.float32)

            # J_y: (xmid[0], xpos[1], xmid[2])
            Xq, Yq, Zq = np.meshgrid(mesh.xmid[0], mesh.xpos[1], mesh.xmid[2], indexing='ij')
            xq_test = torch.tensor(np.hstack((
                Xq.flatten()[:, None],
                Yq.flatten()[:, None],
                Zq.flatten()[:, None]
            )), dtype=torch.float32)

            # J_z: (xmid[0], xmid[1], xpos[2])
            Xr, Yr, Zr = np.meshgrid(mesh.xmid[0], mesh.xmid[1], mesh.xpos[2], indexing='ij')
            xr_test = torch.tensor(np.hstack((
                Xr.flatten()[:, None],
                Yr.flatten()[:, None],
                Zr.flatten()[:, None]
            )), dtype=torch.float32)

            # Group together
            xc_test = [xp_test, xq_test, xr_test]
        
        
        return xc_test
        
    
        


    
        
