import torch
import numpy as np


class quadrature:
    @staticmethod
    def angular_quadrature(n_mu=8, n_phi=8, dim_angular=3, 
                       mu_rule='legendre', phi_rule='legendre', 
                       device='cpu',**kwargs):
        """
        General angular quadrature in 2D or 3D with configurable rules for mu and phi.

        Args:
            n_mu: number of points for mu = cos(theta) ∈ [-1, 1]
            n_phi: number of points for phi ∈ [0, 2π]
            dim_angular: 2 or 3
            mu_rule: 'legendre' or 'uniform'
            phi_rule: 'legendre' or 'uniform'
            device: 'cpu' or 'cuda'

        Returns:
            omega: [Q, dim_angular] direction cosines
            weights: [Q] quadrature weights
        """

        # ---- phi quadrature in [0, 2pi] ----
        if phi_rule == 'legendre':
            phi_np, w_phi_np = np.polynomial.legendre.leggauss(n_phi)
            phi = torch.tensor(phi_np, dtype=torch.float32, device=device)
            w_phi = torch.tensor(w_phi_np, dtype=torch.float32, device=device)

            # Rescale from [-1, 1] → [0, 2π]
            phi = 0.5 * (phi + 1.0) * (2.0 * torch.pi)
            w_phi = w_phi * torch.pi

        elif phi_rule == 'uniform':
            phi = 2 * torch.pi * torch.arange(n_phi, device=device) / n_phi
            w_phi = torch.full((n_phi,), 2 * np.pi / n_phi, device=device)

        elif phi_rule == 'chebyshev':
            k = torch.arange(1, n_phi + 1, device=device)
            phi = torch.pi * (2 * k - 1) / (2 * n_phi)  # [0, π]
            phi = 2 * phi  # scale to [0, 2π]
            w_phi = torch.full((n_phi,), 2*torch.pi / n_phi, device=device)

        elif phi_rule == 'random':
            random_seed = kwargs.get('random_seed',1234)
            torch.random.manual_seed(random_seed)
            phi = 2 * torch.pi * torch.rand(n_mu, device=device)
            w_phi = torch.full((n_phi,), 2 * np.pi / n_phi, device=device)

        else:
            raise ValueError("phi_rule must be 'legendre' or 'uniform'")

        # ---- 2D case: directions on circle ----
        if dim_angular == 2:
            omega_x = torch.cos(phi)
            omega_y = torch.sin(phi)
            omega = torch.stack([omega_x, omega_y], dim=-1)  # [n_phi, 2]
            weights = w_phi
            return omega, weights

        # ---- 3D case ----
        elif dim_angular == 3:
            if mu_rule == 'legendre':
                mu_np, w_mu_np = np.polynomial.legendre.leggauss(n_mu)
                mu = torch.tensor(mu_np, dtype=torch.float32, device=device)
                w_mu = torch.tensor(w_mu_np, dtype=torch.float32, device=device)

            elif mu_rule == 'uniform':
                mu = torch.linspace(-1, 1, n_mu, device=device)
                w_mu = torch.full((n_mu,), 2.0 / n_mu, device=device)

            elif mu_rule == 'chebyshev':
                k = torch.arange(1, n_mu + 1, device=device)
                mu = torch.cos(torch.pi * (2 * k - 1) / (2 * n_mu))
                w_mu = torch.full((n_mu,), torch.pi / n_mu, device=device)
            elif mu_rule == 'random':
                random_seed = kwargs.get('random_seed',1234)
                torch.random.manual_seed(random_seed)
                # Uniform random sampling in [-1, 1]
                mu = 2.0 * torch.rand(n_mu, device=device) - 1.0
                w_mu = torch.full((n_mu,), 2.0 / n_mu, device=device)
            else:
                raise ValueError("mu_rule must be 'legendre' or 'uniform'")

            mu_grid, phi_grid = torch.meshgrid(mu, phi, indexing='ij')       # [n_mu, n_phi]
            w_grid = torch.outer(w_mu, w_phi)                                # [n_mu, n_phi]

            sin_theta = torch.sqrt((1.0 - mu_grid.clamp(-1.0, 1.0)**2))
            omega_x = sin_theta * torch.cos(phi_grid)
            omega_y = sin_theta * torch.sin(phi_grid)
            omega_z = mu_grid

            omega = torch.stack([omega_x, omega_y, omega_z], dim=-1).reshape(-1, 3)  # [Q, 3]
            weights = w_grid.flatten()
            return omega, weights

        else:
            raise ValueError("dim_angular must be 2 or 3")

        
    @staticmethod
    def angular_quadrature_theta_phi(n_theta=8, n_phi=8,
                                  theta_rule='legendre',
                                  phi_rule='legendre',
                                  device='cpu'):
        """
        Angular quadrature over the sphere using theta (0 to pi) and phi (0 to 2pi).

        Args:
            n_theta: Number of points for polar angle theta ∈ [0, π]
            n_phi: Number of points for azimuthal angle phi ∈ [0, 2π]
            theta_rule: 'legendre' or 'uniform'
            phi_rule: 'legendre', 'uniform', or 'chebyshev'
            device: 'cpu' or 'cuda'

        Returns:
            omega: [Q, 3] direction cosines
            weights: [Q] quadrature weights with Jacobian included
        """

        # ---- phi quadrature in [0, 2π] ----
        if phi_rule == 'legendre':
            phi_np, w_phi_np = np.polynomial.legendre.leggauss(n_phi)
            phi = torch.tensor(phi_np, dtype=torch.float32, device=device)
            w_phi = torch.tensor(w_phi_np, dtype=torch.float32, device=device)
            phi = 0.5 * (phi + 1.0) * (2.0 * torch.pi)
            w_phi = w_phi * torch.pi

        elif phi_rule == 'uniform':
            phi = torch.linspace(0, 2 * torch.pi, n_phi, endpoint=False, device=device)
            w_phi = torch.full((n_phi,), 2 * np.pi / n_phi, device=device)

        elif phi_rule == 'chebyshev':
            k = torch.arange(1, n_phi + 1, device=device)
            phi = torch.pi * (2 * k - 1) / (2 * n_phi)
            phi = 2 * phi
            w_phi = torch.full((n_phi,), 2 * torch.pi / n_phi, device=device)

        else:
            raise ValueError("phi_rule must be 'legendre', 'uniform', or 'chebyshev'")

        # ---- theta quadrature in [0, π] ----
        if theta_rule == 'legendre':
            theta_np, w_theta_np = np.polynomial.legendre.leggauss(n_theta)
            theta = torch.tensor(theta_np, dtype=torch.float32, device=device)
            w_theta = torch.tensor(w_theta_np, dtype=torch.float32, device=device)
            theta = 0.5 * (theta + 1.0) * torch.pi
            w_theta = w_theta * 0.5 * torch.pi

        elif theta_rule == 'uniform':
            theta = torch.linspace(0, torch.pi, n_theta, device=device)
            w_theta = torch.full((n_theta,), torch.pi / n_theta, device=device)

        else:
            raise ValueError("theta_rule must be 'legendre' or 'uniform'")

        # ---- Tensor product grid and weights ----
        theta_grid, phi_grid = torch.meshgrid(theta, phi, indexing='ij')
        w_grid = torch.outer(w_theta, w_phi) * torch.sin(theta_grid)  # Include Jacobian sin(theta)

        # ---- Direction cosines ----
        omega_x = torch.sin(theta_grid) * torch.cos(phi_grid)
        omega_y = torch.sin(theta_grid) * torch.sin(phi_grid)
        omega_z = torch.cos(theta_grid)

        omega = torch.stack([omega_x, omega_y, omega_z], dim=-1).reshape(-1, 3)
        weights = w_grid.flatten()

        return omega, weights


    # @staticmethod
    # def gauss_legendre_legendre(n_mu=8, n_phi=8, dim_angular=3, device='cpu'):
    #     """
    #     Gauss–Legendre quadrature in both mu = cos(theta) and phi.
        
    #     Args:
    #         n_mu: number of Gauss–Legendre points for mu in [-1, 1]
    #         n_phi: number of Gauss–Legendre points for phi (mapped to [0, 2π])
    #         dim_angular: 2 or 3
    #         device: 'cpu' or 'cuda'

    #     Returns:
    #         omega: [Q, dim_angular] direction cosines
    #         weights: [Q] quadrature weights
    #     """
        

    #     if dim_angular == 2:
    #         # Gauss–Legendre in φ ∈ [-1, 1], map to [0, 2π]
    #         x_phi_np, w_phi_np = np.polynomial.legendre.leggauss(n_phi)
    #         phi_np = 0.5 * (x_phi_np + 1.0) * 2.0 * np.pi
    #         w_phi_np = w_phi_np * (2.0 * np.pi / 2.0)

    #         phi = torch.tensor(phi_np, dtype=torch.float32, device=device)
    #         weights = torch.tensor(w_phi_np, dtype=torch.float32, device=device)

    #         omega_x = torch.cos(phi)
    #         omega_y = torch.sin(phi)
    #         omega = torch.stack([omega_x, omega_y], dim=-1)  # [Q, 2]
    #         return omega, weights

    #     elif dim_angular == 3:
    #         # Gauss–Legendre in mu = cos(theta)
    #         mu_np, w_mu_np = np.polynomial.legendre.leggauss(n_mu)
    #         mu = torch.tensor(mu_np, dtype=torch.float32, device=device)       # [n_mu]
    #         w_mu = torch.tensor(w_mu_np, dtype=torch.float32, device=device)   # [n_mu]

    #         # Gauss–Legendre in phi, rescaled to [0, 2π]
    #         phi_np, w_phi_np = np.polynomial.legendre.leggauss(n_phi)
    #         phi = torch.tensor(phi_np, dtype=torch.float32, device=device)
    #         w_phi = torch.tensor(w_phi_np, dtype=torch.float32, device=device)

    #         # Rescale phi ∈ [-1, 1] → [0, 2π]
    #         phi = 0.5 * (phi + 1.0) * 2.0 * torch.pi        # [n_phi]
    #         w_phi = w_phi * torch.pi                        # scale from 2π/2

    #         # Tensor grid
    #         mu_grid, phi_grid = torch.meshgrid(mu, phi, indexing='ij')      # [n_mu, n_phi]
    #         w_grid = torch.outer(w_mu, w_phi)                               # [n_mu, n_phi]

    #         sin_theta = torch.sqrt(1.0 - mu_grid**2)
    #         omega_x = sin_theta * torch.cos(phi_grid)
    #         omega_y = sin_theta * torch.sin(phi_grid)
    #         omega_z = mu_grid

    #         omega = torch.stack([omega_x, omega_y, omega_z], dim=-1).reshape(-1, 3)  # [Q, 3]
    #         weights = w_grid.flatten()                                               # [Q]
    #         return omega, weights

    #     else:
    #         raise ValueError("dim_angular must be 2 or 3")
        

    # @staticmethod
    # def gauss_legendre_legendre_theta_phi(n_theta=8, n_phi=8, dim_angular=3, device='cpu'):
    #     """
    #     Angular quadrature using theta ∈ [0, π] and phi ∈ [0, 2π],
    #     both with Gauss–Legendre quadrature (via change of variables).

    #     Args:
    #         n_theta: number of Gauss–Legendre points for θ ∈ [0, π]
    #         n_phi: number of Gauss–Legendre points for φ ∈ [0, 2π]
    #         dim_angular: 2 or 3
    #         device: 'cpu' or 'cuda'

    #     Returns:
    #         omega: [Q, dim_angular] tensor of direction cosines
    #         weights: [Q] quadrature weights
    #     """
    #     if dim_angular == 2:
    #        # Gauss–Legendre in φ ∈ [-1, 1], map to [0, 2π]
    #         x_phi_np, w_phi_np = np.polynomial.legendre.leggauss(n_phi)
    #         phi_np = 0.5 * (x_phi_np + 1.0) * 2.0 * np.pi
    #         w_phi_np = w_phi_np * (2.0 * np.pi / 2.0)

    #         phi = torch.tensor(phi_np, dtype=torch.float32, device=device)
    #         weights = torch.tensor(w_phi_np, dtype=torch.float32, device=device)

    #         omega_x = torch.cos(phi)
    #         omega_y = torch.sin(phi)
    #         omega = torch.stack([omega_x, omega_y], dim=-1)  # [Q, 2]
    #         return omega, weights

    #     elif dim_angular == 3:
    #         # Gauss–Legendre in theta ∈ [0, π]
    #         x_theta_np, w_theta_np = np.polynomial.legendre.leggauss(n_theta)
    #         theta_np = 0.5 * (x_theta_np + 1.0) * np.pi  # map [-1, 1] → [0, π]
    #         w_theta_np = w_theta_np * (np.pi / 2.0)

    #         # Gauss–Legendre in phi ∈ [0, 2π]
    #         x_phi_np, w_phi_np = np.polynomial.legendre.leggauss(n_phi)
    #         phi_np = 0.5 * (x_phi_np + 1.0) * 2.0 * np.pi
    #         w_phi_np = w_phi_np * (2.0 * np.pi / 2.0)

    #         # Convert to torch
    #         theta = torch.tensor(theta_np, dtype=torch.float32, device=device)
    #         phi = torch.tensor(phi_np, dtype=torch.float32, device=device)
    #         w_theta = torch.tensor(w_theta_np, dtype=torch.float32, device=device)
    #         w_phi = torch.tensor(w_phi_np, dtype=torch.float32, device=device)

    #         # Meshgrid
    #         theta_grid, phi_grid = torch.meshgrid(theta, phi, indexing='ij')
    #         w_grid = torch.outer(w_theta, w_phi)

    #         # Spherical to Cartesian
    #         omega_x = torch.sin(theta_grid) * torch.cos(phi_grid)
    #         omega_y = torch.sin(theta_grid) * torch.sin(phi_grid)
    #         omega_z = torch.cos(theta_grid)

    #         omega = torch.stack([omega_x, omega_y, omega_z], dim=-1).reshape(-1, 3)
    #         weights = w_grid.flatten()
    #         return omega, weights

    #     else:
    #         raise ValueError("dim_angular must be 2 or 3")