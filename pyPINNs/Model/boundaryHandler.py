import torch
import torch.nn as nn

class HardBCWrapperGeneral(nn.Module):
    def __init__(self, model, bc_info):
        """
        model: neural network model
        bc_info: list of dictionaries with:
            - region_fn(x): mask of shape [N,1]
            - type: "dirichlet_flux" | "neumann_current" | "robin"
            - bc_value_fn(x): for Dirichlet (optional)
            - g_fn(x): for Neumann/Robin
            - normal_fn(x): returns normal vectors [N, D]
            - blend_fn(x): for Dirichlet blending (optional)
            - alpha, beta: for Robin condition
            - flux_index: index of scalar flux output (usually 0)
            - current_indices: indices of current components
        """
        super().__init__()
        self.model = model
        self.bc_info = bc_info

    def forward(self, x):
        raw_out = self.model(x)  # [N, M]
        out = raw_out.clone()

        for bc in self.bc_info:
            mask = bc["region_fn"](x).bool().squeeze()
            if not mask.any():
                continue

            x_b = x[mask]
            raw_b = raw_out[mask]

            flux_idx = bc.get("flux_index", 0)
            current_idxs = bc.get("current_indices", list(range(1, x.shape[1] + 1)))

            if bc["type"] == "dirichlet_flux":
                g = bc["bc_value_fn"](x_b)  # [Nb, 1]
                B = bc.get("blend_fn", self._default_blend_fn)(x_b)  # [Nb, 1]
                out[mask, flux_idx] = B[:, 0] * raw_b[:, flux_idx] + g[:, 0]

            elif bc["type"] == "neumann_current":
                n = bc["normal_fn"](x_b)  # [Nb, D]
                g = bc["g_fn"](x_b)       # [Nb, 1]
                J = raw_b[:, current_idxs]  # [Nb, D]
                J_dot_n = (J * n).sum(dim=1, keepdim=True)
                J_corr = J - J_dot_n * n + g * n
                out[mask, current_idxs] = J_corr

            elif bc["type"] == "robin":
                alpha = bc["alpha"]
                beta = bc["beta"]
                g = bc["g_fn"](x_b)       # [Nb, 1]
                B = bc.get("blend_fn", self._default_blend_fn)(x_b)  # [Nb, 1]
                phi = raw_b[:, flux_idx:flux_idx+1]
                corrected = (beta * phi * B + g) / (alpha + beta)
                out[mask, flux_idx] = corrected[:, 0]

        return out

    def _default_blend_fn(self, x):
        return torch.prod(x * (1 - x), dim=1, keepdim=True)




def generate_bc_info_from_list(domain, boundary_conditions, flux_index=0, current_indices=None, tol=1e-6):
    ndim = len(domain)
    if current_indices is None:
        current_indices = list(range(1, 1 + ndim))

    bc_info_list = []

    for i in range(ndim):  # loop x, y, z
        for side in [0, 1]:  # min, max
            bc = boundary_conditions[i][side]
            if bc is None:
                continue

            face_val = domain[i][side]
            side_str = "min" if side == 0 else "max"

            def region_fn_factory(dim, val):
                return lambda x, dim=dim, val=val: (torch.abs(x[:, dim] - val) < tol).unsqueeze(1).float()

            def normal_fn_factory(dim, side):
                def normal_fn(x):
                    n = torch.zeros((x.shape[0], ndim), device=x.device)
                    n[:, dim] = -1.0 if side == "min" else 1.0
                    return n
                return normal_fn

            info = {
                "region_fn": region_fn_factory(i, face_val),
                "flux_index": flux_index,
                "current_indices": current_indices,
                "type": bc["type"]
            }

            if bc["type"] == "dirichlet_flux":
                info["bc_value_fn"] = bc["bc_value_fn"]
                info["blend_fn"] = bc.get("blend_fn", None)

            elif bc["type"] == "neumann_current":
                info["g_fn"] = bc["g_fn"]
                info["normal_fn"] = bc.get("normal_fn", normal_fn_factory(i, side_str))

            elif bc["type"] == "robin":
                info.update({
                    "alpha": bc["alpha"],
                    "beta": bc["beta"],
                    "g_fn": bc["g_fn"],
                    "normal_fn": bc.get("normal_fn", normal_fn_factory(i, side_str)),
                    "blend_fn": bc.get("blend_fn", None)
                })

            bc_info_list.append(info)

    return bc_info_list


##===========================================================================================
# Example 


domain = [[0.0, 1.0], [0.0, 1.0]]  # 2D unit square

boundary_conditions = [
    [  # x-min, x-max
        {"type": "dirichlet_flux", "bc_value_fn": lambda x: torch.zeros((x.shape[0], 1))},
        {"type": "robin", "alpha": 1.0, "beta": 2.0, "g_fn": lambda x: torch.ones((x.shape[0], 1))}
    ],
    [  # y-min, y-max
        {"type": "neumann_current", "g_fn": lambda x: torch.zeros((x.shape[0], 1))},
        {"type": "neumann_current", "g_fn": lambda x: 0.5 * torch.ones((x.shape[0], 1))}
    ]
]

bc_info = generate_bc_info_from_list(domain, boundary_conditions)

# model_bc = HardBCWrapperGeneral(model, bc_info)
