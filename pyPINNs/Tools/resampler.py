
import torch
import torch.nn as nn
from .visualization import visualization


class SamplerNet(nn.Module):
    def __init__(self, dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
            nn.Softplus()  # ensure positive density
        )

    def forward(self, X):
        return self.net(X)  # output: unnormalized sampling intensity

class Resampler:
    def __init__(self, model, data, save_dir='', interval=4000, min_iter=20000, max_iter=100000, method='EVO', sampler_dim=2, sampler_hidden=64, lr_sampler=1e-3):
        self.model = model
        self.data = data
        self.save_dir = save_dir
        self.interval = interval
        self.min_iter = min_iter
        self.max_iter = max_iter
        self.method = method

        # Neural Adaptive Sampler
        if method == 'NAS':
            self.sampler = SamplerNet(sampler_dim, sampler_hidden).to(self.data.device)
            self.opt_sampler = torch.optim.Adam(self.sampler.parameters(), lr=lr_sampler)


    def should_resample(self, iteration):
        return (
            iteration % self.interval == 0 and 
            self.min_iter < iteration <= self.max_iter
        )

    def apply(self, iteration):
        print(f'==>> Resampling ({self.method}) at iteration {iteration}')
        X = self.data.X_train
        n_total = X.shape[0]
        residu = self.model.residual_PDE(X) # [N, G,(1+d)] for mixed form
        residu = torch.sum(residu**2, dim=tuple(range(1, residu.ndim))) # 1D tensor
        # residu = torch.mean(torch.sum(residu**2, dim=-1), dim=-1)
        
        show=True
        # Optional visualization
        if iteration % self.interval == 0 and show :
            visualization.viewCrossSection(
                self.data.X_train, S=residu, 
                pathFile=f'{self.save_dir}/residu_train_{iteration}.png'
            )
        if self.method =='EVO':
            # X = self.data.X_train
            # n_total = X.shape[0]
            # residu = self.model.residual_PDE(X) # [N, G,(1+d)] for mixed form
            # residu = torch.sum(residu**2, dim=tuple(range(1, residu.ndim))) # 1D tensor
            
            # threshold = torch.mean(residu)
            threshold = torch.quantile(residu, 0.75) # we can take mean or quantile here
            indices = torch.nonzero((residu>=threshold)*1,as_tuple=True)[0]
            X_evo = X[indices]
            X_evo = X_evo.clone().detach().requires_grad_(True)

            n_rand = X.shape[0]-X_evo.shape[0]
            X_rand = self.data.domain.add_collocation_points(n_collocation=n_rand,random_seed=None)
            X_rand = X_rand.requires_grad_().float().to(self.data.device) 

            X_new = torch.vstack((X_evo,X_rand))
            self.data.X_train = X_new.clone().detach().requires_grad_(True)


        # if iteration % self.interval == 0 and show :
        #     visualization.viewCrossSection(
        #         self.data.X_train, S=None, 
        #         pathFile=f'{self.save_dir}/X_train_new_{iteration}.png'
        #     )

        # === RES METHOD ===
        elif self.method == 'RES':
            # X = self.data.X_train
            # n_total = X.shape[0]
            # residu = self.model.residual_PDE(X) # [N, G,(1+d)] for mixed form
            # residu = torch.sum(residu**2, dim=tuple(range(1, residu.ndim))) # 1D tensor

            # Normalize residuals into probabilities
            prob = residu / torch.sum(residu)
            eps = 0.2  # exploration factor
            p_mix = (1 - eps) * prob + eps * torch.ones_like(prob) / len(prob)
            p_mix = p_mix / torch.sum(p_mix)  # normalize (safety)
            indices = torch.multinomial(p_mix, n_total, replacement=True)
            X_new = X[indices]
            self.data.X_train = X_new.clone().detach().requires_grad_(True)
            print(f"RES resampling: eps={eps}, replacement=True, {n_total} samples.")

        # === HYBRID METHOD ===
        elif self.method == 'HYBRID':
            # X = self.data.X_train
            # n_total = X.shape[0]
            # residu = self.model.residual_PDE(X) # [N, G,(1+d)] for mixed form
            # residu = torch.sum(residu**2, dim=tuple(range(1, residu.ndim))) # 1D tensor

            frac_res = 0.8  # 80% residual-weighted, 20% random
            n_res = int(frac_res * n_total)
            n_rand = n_total - n_res

            # Residual-weighted sampling
            prob = residu / torch.sum(residu)
            indices_res = torch.multinomial(prob, n_res, replacement=True)
            X_res = X[indices_res]

            # Uniform random points for exploration
            X_rand = self.data.domain.add_collocation_points(n_collocation=n_rand,random_seed=None)
            X_rand = X_rand.requires_grad_().float().to(self.data.device)

            # Combine
            X_new = torch.vstack((X_res, X_rand))
            self.data.X_train = X_new.clone().detach().requires_grad_(True)
            print(f"Hybrid resampling: {n_res} residual-weighted + {n_rand} random points.")

        elif self.method == 'RAD':
            # === Parameters ===
            greedyNum = 2*256                      # number of points to select per RAD step
            probeNum = 4*1024                       # number of new probe points
            max_points = 20*1024                    # max allowed training points
            eps = 0.1                             # optional random exploration fraction

            # === Generate new probe points ===
            X_probe = self.data.domain.add_collocation_points(n_collocation=probeNum, random_seed=None)
            X_probe = X_probe.clone().detach().float().to(self.data.device).requires_grad_(True)
            self.model.get_cross_sections(X_probe)
            # === Compute residuals ===
            residu = self.model.residual_PDE(X_probe)                    # [N, G, (1+d)]
            residu = torch.sum(residu**2, dim=tuple(range(1, residu.ndim)))  # [N] sum over groups/components

            # === Optional residual-based mixture for exploration ===
            if eps > 0:
                prob = residu / torch.sum(residu)
                p_mix = (1 - eps) * prob + eps * torch.ones_like(prob) / len(prob)
                p_mix = p_mix / torch.sum(p_mix)  # normalize
            else:
                p_mix = residu / torch.sum(residu)

            # === Sample high-residual points ===
            indices = torch.multinomial(p_mix, greedyNum, replacement=True)
            X_new = X_probe[indices].clone().detach().requires_grad_(True)

            # === Append to existing training set ===
            X = torch.vstack((self.data.X_train, X_new))

            # === Optional: cap dataset size ===
            if X.shape[0] > max_points:
                X = X[-max_points:]  # keep the most recent points

            self.data.X_train = X
            print(f"==>> X-RAD: {X.shape} (added {greedyNum} points, eps={eps})")

        elif self.method == 'RAR':
            # === Parameters ===
            n_probe = 4*1024          # number of new candidate points
            top_k = 2*256            # number of high-residual points to keep
            eps = 0.1               # exploration fraction (0 = none, 0.1 = 10%)
            max_points = 20*1024       # maximum allowed training points

            # === Step 1: Generate new probe points ===
            X_probe = self.data.domain.add_collocation_points(n_collocation=n_probe)
            X_probe = X_probe.clone().detach().float().to(self.data.device).requires_grad_(True)
            self.model.get_cross_sections(X_probe)
            # === Step 2: Compute residuals ===
            residu = self.model.residual_PDE(X_probe)                  # [N, G, (1+d)]
            # Aggregate residuals over groups/components
            residu = torch.sum(residu**2, dim=tuple(range(1, residu.ndim)))  # [N]

            # === Step 3: Select top-k points ===
            _, indices_topk = torch.topk(residu, k=top_k, dim=0, sorted=False)
            X_topk = X_probe[indices_topk]

            # === Step 4: Optional exploration (random points) ===
            n_explore = int(eps * top_k)
            if n_explore > 0:
                X_rand = self.data.domain.add_collocation_points(n_collocation=n_explore)
                X_rand = X_rand.clone().detach().float().to(self.data.device).requires_grad_(True)
                X_selected = torch.vstack([X_topk[:-n_explore], X_rand])
            else:
                X_selected = X_topk

            # === Step 5: Append to existing training points ===
            X = torch.vstack([self.data.X_train, X_selected])

            # === Step 6: Cap dataset size ===
            if X.shape[0] > max_points:
                X = X[-max_points:]  # keep the most recent points

            self.data.X_train = X
            print(f"==>> X-RAR: {X.shape} (top-k={top_k}, eps={eps}, appended points)")

        if self.method == 'NAS':
            max_points = 20*1024       # maximum allowed training points
            # 1. Update sampler (maximize residuals)
            self._update_sampler()

            # 2. Sample new points from sampler
            X_new = self._sample_from_sampler(n_samples=1024)

            # 3. Append new points to training set
            X = torch.vstack([self.data.X_train, X_new])

            # 4. Cap dataset
            if X.shape[0] > max_points:
                X = X[-max_points:]
            self.data.X_train = X

            print(f"==>> X-NAS: {X.shape} (added {X_new.shape[0]} points)")


    # --- Step 1: Sampler update (adversarial) ---
    def _update_sampler(self):
        n_probe = 4000
        entropy_reg=0.01
        # Generate candidate points
        X_probe = self.domain.add_collocation_points(n_collocation=n_probe)
        X_probe = X_probe.clone().detach().float().to(self.data.device)
        self.model.get_cross_sections(X_probe)
        # Compute residuals (no grad for PINN)
        with torch.no_grad():
            residu = self.model.residual_PDE(X_probe)          # [N, G, (1+d)]
            residu = torch.sum(residu**2, dim=tuple(range(1, residu.ndim)))  # [N]
            # mu = residu / (residu.sum() + 1e-12)  # normalized residual distribution

        # Compute sampler density
        s_pred = self.sampler(X_probe).squeeze() + 1e-12  # prevent zeros
        p_phi = s_pred / s_pred.sum()

        # KL divergence loss
        # loss_sampler = torch.sum(mu * (torch.log(mu) - torch.log(p_phi)))

        # Adversarial loss (maximize residuals)
        loss_sampler = -torch.sum(p_phi * residu)

        # Optional entropy regularization
        entropy = -torch.sum(p_phi * torch.log(p_phi + 1e-12))
        loss_sampler -= entropy_reg * entropy  # encourage exploration

        # Update sampler
        self.opt_sampler.zero_grad()
        loss_sampler.backward()
        self.opt_sampler.step()

        print(f"Sampler updated: adversarial loss={loss_sampler.item():.4e}, entropy={entropy.item():.4e}")

    # --- Step 2: Sample new points from sampler ---
    def _sample_from_sampler(self, n_samples):
        n_probe = max(4 * n_samples, 1024)  # generate enough candidates
        X_probe = self.domain.add_collocation_points(n_collocation=n_probe)
        X_probe = X_probe.clone().detach().float().to(self.data.device).requires_grad_(True)

        with torch.no_grad():
            s_values = self.sampler(X_probe).squeeze() + 1e-12
            prob = s_values / s_values.sum()
            indices = torch.multinomial(prob, n_samples, replacement=True)
            X_new = X_probe[indices].clone().detach().requires_grad_(True)

        return X_new


