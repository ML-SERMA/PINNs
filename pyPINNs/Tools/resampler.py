
import torch
from .visualization import visualization


class Resampler:
    def __init__(self, model, data, save_dir='', interval=4000, min_iter=20000, max_iter=100000, method='EVO'):
        self.model = model
        self.data = data
        self.save_dir = save_dir
        self.interval = interval
        self.min_iter = min_iter
        self.max_iter = max_iter
        self.method = method


    def should_resample(self, iteration):
        return (
            iteration % self.interval == 0 and 
            self.min_iter < iteration <= self.max_iter
        )

    def apply(self, iteration):
        print('Resampling!')
        X = self.data.X_train
        residu = self.model.residual_PDE(X) # [N, G*(1+d)] for mixed form
        residu = torch.sum(residu**2, dim=tuple(range(1, residu.ndim))) # 1D tensor
        # distribution = residu/torch.sum(residu)
        # greedyNum = self.data.X_train.shape[0]
        # indices = torch.multinomial(distribution,greedyNum)
        # X_new = X_test[indices]
        # X_new = X_new.clone().detach().requires_grad_(True)
        # X = torch.vstack((X,X_new))

        show=False
        # Optional visualization
        if iteration % self.interval == 0 and show :
            visualization.viewCrossSection(
                self.data.X_train, S=residu, 
                pathFile=f'{self.save_dir}/residu_train_{iteration}.png'
            )
        if self.method =='EVO':
            threshold = torch.mean(residu)
            indices = torch.nonzero((residu>=threshold)*1,as_tuple=True)[0]
            X_evo = X[indices]
            X_evo = X_evo.clone().detach().requires_grad_(True)

            n_rand = X.shape[0]-X_evo.shape[0]
            X_rand = self.data.domain.add_collocation_points(n_collocation=n_rand,random_seed=None)
            X_rand = X_rand.requires_grad_().float().to(self.data.device) 

            X_new = torch.vstack((X_evo,X_rand))
            X_new = X_new.clone().detach().requires_grad_(True)
            print(f"==>> X-EVO: {X_new.shape}")
            self.data.X_train = X_new

        if iteration % self.interval == 0 and show:
            visualization.viewCrossSection(
                self.data.X_train, S=None, 
                pathFile=f'{self.save_dir}/X_train_new_{iteration}.png'
            )
