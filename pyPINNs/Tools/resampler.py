
import torch
from .visualization import visualization


class Resampler:
    def __init__(self, model, data, save_dir='', interval=5000, min_iter=20000, max_iter=100000, method='RAD'):
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
        residu = self.model.residual_PDE(self.data.X_train)
        residu = torch.pow(residu, 2).sum(dim=1, keepdim=True)

        # Optional visualization
        if iteration % self.interval == 0:
            visualization.viewCrossSection(
                self.data.X_train, S=residu, 
                pathFile=f'{self.save_dir}/residu_train_{iteration}.png'
            )

        self.data.X_train = self.model.adaptive_sampling(self.data.X_train, method=self.method)

        if iteration % self.interval == 0:
            visualization.viewCrossSection(
                self.data.X_train, S=None, 
                pathFile=f'{self.save_dir}/X_train_{iteration}.png'
            )
