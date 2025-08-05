import torch

class Swish(torch.nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class Snake(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = 0.5

    def forward(self, x):
        return x + torch.sin(self.alpha * x) ** 2 / self.alpha


class Sin(torch.nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)
    
    
class ReLU2(torch.nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, x):
        return torch.pow(torch.relu(x), 2)

# Custom activation function
class RegionBasedActivation(torch.nn.Module):
    def forward(self, x, x_input):
        x1, x2 = x_input[:, 0:1], x_input[:, 1:2]  # Extract original input

        # Choose activation function for all neurons in the layer
        # activation_choice = torch.where((x1 >= 0.5) & (x2 <= 0.5), torch.tanh(x),  #Tanh
        #                     torch.where((x1 <= 0.5) & (x2 >= 0.5), torch.tanh(x),  # Tanh
        #                     torch.sin(x) ))  # Sin for last case

        # activation_choice = torch.where((x1 > 1.0/3) & (x1 < 2.0/3) & (x2 > 1.0/3) & (x2 < 2.0/3), Relu2()(x), torch.relu(x) )  
        activation_choice = ((x1>=0.5)&(x2<=0.5))*torch.tanh(x) + ((x1<=0.5)&(x2>=0.5))*torch.tanh(x) +\
                            ((x1<0.5)&(x2<0.5))*torch.sin(x) + ((x1>0.5)&(x2>0.5))*torch.sin(x)


        
        return activation_choice
        


def get_activation(name):
    if name in ['tanh', 'Tanh']:
        return torch.nn.Tanh()
    elif name in ['relu', 'ReLU']:
        return torch.nn.ReLU(inplace=True)
    elif name in ['lrelu', 'LReLU']:
        return torch.nn.LeakyReLU(inplace=True)
    elif name in ['sigmoid', 'Sigmoid']:
        return torch.nn.Sigmoid()
    elif name in ['softplus', 'Softplus']:
        return torch.nn.Softplus(beta=4)
    elif name in ['celu', 'CeLU']:
        return torch.nn.CELU()
    elif name in ['swish','Swish']:
        return Swish()
    elif name in ['sin','Sin']:
        return Sin()
    elif name in ['snake','Snake']:
        return Snake()
    elif name in ['RegionBase']:
        return RegionBasedActivation()
    elif name in ['relu2','ReLU2']:
        return ReLU2()
    else:
        raise ValueError('Unknown activation function')