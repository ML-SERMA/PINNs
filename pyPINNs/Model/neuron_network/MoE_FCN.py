import torch
from .FCN import FCN_FF

class MoE_FCN(torch.nn.Module):
    def __init__(self,MoE_layers,MoE_activations, gate_layers, gate_activation,device,fourier_mapping_size=None,hard_BC =None):
        super().__init__() 
        self.MoE_layers = MoE_layers
        self.MoE_activations = MoE_activations
        self.gate_layers = gate_layers
        self.gate_activation = gate_activation
        self.device = device
        self.fourier_mapping_size =fourier_mapping_size
        self.hard_BC = hard_BC
        self.n_models = len(self.MoE_layers)
        self.pinns = torch.nn.ModuleList([FCN_FF(self.MoE_layers[i],self.MoE_activations[i],self.device,self.fourier_mapping_size,self.hard_BC).to(self.device)
                    for i in range(self.n_models)])
        if self.n_models >1:
            self.gate = FCN_FF(self.gate_layers,self.gate_activation,self.device,fourier_mapping_size=None,hard_BC =None).to(self.device)
    def forward(self,x):

        if torch.is_tensor(x) != True:         
            x = torch.from_numpy(x) 
        # create predictions with set of experts
        u = [self.pinns[i](x) for i in range(self.n_models)]

        if self.n_models >1:
            logits = self.gate(x)
            weights = torch.nn.Softmax(dim=1)(logits)
            weighted_u = torch.sum(torch.stack([u[i]*weights[:,i:i+1] for i in range(self.n_models)],dim=1),dim=1)
            # weighted_u = torch.sum(weights[:,:,None]*torch.stack(u,dim=1),dim=1)
            # weighted_u = torch.sum(torch.stack([self.pinns[i](x)*weights[:,i:i+1] for i in range(self.n_models)],dim=1),dim=1)

            # input('enter')

            return weighted_u
        else:
            return u[0]