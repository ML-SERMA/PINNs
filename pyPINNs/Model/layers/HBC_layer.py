
import torch

class HBC_layer(torch.nn.Module):
    def __init__(self, ):
        super(HBC_layer, self).__init__()
     
        
    def forward(self, x):
        if self.disable:
            y = self.fc(x) * self.coeff2.detach() * self.scaler 
        else:
            y = self.fc(x) * self.coeff2 * self.scaler 
        return y 