import torch


class LLAF(torch.nn.Module):
    def __init__(self, input_channel, output_channel, scaler=10, channel_wise=False):
        super(LLAF, self).__init__()
     
        self.fc = torch.nn.Linear(input_channel, output_channel).double()
        self.channel_wise = channel_wise 
        if self.channel_wise:
            self.coeff2 = torch.nn.parameter.Parameter(1.0 / scaler * torch.ones(output_channel).double())
        else:
            self.coeff2 = torch.nn.parameter.Parameter(1.0 / scaler * torch.ones(1).double())
        self.scaler = scaler 
        self.disable = False 
        # print(self.scaler)
    def forward(self, x):
        if self.disable:
            y = self.fc(x) * self.coeff2.detach() * self.scaler 
        else:
            y = self.fc(x) * self.coeff2 * self.scaler 
        return y 
