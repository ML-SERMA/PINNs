import torch
import torch.nn as nn
from ..activations import get_activation

# Define block
class ResConvBlock(nn.Module):
	def __init__(self, channel_num):
		super(ResConvBlock, self).__init__()
		
		#TODO: 3x3 convolution -> relu
		#the input and output channel number is channel_num
		self.conv_block1 = nn.Sequential(
			nn.Conv2d(channel_num, channel_num, 3, padding=1),
			nn.BatchNorm2d(channel_num),
			nn.ReLU(),
		) 
		self.conv_block2 = nn.Sequential(
			nn.Conv2d(channel_num, channel_num, 3, padding=1),
			nn.BatchNorm2d(channel_num),
		)
		self.relu = nn.ReLU()
	
	def forward(self, x):
		
		#TODO: forward
		residual = x
		x = self.conv_block1(x)
		x = self.conv_block2(x)
		x = x + residual
		out = self.relu(x)
		return out
	

class ResBasicBlock(nn.Module):
    """
    IMplementation of the block used in the Deep Ritz
    Paper

    Parameters:
    in_N  -- dimension of the input
    width -- number of nodes in the interior middle layer
    out_N -- dimension of the output
    phi   -- activation function used
    """

    def __init__(self, in_N, width, out_N,activation='Tanh'):
        super(ResBasicBlock, self).__init__()
        # create the necessary linear layers
        self.L1 = nn.Linear(in_N, width)
        self.L2 = nn.Linear(width, out_N)
        # choose appropriate activation function
        self.phi = get_activation(activation)

    def forward(self, x):
        return self.phi(self.L2(self.phi(self.L1(x)))) + x