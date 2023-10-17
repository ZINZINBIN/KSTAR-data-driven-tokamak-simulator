import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from typing import List
import numpy as np

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x:torch.Tensor):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1, self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs:int, num_channels:List, dilation_size = 2, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        if np.isscalar(dilation_size): dilation_size = [dilation_size**i for i in range(num_levels)]
        for i in range(num_levels):
            dilation = dilation_size[i]
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                     padding=(kernel_size-1) * dilation, dilation=dilation, 
                                     dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x:torch.Tensor):
        return self.network(x)
    
class TCN(nn.Module):
    def __init__(self, input_size, hidden_dim, num_channels, kernel_size, dropout, dilation_size):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout, dilation_size=dilation_size)
        self.linear = nn.Linear(num_channels[-1], hidden_dim)

    def forward(self, inputs:torch.Tensor):
        """Inputs have to have dimension (N, C_in, L_in)"""
        x = self.tcn(inputs) 
        x = self.linear(x.permute(0,2,1)).mean(dim = 1)
        return torch.sigmoid(x)
    
def calc_seq_length(kernel_size,dilation_sizes,nlevel):
    """Assumes kernel_size scalar, dilation_size exponential increase"""
    if np.isscalar(dilation_sizes): dilation_sizes = dilation_sizes**np.arange(nlevel)
    return 1 + 2*(kernel_size-1)*np.sum(dilation_sizes)

def calc_dilation(kernel_size, dilation_sizes, nlevel, nrecept):
    nrecepttotal = calc_seq_length(kernel_size, dilation_sizes, nlevel)
    nlastlevel = calc_seq_length(kernel_size, dilation_sizes, nlevel-1)
    last_dilation = int(np.ceil((nrecept - nlastlevel)/(2.*(kernel_size-1))))
    dilation_sizes = (dilation_sizes**np.arange(nlevel-1)).tolist() + [last_dilation]
    return dilation_sizes