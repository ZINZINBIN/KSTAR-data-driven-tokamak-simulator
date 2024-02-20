import torch 
import torch.nn as nn
import torch.nn.functional as F

# 1-D Fourier Neural Operator
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, modes):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        
        self.scale = (1 / (in_channels * out_channels))
        self.weights_1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes, dtype = torch.cfloat))
        
    def comp1_mul1d(self, input:torch.Tensor, weights:torch.Tensor):
        return torch.einsum("bix,iox->box", input, weights)
    
    def forward(self, x:torch.Tensor):
        B = x.size()[0]
        
        x_ft = torch.fft.rfft(x)
        
        out_ft = torch.zeros(B, self.out_channels, x.size()[-1] // 2 + 1, device = x.device, dtype = torch.cfloat)
        out_ft[:,:,:self.modes] = self.comp1_mul1d(x_ft[:,:,:self.modes], self.weights_1)
        
        x = torch.fft.irfft(out_ft, n = x.size()[-1])
        return x