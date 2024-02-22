import torch 
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from pytorch_model_summary import summary

# 1-D Fourier Neural Operator
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, modes:int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        
        self.scale = (1 / (in_channels * out_channels))
        self.weights_1 = nn.Parameter(self.scale * torch.view_as_real(torch.rand(in_channels, out_channels, self.modes, dtype = torch.cfloat)))
        
    def comp1_mul1d(self, input:torch.Tensor, weights:torch.Tensor):
        return torch.einsum("bix,iox->box", input, torch.view_as_complex(weights))
    
    def forward(self, x:torch.Tensor):
        B = x.size()[0]
        
        x_ft = torch.fft.rfft(x)
        
        out_ft = torch.zeros(B, self.out_channels, x.size()[-1] // 2 + 1, device = x.device, dtype = torch.cfloat)
        out_ft[:,:,:self.modes] = self.comp1_mul1d(x_ft[:,:,:self.modes], self.weights_1)
        
        x = torch.fft.irfft(out_ft, n = x.size()[-1])
        return x

class FNO(nn.Module):
    def __init__(
        self,        
        state_dim:int, 
        control_dim:int, 
        modes:int,
        width:int
        ):
        super(FNO, self).__init__()
        self.state_dim = state_dim
        self.control_dim = control_dim
        
        self.modes1 = modes
        self.width = width
        
        # Encoder for handling the trajectory of the previous input and state information
        self.enc = nn.LSTM(state_dim + control_dim, state_dim, 1, True, False)
        self.fc0 = nn.Linear(1, self.width)
        
        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, traj: torch.Tensor, state : torch.Tensor, control:torch.Tensor):
        '''
            traj: (B,T-1,S+C)
            control: (B,1,C)
            state: (B,1,S)
        '''
        
        if state.ndim == 2:
            state = state.unsqueeze(1)
            control = control.unsqueeze(1)
        
        h0 = torch.zeros((1, traj.size(0), self.state_dim)).to(traj.device)
        c0 = torch.zeros((1, traj.size(0), self.state_dim)).to(traj.device)
        
        vec = torch.cat((state, control), dim = -1)
        traj = torch.cat((traj, vec), dim = 1)
        
        _, (hn, _) = self.enc(traj.permute(1,0,2), (h0, c0))
        hn = hn.permute(1,2,0) # hn: (B,S,1)
        
        x = self.fc0(hn) # x: (B,S,W)
        x = x.permute(0, 2, 1) 
    
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2
        
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        
        x = x.squeeze(-1)
        return x

    def summary(self, device='cpu'):
        traj = torch.zeros((1, 20, self.state_dim + self.control_dim)).to(device)
        state = torch.zeros((1, 1, self.state_dim)).to(device)
        control = torch.zeros((1, 1, self.control_dim)).to(device)
        summary(self, traj, state, control, batch_size = 1, show_input = True, print_summary=True)
        
if __name__ == "__main__":
    
     # torch device state
    print("================= device setup =================")
    print("torch device avaliable : ", torch.cuda.is_available())
    print("torch current device : ", torch.cuda.current_device())
    print("torch device num : ", torch.cuda.device_count())

    # torch cuda initialize and clear cache
    torch.cuda.init()
    torch.cuda.empty_cache()
    
    # device allocation
    if(torch.cuda.device_count() >= 1):
        device = "cuda:" + str(0)
    else:
        device = 'cpu'
    
    # configuration
    state_dim = 12
    control_dim = 8
    time_length = 20
    width = 32
    modes = 4
   
    # model
    network = FNO(state_dim, control_dim, modes, width)
    network.to(device)
    network.summary(device)
    
    # input
    batch_size = 32
    state = torch.zeros((batch_size, 1, state_dim))
    control = torch.zeros((batch_size, 1, control_dim))
    traj = torch.zeros((batch_size, time_length, state_dim + control_dim))
    
    output = network(traj.to(device), state.to(device), control.to(device))
    print("output size: ", output.size())