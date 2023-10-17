import torch
import torch.nn as nn
import numpy as np
from src.simulators.basic_model.noise_layer import NoiseLayer

class ResBlock(nn.Module):
    def __init__(self, in_channels : int, out_channels : int):
        super().__init__()
        self.resblock = nn.Sequential(
            nn.Conv2d(in_channels, (in_channels + out_channels) // 2, kernel_size = 3, padding = 1),
            nn.BatchNorm2d((in_channels + out_channels) // 2),
            nn.ReLU(),
            nn.Conv2d((in_channels + out_channels) // 2, out_channels, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU()
    
    def forward(self, x : torch.Tensor):
        out = self.resblock(x)
        out = out + x
        out = self.relu(out)
        return out

class ContourRegressor(nn.Module):
    def __init__(
        self, 
        nx : int, 
        ny : int, 
        params_dim : int,
        n_PFCs :int,
        Rmin : float = 1, 
        Rmax : float = 4, 
        Zmin : float = -1.0, 
        Zmax : float = 1.0
        ):
        super().__init__()
        self.nx = nx
        self.ny = ny
        
        self.params_dim = params_dim
        self.n_PFCs = n_PFCs
        
        self.Rmin = Rmin
        self.Zmin = Zmin
        self.Rmax = Rmax
        self.Zmax = Zmax
        
        self.encoder_params = nn.Sequential(
            nn.Linear(params_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )
        
        self.encoder_PFCs = nn.Sequential(
            nn.Linear(n_PFCs, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )
        
        self.noise = NoiseLayer(0, 0.25)
        self.conv_layer = nn.Sequential(
            nn.LayerNorm((nx,ny)),
            nn.ReLU(),
            nn.Conv2d(1,32, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            ResBlock(32, 32),
            nn.Conv2d(32, 64,kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            ResBlock(64, 64),
        )
        
        input_dim = self.compute_hidden_dim() + 64 * 2
        
        self.cen_regressor = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 2),
        )
        
        self.rad_regressor = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 256),
        )
        
    def compute_hidden_dim(self):
        self.eval()
        sample_data = torch.zeros((1,self.nx,self.ny))
        sample_data = self.conv_layer(sample_data.unsqueeze(1)).view(1,-1)
        return sample_data.size()[1]
        
    def forward(self, x : torch.Tensor, x_params : torch.Tensor, x_PFCs : torch.Tensor):
        batch_size = x.size()[0]
        x = self.noise(x)
        x = self.conv_layer(x.unsqueeze(1)).view(batch_size, -1)
        
        x_params = self.noise(x_params)
        x_PFCs = self.noise(x_PFCs)
        x_params = self.encoder_params(x_params)
        x_PFCs = self.encoder_PFCs(x_PFCs)
        
        x = torch.concat([x, x_params, x_PFCs], axis = 1)
        
        cen = self.cen_regressor(x.view(batch_size, -1)).clamp(
            min = torch.Tensor([1.6,-0.1]).repeat(batch_size,1).to(x.device),
            max = torch.Tensor([1.9, 0.1]).repeat(batch_size,1).to(x.device),
        )
        
        rad = self.rad_regressor(x.view(batch_size, -1)).clamp(
            min = torch.Tensor([0.25]).unsqueeze(0).repeat(batch_size,1).to(x.device),
            max = torch.Tensor([0.9]).unsqueeze(0).repeat(batch_size,1).to(x.device),
        )
        
        cen = self.cen_regressor(x)
        rad = self.rad_regressor(x)
        
        return cen, rad
    
    def compute_rzbdys(self, x:torch.Tensor, x_params : torch.Tensor, x_PFCs : torch.Tensor, smooth : bool = True):
    
        with torch.no_grad():
            cen, rad = self.forward(x, x_params, x_PFCs)
            cen = cen.detach().squeeze(0).cpu().numpy()
            rad = rad.detach().squeeze(0).cpu().numpy()
            
            theta = np.linspace(0,2*3.142,256)
            
            # smoothing
            if smooth:
                rad_ = np.zeros((len(rad) + 4))
                rad_[2:-2] = rad
                rad_[0:2] = rad[0:2]
                rad_[-2:] = rad[-2:]
                rad = np.convolve(rad_, [0.2,0.2,0.2,0.2,0.2], 'valid')
            
            rzbdys = np.zeros((256,2))
            rzbdys[:,0] = cen[0] + rad * np.cos(theta)
            rzbdys[:,1] = cen[1] + rad * np.sin(theta)
            
        return rzbdys

    def compute_boundary_index(self, rzbdys:torch.Tensor):
        return None
    
    def compute_shape_parameters(self, x : torch.Tensor, x_params : torch.Tensor, x_PFCs : torch.Tensor, smooth : bool = False):
        with torch.no_grad():
            rzbdy = self.compute_rzbdys(x, x_params, x_PFCs, smooth)
            big_ind = 1
            small_ind = 1

            len2 = len(rzbdy)

            for i in range(len2-1):

                if (rzbdy[i,1] > rzbdy[big_ind,1]):
                    big_ind = i

                if (rzbdy[i,1] < rzbdy[small_ind,1]):
                    small_ind = i

            a = (max(rzbdy[:,0]) - min(rzbdy[:,0])) * 0.5
            R = (max(rzbdy[:,0]) + min(rzbdy[:,0])) * 0.5

            r_up = rzbdy[big_ind,0]
            r_low = rzbdy[small_ind,0]
            
            k = (rzbdy[big_ind,1]-rzbdy[small_ind,1])/a * 0.5
            triu = (R-r_up)/a
            tril = (R-r_low)/a
            
            return k, triu, tril, R, a
    