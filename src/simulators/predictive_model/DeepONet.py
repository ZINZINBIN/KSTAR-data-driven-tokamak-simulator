import torch
import torch.nn as nn
from typing import Optional
from src.simulators.predictive_model.tcn import TCN, calc_dilation
from pytorch_model_summary import summary

def gradient(u:torch.Tensor, x:torch.Tensor):
    ux = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    return ux

class FNN(nn.Module):
    def __init__(self, input_dim : int, output_dim:int, layers:int = 2, hidden_dim:int = 64, softmax : bool = False):
        super(FNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = layers
        self.hidden_dim = hidden_dim
        self.softmax = softmax
        
        self.modus = self.create_modules()
        
    def create_modules(self):
        modules = nn.ModuleDict()
        
        if self.layers > 1:
            modules['LinM{}'.format(1)] = nn.Linear(self.input_dim, self.hidden_dim)
            modules['NonM{}'.format(1)] = nn.LeakyReLU(0.01)
                
            for i in range(2, self.layers):
                modules['LinM{}'.format(i)] = nn.Linear(self.hidden_dim, self.hidden_dim)
                modules['NonM{}'.format(i)] = nn.LeakyReLU(0.01)
            modules['LinMout'] = nn.Linear(self.hidden_dim, self.output_dim)
            
        else:
            modules['LinMout'] = nn.Linear(self.input_dim, self.output_dim)
        
        return modules
    
    def initialize_weights(self):
        
        for i in range(1, self.layers):
            nn.init.xavier_normal_(self.modus['LinM{}'.format(i)].weight)
            nn.init.constant_(self.modus['LinM{}'.format(i)].bias, 0)
            
        nn.init.xavier_normal_(self.modus['LinMout'].weight)
        nn.init.constant_(self.modus['LinMout'].bias, 0)
        
    def forward(self, x:torch.Tensor):
        
        for i in range(1, self.layers):
            LinM = self.modus['LinM{}'.format(i)]
            NonM = self.modus['NonM{}'.format(i)]
            x = NonM(LinM(x))
            
        x = self.modus['LinMout'](x)
        
        if self.softmax:
            x = nn.functional.softmax(x, dim=-1)
            
        return x
    
class DeepONet(nn.Module):
    def __init__(
        self, 
        dt : float, 
        state_dim:int, 
        control_dim:int, 
        branch_hidden_dim:int, 
        branch_layers:int, 
        trunk_hidden_dim:int, 
        trunk_layers:int,
        enc_hidden_dim:int = 64, 
        enc_n_channel:int = 64,
        enc_kernel_size:int=3,
        enc_depth:int = 4,
        dropout:float = 0.25,
        dilation_size:int = 2,
        q:int = 101,
        ):
        
        super(DeepONet, self).__init__()
        self.branch_network = self.create_network(state_dim + control_dim + enc_hidden_dim, state_dim * q, branch_hidden_dim, branch_layers)
        self.trunk_network = self.create_network(1, state_dim * q, trunk_hidden_dim, trunk_layers)
        self.enc_network = self.create_encoder_network(state_dim + control_dim, enc_hidden_dim, enc_n_channel, enc_kernel_size, dropout, dilation_size, enc_depth)
        
        self.dt = dt
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.branch_hidden_dim = branch_hidden_dim
        self.branch_layers = branch_layers
        self.trunk_hidden_dim = trunk_hidden_dim
        self.trunk_layers = trunk_layers  
        self.enc_hidden_dim = enc_hidden_dim 
        self.enc_n_channel = enc_n_channel
        self.enc_kernel_size = enc_kernel_size
        self.enc_depth = enc_depth
        self.dropout = dropout
        self.dilation_size = dilation_size
        self.q = q
    
        self.params = nn.ParameterDict()
        self.params['bias'] = nn.Parameter(torch.zeros([1]), requires_grad = True)
        
    def forward(self, traj: torch.Tensor, state : torch.Tensor, control:torch.Tensor, dt:Optional[float] = None):
        
        if traj.size()[2] == self.state_dim + self.control_dim:
            traj = traj.permute(0,2,1)
        
        traj = self.enc_network(traj)
        
        if dt is None:
            dt = nn.Parameter(torch.ones([1]).unsqueeze(0).repeat((len(traj),1)) * self.dt, requires_grad = True)
    
        x_branch = self.compute_branch_network(state, control, traj).view(state.size()[0],state.size()[1], self.q)
        x_trunk = self.compute_trunk_network(dt.to(state.device)).view(state.size()[0],state.size()[1], self.q)
        output = self.compute_local_operation(x_branch, x_trunk)
        return output
    
    def predict(self, traj: torch.Tensor, state : torch.Tensor, control:torch.Tensor, next_control:torch.Tensor):
        
        dt1 = nn.Parameter(torch.ones([1]).unsqueeze(0).repeat((len(traj),1)) * 0, requires_grad = True).to(traj.device)
        dt2 = nn.Parameter(torch.ones([1]).unsqueeze(0).repeat((len(traj),1)) * self.dt, requires_grad = True).to(traj.device)
        
        k1 = gradient(self.forward(traj, state, control, dt1), dt1)
        k2 = gradient(self.forward(traj, state, next_control, dt2), dt2)
        next_state = state + 0.5 * self.dt * (k1 + k2)
        return next_state
    
    def create_network(self, input_dim, output_dim, hidden_dim, layers):
        network = FNN(input_dim, output_dim, layers, hidden_dim, False)
        return network

    def compute_branch_network(self, state:torch.Tensor, control:torch.Tensor, traj:torch.Tensor):
        x = torch.concat([state, control, traj], dim = 1)
        x_branch = self.branch_network(x)
        return x_branch
    
    def compute_trunk_network(self, dt:torch.Tensor):
        return self.trunk_network(dt)
    
    def compute_local_operation(self, x_branch:torch.Tensor, x_trunk : torch.Tensor):
        return torch.sum(x_branch * x_trunk, dim = -1, keepdim = False) + self.params['bias']
    
    def create_encoder_network(self, input_dim:int, hidden_dim:int, num_channel:int, kernel_size:int, dropout:float, dilation_size:int, depth:int):
        num_channels = [num_channel] * depth
        dilation_size = calc_dilation(kernel_size, dilation_size, depth, 128)
        return TCN(input_dim, hidden_dim, num_channels, kernel_size, dropout, dilation_size)
    
    def summary(self):
        traj = torch.zeros((1, 100, self.state_dim + self.control_dim))
        state = torch.zeros((1, self.state_dim ))
        control = torch.zeros((1, self.control_dim))
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
    branch_hidden_dim = 32
    branch_layers = 2
    trunk_hidden_dim = 32
    trunk_layers = 2
    
    time_length = 100
    
    enc_kernel_size = 3
    enc_hidden_dim = 32
    enc_n_channel = 32
    enc_depth = 4
    dropout = 0.2
    dilation_size = 2
    dt = 0.01
    
    # model
    network = DeepONet(dt, state_dim, control_dim, branch_hidden_dim, branch_layers, trunk_hidden_dim, trunk_layers, enc_hidden_dim, enc_n_channel, enc_kernel_size, enc_depth, dropout, dilation_size)
    network.summary()
    
    network.to(device)
    
    # input
    batch_size = 32
    state = torch.zeros((batch_size, state_dim))
    control = torch.zeros((batch_size, control_dim))
    traj = torch.zeros((batch_size, time_length, state_dim + control_dim))
    
    output = network(traj.to(device), state.to(device), control.to(device))
    print("output size: ", output.size())
    
    # RK prediction
    next_control = torch.zeros((batch_size, control_dim))
    pred = network.predict(traj.to(device), state.to(device), control.to(device), next_control.to(device))
    print("RK prediction size: ", pred.size())
    