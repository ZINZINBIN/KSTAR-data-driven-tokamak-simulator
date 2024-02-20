import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# define operator: (x,y) coordinate
class LaplaceOperator:
    def __init__(self, dx:float, dy:float, nx:int,ny:int):
        self.dx = dx
        self.dy = dy
        self.nx = nx
        self.ny = ny
        
        self.L = torch.zeros((nx * ny, nx * ny), dtype = torch.float32)
        
        idx_dx = ny
        idx_dy = 1
        
        for i in range(1,nx-1):
            for j in range(1,ny-1):
                
                idx = ny * i + j

                self.L[idx, idx] = (-1) * 2.0 * (1 / dx ** 2 + 1 / dy ** 2)
                self.L[idx - idx_dx, idx] = 1.0 / dx ** 2
                self.L[idx + idx_dx, idx] = 1.0 / dx ** 2
                
                self.L[idx, idx - idx_dy] = 1.0 / dy ** 2
                self.L[idx, idx + idx_dy] = 1.0 / dy ** 2

    def __call__(self, u:torch.Tensor):
        return torch.matmul(self.L, u.view(-1, self.nx * self.ny)).view(self.nx, self.ny)
    
class ElipticOperator:
    def __init__(self, dx:float, dy:float, nx:int,ny:int):
        self.dx = dx
        self.dy = dy
        self.nx = nx
        self.ny = ny
        
        self.L = torch.zeros((nx * ny, nx * ny), dtype = torch.float32)
        
        idx_dx = ny
        idx_dy = 1
        
        for i in range(1,nx-1):
            for j in range(1,ny-1):
                
                idx = ny * i + j

                self.L[idx, idx] = (-1) * 2.0 * (1 / dx ** 2 + 1 / dy ** 2)
                self.L[idx - idx_dx, idx] = 1.0 / dx ** 2
                self.L[idx + idx_dx, idx] = 1.0 / dx ** 2
                
                self.L[idx, idx - idx_dy] = 1.0 / dy ** 2
                self.L[idx, idx + idx_dy] = 1.0 / dy ** 2
        
    def __call__(self, u:torch.Tensor):
        return torch.matmul(self.L, u.view(-1, self.nx * self.ny)).view(self.nx, self.ny)
    
def external_function(u:torch.Tensor):
    return (torch.pi ** 2 / 2 - 2 - u) * u
    
def exact_solution(x,y,t):
    return 2 * np.exp(-2 * t) * np.sin(np.pi * x / 2) * np.cos(np.pi * y / 2)

class CANNdataset(Dataset):
    def __init__(self, u_exact):
        
        self.u_exact = u_exact
        self.nx = u_exact.shape[0]
        self.ny = u_exact.shape[1]
        self.nt = u_exact.shape[2]
        
        self.indices = []
        
        for t in range(0, self.nt-1):
            for i in range(0,self.nx):
                for j in range(0, self.ny):
                    
                    xl = i-1
                    xr = i+1
                    yl = j-1
                    yr = j+1
                    
                    if i == 0:
                        xl = self.nx - 1
                    elif i == self.nx-1:
                        xr = 0
                        
                    if j == 0:
                        yl = self.ny - 1
                    elif j == self.ny - 1:
                        yr = 0
                    
                    indice = [(xl,j,t),(xr,j,t),(i,j,t),(i,yl,t),(i,yr,t),(i,j,t+1)]
                    self.indices.append(indice)
    
    def __getitem__(self, idx: int):
        indice = self.indices[idx]
        uij = self.call_component(indice[2])
        vij = np.array([self.call_component(indice[0]),self.call_component(indice[1]), self.call_component(indice[2]), self.call_component(indice[3]),self.call_component(indice[4])])
        uij_next = self.call_component(indice[5])
        return uij, vij, uij_next
        
    def call_component(self, indice):
        return self.u_exact[indice[0],indice[1],indice[2]]
    
    def __len__(self):
        return len(self.indices)

class Approximator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(5, 16),
            nn.Tanh(),
            nn.Linear(16,16),
            nn.Tanh(),
            nn.Linear(16,1),
        )
    
    def forward(self, v:torch.Tensor):
        dv = self.layers(v.float())
        return dv
    
    def predict(self, u:torch.Tensor, v:torch.Tensor):
        dv = self.forward(v)
        
        if u.ndim == 1:
            u = u.unsqueeze(1)
        
        return u.float() + dv.float()
    
def train_per_epoch(
    model:Approximator,
    dataloader:DataLoader,
    loss_fn:nn.Module,
    device:str,
    optimizer:torch.optim.Optimizer,
):
    
    model.train()
    model.to(device)
    train_loss = 0
    
    for batch_idx, (uij, vij, uij_next) in enumerate(dataloader):
        
        optimizer.zero_grad()
        output = model.predict(uij.float().to(device), vij.float().to(device))

        loss = loss_fn(output, uij_next.unsqueeze(1).float().to(device))
        
        if not torch.isfinite(loss):
            break
        else:
            loss.backward()
        
        optimizer.step()
        train_loss += loss.item()

    # train_loss /= (batch_idx + 1)
    return train_loss

def predict_one_step(model:Approximator, u_init:np.ndarray, device:str):
    
    nx = u_init.shape[0]
    ny = u_init.shape[1]
    u_next = np.zeros((nx,ny))
    
    indices = []
    
    for i in range(0,nx):
        for j in range(0,ny):
            
            xl = i-1
            xr = i+1
            yl = j-1
            yr = j+1
            
            if i == 0:
                xl = nx-1
            elif i == nx-1:
                xr = 0
                
            if j == 0:
                yl = ny-1
            elif j == ny-1:
                yr = 0
            
            indice = [(xl,j),(xr,j),(i,j),(i,yl),(i,yr)]
            indices.append(indice)
    
    model.eval()
    
    for indice in indices:
        
        uij = np.array(u_init[indice[2]])
        vij = np.array([u_init[indice[0]], u_init[indice[1]], u_init[indice[2]], u_init[indice[3]], u_init[indice[4]]])
        
        uij = torch.from_numpy(uij).unsqueeze(0).float()
        vij = torch.from_numpy(vij).unsqueeze(0).float()
        
        with torch.no_grad():
            uij_next = model.predict(uij.to(device), vij.to(device)).squeeze(0).detach().cpu().numpy()
            u_next[indices[2]] = uij_next

    return u_next
    
# torch device state
print("=============== Device setup ===============")
print("torch device avaliable : ", torch.cuda.is_available())
print("torch current device : ", torch.cuda.current_device())
print("torch device num : ", torch.cuda.device_count())

# torch cuda initialize and clear cache
torch.cuda.init()
torch.cuda.empty_cache()

if __name__ == "__main__":
    
    # device allocation
    if(torch.cuda.device_count() >= 1):
        device = "cuda:{}".format(0)
    else:
        device = 'cpu'
    
    n = 40
    xlin = np.linspace(-1,1,n)
    y_mesh, x_mesh = np.meshgrid(xlin, xlin)
    
    X_mesh = torch.from_numpy(x_mesh)
    Y_mesh = torch.from_numpy(y_mesh)
    
    u_init = 2 * torch.sin(X_mesh * torch.pi / 2) * torch.cos(Y_mesh * torch.pi / 2)
    
    ti = 0
    tf = 2.0
    nt = 100
    tlin = np.linspace(ti,tf,nt)
    
    u_exact = np.zeros((n,n,nt))
    
    for idx, t in enumerate(tlin):
        u_exact[:,:,idx] = exact_solution(X_mesh, Y_mesh, t)
    
    train_data = CANNdataset(u_exact[:,:,:])
    train_loader = DataLoader(train_data, batch_size = 100, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    
    model = Approximator()
    model.to(device)
    
    loss_fn = torch.nn.MSELoss(reduction = 'sum')

    num_epoch = 1024
    verbose = 4
    optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3)
    
    for epoch in tqdm(range(num_epoch), 'training process'):
        train_loss = train_per_epoch(model, train_loader, loss_fn, device, optimizer)
        
        if epoch % verbose == 0:
            print('epoch:{:04d} | train loss : {:.5f}'.format(epoch, train_loss))
            
    # Exact solution
    fig, ax = plt.subplots(subplot_kw=dict(projection = '3d'))
    surf = ax.plot_surface(x_mesh, y_mesh, u_exact[:,:,-1])
    fig.tight_layout()
    plt.savefig("./results/test-exact.png")
    
    # Prediction solution
    
    u_next = None
    
    for idx, t in enumerate(tqdm(tlin)):
        
        if idx == 0:
            u_next = predict_one_step(model, u_init, device)
        
        else:
            u_next = predict_one_step(model, u_next, device)
            
    fig, ax = plt.subplots(subplot_kw=dict(projection = '3d'))
    surf = ax.plot_surface(x_mesh, y_mesh, u_next)
    fig.tight_layout()
    plt.savefig("./results/test-prediction.png")