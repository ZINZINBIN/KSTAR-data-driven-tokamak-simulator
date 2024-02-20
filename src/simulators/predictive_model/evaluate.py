import torch
import numpy as np
from torch.utils.data import DataLoader
from src.utility.metrics import compute_metrics

def evaluate(
    test_loader : DataLoader, 
    model : torch.nn.Module,
    loss_fn : torch.nn.Module,
    device : str = "cpu",
    is_print : bool = True,
    ):

    model.eval()
    model.to(device)
    test_loss = 0
    
    pts = []
    gts = []

    for batch_idx, (traj, state, control, next_state, next_control)  in enumerate(test_loader):
        with torch.no_grad():
            
            output = model(traj.to(device), state.to(device), control.to(device))
            loss = loss_fn(output, next_state.to(device))
            
            test_loss += loss.item()
            
            pts.append(output.cpu().numpy().reshape(-1, output.size()[-1]))
            gts.append(next_state.cpu().numpy().reshape(-1, next_state.size()[-1]))
            
    test_loss /= (batch_idx + 1)
    
    pts = np.concatenate(pts, axis = 0)
    gts = np.concatenate(gts, axis = 0)
    
    mse, rmse, mae, r2 = compute_metrics(gts,pts,None,is_print)

    return test_loss, mse, rmse, mae, r2