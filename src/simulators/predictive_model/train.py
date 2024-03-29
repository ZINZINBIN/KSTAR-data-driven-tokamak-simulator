from typing import Optional, List, Literal, Union
import torch
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from src.simulators.predictive_model.predict import predict_tensorboard
from src.simulators.predictive_model.DeepONet import DeepONet
from src.simulators.predictive_model.evaluate import evaluate
from torch.utils.tensorboard import SummaryWriter

def train_per_epoch(
    train_loader : DataLoader, 
    model : DeepONet,
    optimizer : torch.optim.Optimizer,
    scheduler : Optional[torch.optim.lr_scheduler._LRScheduler],
    loss_fn : torch.nn.Module,
    device : str = "cpu",
    max_norm_grad : Optional[float] = None,
    ):

    model.train()
    model.to(device)

    train_loss = 0

    for batch_idx, (traj, state, control, next_state, next_control) in enumerate(train_loader):
        
        if traj.size()[0] <= 1:
            continue
        
        optimizer.zero_grad()
        output = model(traj.to(device), state.to(device), control.to(device))
        loss = loss_fn(output, next_state.to(device))
        
        if not torch.isfinite(loss):
            print("train_per_epoch | warning : loss nan occurs")
            break
        else:
            loss.backward()
        
        # use gradient clipping
        if max_norm_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm_grad)

        optimizer.step()

        train_loss += loss.item()

    if scheduler:
        scheduler.step()

    train_loss /= (batch_idx + 1)

    return train_loss

def valid_per_epoch(
    valid_loader : DataLoader, 
    model : torch.nn.Module,
    loss_fn : torch.nn.Module,
    device : str = "cpu",
    ):

    model.eval()
    model.to(device)
    valid_loss = 0
    
    for batch_idx, (traj, state, control, next_state, next_control) in enumerate(valid_loader):
        with torch.no_grad():
            if traj.size()[0] <= 1:
                continue
        
            output = model(traj.to(device), state.to(device), control.to(device))
            loss = loss_fn(output, next_state.to(device))
            valid_loss += loss.item()
        
    valid_loss /= (batch_idx + 1)
    return valid_loss   

def train(
    train_loader : DataLoader, 
    valid_loader : DataLoader,
    model : torch.nn.Module,
    optimizer : torch.optim.Optimizer,
    scheduler : Optional[torch.optim.lr_scheduler._LRScheduler],
    loss_fn : torch.nn.Module,
    device : str = "cpu",
    num_epoch : int = 64,
    verbose : Optional[int] = 8,
    save_best : str = "./weights/best.pt",
    save_last : str = "./weights/last.pt",
    max_norm_grad : Optional[float] = None,
    tensorboard_dir : Optional[str] = None,
    test_for_check_per_epoch : Optional[DataLoader] = None,
    multi_step_validation : bool = False,
    ):

    train_loss_list = []
    valid_loss_list = []

    best_epoch = 0
    best_loss = torch.inf
    
    # tensorboard setting
    if tensorboard_dir:
        writer = SummaryWriter(tensorboard_dir)
    else:
        writer = None

    for epoch in tqdm(range(num_epoch), desc = "training process"):

        train_loss = train_per_epoch(
            train_loader, 
            model,
            optimizer,
            scheduler,
            loss_fn,
            device,
            max_norm_grad,
        )
        
        valid_loss = valid_per_epoch(
            valid_loader, 
            model,
            loss_fn,
            device,
        )
     
        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)

        if verbose:
            if epoch % verbose == 0:
                print("epoch : {}, train loss : {:.3f}, valid loss : {:.3f},".format(epoch+1, train_loss, valid_loss))
                
                if test_for_check_per_epoch:
                    model.eval()
                    test_loss, mse, rmse, mae, r2 = evaluate(test_for_check_per_epoch, model, loss_fn, device, False)
                
                    writer.add_scalars('test', 
                                        {
                                            'loss' : test_loss,
                                            'mse':mse,
                                            'rmse':rmse,
                                            'mae':mae,
                                            'r2':r2,
                                        }, 
                                        epoch + 1)
                    
                    fig = predict_tensorboard(model, test_for_check_per_epoch.dataset, device)
                    
                    # model performance check in tensorboard
                    writer.add_figure('model performance', fig, epoch+1)
                    model.train()
                
        # tensorboard recording
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/valid', valid_loss, epoch)
        
        # save the best parameters
        if best_loss > valid_loss:
            best_loss = valid_loss
            best_epoch  = epoch
            torch.save(model.state_dict(), save_best)

        # save the last parameters
        torch.save(model.state_dict(), save_last)
        
    print("training process finished, best loss : {:.3f}, best epoch : {}".format(best_loss, best_epoch))
    
    if writer:
        writer.close()

    return  train_loss_list, valid_loss_list