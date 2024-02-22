from typing import Optional, List, Literal, Union
import torch, random, os
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from src.simulators.predictive_model.predict import predict_tensorboard
from src.simulators.predictive_model.FNO import FNO
from src.simulators.predictive_model.evaluate import evaluate

# distributed data parallel package
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.tensorboard import SummaryWriter

def set_random_seeds(random_seed:int = 42):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def get_distributed_loader(train_dataset : Dataset, valid_dataset : Dataset, num_replicas : int, rank : int, num_workers : int, batch_size : int = 32):
    train_sampler = DistributedSampler(train_dataset, num_replicas=num_replicas, rank = rank, shuffle = True)
    valid_sampler = DistributedSampler(valid_dataset, num_replicas=num_replicas, rank = rank, shuffle = True)

    train_loader = DataLoader(train_dataset, batch_size, sampler = train_sampler, num_workers = num_workers, pin_memory=True, drop_last = True)
    valid_loader = DataLoader(valid_dataset, batch_size, sampler = valid_sampler, num_workers = num_workers, pin_memory=True, drop_last = True)

    return train_loader, valid_loader

def train_per_epoch(
    rank : int, 
    world_size : int, 
    batch_size : Optional[int],
    model : FNO,
    train_dataset : Dataset,
    valid_dataset : Dataset,
    loss_fn : torch.nn.Module,
    max_norm_grad : Optional[float] = None,
    model_filepath : str = "./weights/distributed.pt",
    random_seed : int = 42,
    resume : bool = True,
    learning_rate : float = 1e-3
    ):
    
    device = torch.device("cuda:{}".format(rank))
    set_random_seeds(random_seed)

    model.train()
    model.to(device)
    ddp_model = DDP(model, device_ids = [device], output_device=device)

    optimizer = torch.optim.RMSprop(ddp_model.parameters(), lr = learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = 8, T_mult = 2)
    
    if not os.path.isfile(model_filepath) and dist.get_rank() == 0:
        torch.save(model.state_dict(), model_filepath)
        
    dist.barrier()
    
    # continue learning
    if resume == True:
        map_location = {"cuda:0":"cuda:{}".format(rank)}
        ddp_model.load_state_dict(torch.load(model_filepath, map_location=map_location), strict = False)
        
    train_loader, valid_loader = get_distributed_loader(train_dataset, valid_dataset, num_replicas=world_size, rank = rank, num_workers = 4, batch_size = batch_size)

    train_loss = 0

    # training process
    for batch_idx, (traj, state, control, next_state, next_control) in enumerate(train_loader):
        
        if traj.size()[0] <= 1:
            continue
        
        optimizer.zero_grad()
        output = ddp_model(traj.to(device), state.to(device), control.to(device))
        loss = loss_fn(output, next_state.to(device))
        
        if not torch.isfinite(loss):
            continue
        else:
            loss.backward()
        
        # use gradient clipping
        if max_norm_grad:
            torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_norm_grad)

        optimizer.step()

        train_loss += loss.item()

    if scheduler:
        scheduler.step()

    train_loss /= (batch_idx + 1)
    
    # validation process
    model.eval()
    valid_loss = 0
    
    for batch_idx, (traj, state, control, next_state, next_control) in enumerate(valid_loader):
        with torch.no_grad():
            
            if traj.size()[0] <= 1:
                continue
        
            output = ddp_model(traj.to(device), state.to(device), control.to(device))
            loss = loss_fn(output, next_state.to(device))
            valid_loss += loss.item()
        
    valid_loss /= (batch_idx + 1)
    
    return train_loss, valid_loss

def train_per_proc(
    rank : int, 
    world_size : int, 
    batch_size : Optional[int],
    model : FNO,
    train_dataset : Dataset,
    valid_dataset : Dataset,
    loss_fn : torch.nn.Module,
    max_norm_grad : Optional[float] = None,
    model_filepath : str = "./weights/distributed.pt",
    random_seed : int = 42,
    resume : bool = True,
    learning_rate : float = 1e-3,
    num_epoch : int = 64,
    verbose : Optional[int] = 8,
    save_best : str = "./weights/best.pt",
    tensorboard_dir : Optional[str] = None,
    test_for_check_per_epoch : Optional[DataLoader] = None,
    ):
    
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank = rank, world_size = world_size)

    train_loss_list = []
    valid_loss_list = []

    best_epoch = 0
    best_loss = torch.inf
    
    # tensorboard setting
    if dist.get_rank() == 0 and tensorboard_dir:
        writer = SummaryWriter(tensorboard_dir)
    else:
        writer = None
    
    print("# rank : {} training process proceeding...".format(rank))
    for epoch in range(num_epoch):
        train_loss, valid_loss = train_per_epoch(
            rank,
            world_size,
            batch_size,
            model,
            train_dataset,
            valid_dataset,
            loss_fn,
            max_norm_grad,
            model_filepath,
            random_seed,
            resume,
            learning_rate
        )
        
        dist.barrier()
        
        if dist.get_rank() == 0:
            train_loss_list.append(train_loss)
            valid_loss_list.append(valid_loss)

            if verbose:
                if epoch % verbose == 0:
                    print("epoch : {}, train loss : {:.3f}, valid loss : {:.3f},".format(epoch+1, train_loss, valid_loss))
                    
                    if test_for_check_per_epoch and writer is not None:
                        model.eval()
                        
                        device = torch.device("cuda:{}".format(rank))
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
            if writer is not None:
                writer.add_scalar('Loss/train', train_loss, epoch)
                writer.add_scalar('Loss/valid', valid_loss, epoch)
        
            # save the best parameters
            if best_loss > valid_loss:
                best_loss = valid_loss
                best_epoch  = epoch
                torch.save(model.state_dict(), save_best)

            # save the last parameters
            torch.save(model.state_dict(), model_filepath)
    
    if dist.get_rank() == 0:
        print("# training process finished, best loss : {:.3f}, best epoch : {}".format(best_loss, best_epoch))
    
    if writer is not None:
        writer.close()
        
    # clean up
    dist.destroy_process_group()

    return train_loss_list, valid_loss_list

def train(
    batch_size : Optional[int],
    model : FNO,
    train_dataset : Dataset,
    valid_dataset : Dataset,
    random_seed : int = 42,
    resume : bool = True,
    learning_rate : float = 1e-3,
    loss_fn = None,
    max_norm_grad : Optional[float] = None,
    model_filepath : str = "./weights/distributed.pt",
    num_epoch : int = 64,
    verbose : Optional[int] = 8,
    save_best : str = "./weights/distributed_best.pt",
    tensorboard_dir : Optional[str] = None,
    test_for_check_per_epoch : Optional[DataLoader] = None,
):
    
    world_size = dist.get_world_size()

    mp.spawn(
        train_per_proc,
        args = (world_size, batch_size, model, train_dataset,valid_dataset, loss_fn, max_norm_grad, model_filepath, random_seed, resume, learning_rate, num_epoch, verbose, save_best, tensorboard_dir, test_for_check_per_epoch),
        nprocs = world_size,
        join = True
    )
    
    print("# Distributed training process is complete")
    
def example(rank, world_size):
    dist.init_process_group("gloo", rank = rank, world_size=world_size)
    model = torch.nn.Linear(10,10).to(rank)
    ddp_model = DDP(model, device_ids = [rank])

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr = 1e-3)

    outputs = ddp_model(torch.randn(20,10).to(rank))
    labels = torch.randn(20,10).to(rank)
    
    optimizer.zero_grad()
    loss_fn(outputs, labels).backward()
    optimizer.step()

    print("rank : {} process".format(rank))

def main():
    world_size = 4
    mp.spawn(
        example,
        args = (world_size,),
        nprocs = world_size,
        join = True
    )

if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    main()