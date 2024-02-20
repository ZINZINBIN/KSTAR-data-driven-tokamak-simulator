import torch, os, argparse, warnings
import numpy as np
import pandas as pd
from src.configuration.sim_config import SimConfig
from src.utility.preprocessing import seed_everything, preparing_0D_dataset, get_range_of_output
from src.simulators.predictive_model.dataset import KSTARDataset
from src.simulators.predictive_model.DeepONet import DeepONet
from src.simulators.predictive_model.train import train
from src.simulators.predictive_model.loss import CustomLoss
from torch.utils.data import DataLoader

warnings.filterwarnings(action = 'ignore')

def parsing():
    parser = argparse.ArgumentParser(description="training predictive model")
    
    # random seed
    parser.add_argument("--random_seed", type = int, default = 42)
    
    # tag and result directory
    parser.add_argument("--tag", type = str, default = "DeepONet")
    parser.add_argument("--save_dir", type = str, default = "./results")
    
    # gpu allocation
    parser.add_argument("--gpu_num", type = int, default = 0)
      
    # training setup
    parser.add_argument("--batch_size", type = int, default = 1024)
    parser.add_argument("--lr", type = float, default = 1e-3)
    parser.add_argument("--num_epoch", type = int, default = 32)
    parser.add_argument("--pin_memory", type = bool, default = True)
    parser.add_argument("--num_workers", type = int, default = 4)
    parser.add_argument("--verbose", type = int, default = 4)
    parser.add_argument("--max_norm_grad", type = float, default = 1.0)
    parser.add_argument("--multi_step_validation", type = bool, default = False)
    parser.add_argument("--evaluate_multi_step", type = bool, default = False)
    
    # scaling
    parser.add_argument("--use_scaler", type = bool, default = True)
    parser.add_argument("--scaler", type = str, default = 'Robust', choices = ['Standard', 'Robust', 'MinMax'])
    
    # test shot num
    parser.add_argument("--test_shot_num", type = int, default = 30312)
    
    # scheduler for training
    parser.add_argument("--gamma", type = float, default = 0.95)
    parser.add_argument("--step_size", type = int, default = 8)
    
    # directory
    parser.add_argument("--root_dir", type = str, default = "./weights/")
    
    # model properties
    parser.add_argument("--traj_len", type = int, default = 20)
    parser.add_argument("--dt", type = float, default = 0.05)
    
    # Forgetting setup
    parser.add_argument("--use_forgetting", type = bool, default = False)
    parser.add_argument("--scale_forgetting", type = float, default = 0.1)
    
    args = vars(parser.parse_args())

    return args

# torch device state
print("=============== Device setup ===============")
print("torch device avaliable : ", torch.cuda.is_available())
print("torch current device : ", torch.cuda.current_device())
print("torch device num : ", torch.cuda.device_count())

# torch cuda initialize and clear cache
torch.cuda.init()
torch.cuda.empty_cache()

if __name__ == "__main__":
    
    args = parsing()
    
    # seed initialize
    seed_everything(args['random_seed'], False)
    
    # device allocation
    if(torch.cuda.device_count() >= 1):
        device = "cuda:{}".format(args['gpu_num'])
    else:
        device = 'cpu'
        
    # tag labeling
    tag = "{}_traj_{}_scaler_{}_seed_{}".format(args['tag'], args['traj_len'], args['scaler'], args['random_seed'])

    # save directory
    save_dir = os.path.join(args['save_dir'], tag)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    if not os.path.isdir("./weights"):
        os.mkdir("./weights")
        
    if not os.path.isdir("./runs"):
        os.mkdir("./runs")
    
    # configuration
    config = SimConfig()

    # load dataset for training
    print("=============== Load dataset ===============")
    df = pd.read_pickle("./dataset/KSTAR_tokamak_rl_control_data.pkl").reset_index()
    
    # columns for use
    state_col = config.feat_predicive['state']
    control_col = config.feat_predicive['control']
    
    # load dataset
    ts_train, ts_valid, ts_test, state_scaler, control_scaler = preparing_0D_dataset(df, state_col, control_col, args['scaler'], args['random_seed'])
    
    traj_len = args['traj_len']
    batch_size = args['batch_size']
    
    train_data = KSTARDataset(ts_train.copy(deep = True), traj_len, state_col, control_col, state_scaler, control_scaler, False, args['dt'])
    valid_data = KSTARDataset(ts_valid.copy(deep = True), traj_len, state_col, control_col, state_scaler, control_scaler, False, args['dt'])
    test_data = KSTARDataset(ts_test.copy(deep = True), traj_len, state_col, control_col, state_scaler, control_scaler, False, args['dt'])
   
    print("=============== Dataset information ===============")
    print("train data : ", train_data.__len__())
    print("valid data : ", valid_data.__len__())
    print("test data : ", test_data.__len__())

    train_loader = DataLoader(train_data, batch_size = batch_size, num_workers = args['num_workers'], shuffle = True, pin_memory = True)
    valid_loader = DataLoader(valid_data, batch_size = batch_size, num_workers = args['num_workers'], shuffle = True, pin_memory = True)
    test_loader = DataLoader(test_data, batch_size = batch_size, num_workers = args['num_workers'], shuffle = True, pin_memory = True)
    
    # data range
    ts_data = pd.concat([train_data.data, valid_data.data, test_data.data], axis = 1)
    range_info = get_range_of_output(ts_data, state_col, control_col)
    
    # model argument
    model = DeepONet(
        args['dt'], len(state_col), len(control_col), 
        config.model_config["branch_hidden_dim"], config.model_config["branch_layers"],
        config.model_config["trunk_hidden_dim"], config.model_config["trunk_layers"],
        config.model_config["enc_hidden_dim"], config.model_config["enc_n_channel"], config.model_config["enc_kernel_size"], 
        config.model_config["enc_depth"],config.model_config["dropout"], config.model_config["dilation_size"], 
        config.model_config["q"]
    )

    model.summary()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr = args['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = args['step_size'], gamma=args['gamma'])

    # Define weight directory
    save_best_dir = os.path.join(args['root_dir'], "{}_best.pt".format(tag))
    save_last_dir = os.path.join(args['root_dir'], "{}_last.pt".format(tag))
    tensorboard_dir = os.path.join("./runs/", "tensorboard_{}".format(tag))

    loss_fn = CustomLoss()
    
    print("=============== Training process ===============")
    print("Process : {}".format(tag))
    train_loss, valid_loss = train(
        train_loader,
        valid_loader,
        model,
        optimizer,
        scheduler,
        loss_fn,
        device,
        args['num_epoch'],
        args['verbose'],
        save_best = save_best_dir,
        save_last = save_last_dir,
        max_norm_grad = args['max_norm_grad'],
        tensorboard_dir = tensorboard_dir,
        test_for_check_per_epoch = test_loader,
        multi_step_validation = args['multi_step_validation']
    )

    model.load_state_dict(torch.load(save_best_dir))
    print("=============== Evaluation process ===============")
    