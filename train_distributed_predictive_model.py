import torch, os, argparse, warnings, random
import pandas as pd
from src.configuration.sim_config import SimConfig
from src.utility.preprocessing import seed_everything, preparing_0D_dataset, get_range_of_output
from src.simulators.predictive_model.dataset import KSTARDataset
from src.simulators.predictive_model.FNO import FNO
from src.simulators.predictive_model.train_distributed import train
from src.simulators.predictive_model.loss import CustomLoss
from torch.utils.data import DataLoader
import torch.distributed as dist

warnings.filterwarnings(action = 'ignore')

def parsing():
    parser = argparse.ArgumentParser(description="training predictive model with distributed data parallel method")
    
    # random seed
    parser.add_argument("--random_seed", type = int, default = 42)
    
    # tag and result directory
    parser.add_argument("--tag", type = str, default = "FNO")
    parser.add_argument("--save_dir", type = str, default = "./results")
      
    # training setup
    parser.add_argument("--batch_size", type = int, default = 1024)
    parser.add_argument("--lr", type = float, default = 2e-4)
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

# torch cuda initialize and clear cache
torch.cuda.init()
torch.cuda.empty_cache()

if __name__ == "__main__":
    
    # initialize process group
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "{:05d}".format(random.randint(10000,65535))
    
    # torch device state
    print("=============== Device setup ===============")
    print("torch device avaliable : ", torch.cuda.is_available())
    print("torch current device : ", torch.cuda.current_device())
    print("torch device num : ", torch.cuda.device_count())

    args = parsing()
    
    # seed initialize
    seed_everything(args['random_seed'], False)
    
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
    ts_train, ts_valid, ts_test, state_scaler, control_scaler = preparing_0D_dataset(df, state_col, control_col, args['scaler'], args['random_seed'], False)
    
    traj_len = args['traj_len']
    batch_size = args['batch_size']
    
    train_data = KSTARDataset(ts_train.copy(deep = True), traj_len, state_col, control_col, state_scaler, control_scaler, False, args['dt'], True)
    valid_data = KSTARDataset(ts_valid.copy(deep = True), traj_len, state_col, control_col, state_scaler, control_scaler, False, args['dt'], True)
    test_data = KSTARDataset(ts_test.copy(deep = True), traj_len, state_col, control_col, state_scaler, control_scaler, False, args['dt'], True)
    
    # test during training process
    test_loader = DataLoader(test_data, batch_size = batch_size, num_workers = args['num_workers'], shuffle = True, pin_memory = True)
   
    # if dist.get_rank() == 0:
    print("========== Dataset information ==========")
    print("train data : ", train_data.__len__())
    print("valid data : ", valid_data.__len__())
    print("test data : ", test_data.__len__())
    
    # data range
    ts_data = pd.concat([train_data.data, valid_data.data, test_data.data], axis = 1)
    range_info = get_range_of_output(ts_data, state_col, control_col)
    
    # model argument
    model = FNO(
        state_dim = len(state_col),
        control_dim = len(control_col),
        modes = 4,
        width = 64,
    )
    
    model.summary()

    # Define weight directory
    save_best_dir = os.path.join(args['root_dir'], "{}_best.pt".format(tag))
    save_last_dir = os.path.join(args['root_dir'], "{}_last.pt".format(tag))
    tensorboard_dir = os.path.join("./runs/", "tensorboard_{}".format(tag))

    loss_fn = CustomLoss()
    
    print("=============== Training process ===============")
    print("Process : {}".format(tag))
    
    train(
        batch_size = args['batch_size'],
        model = model,
        train_dataset = train_data,
        valid_dataset = valid_data,
        random_seed = args['random_seed'],
        resume = False,
        learning_rate  = args['lr'],
        loss_fn = loss_fn,
        max_norm_grad = args['max_norm_grad'],
        model_filepath = "./weights/distributed.pt",
        num_epoch = args['num_epoch'],
        verbose = args['verbose'],
        save_best = save_best_dir,
        tensorboard_dir = tensorboard_dir,
        test_for_check_per_epoch = test_loader,
    )
