from sklearn.model_selection import train_test_split
import random, torch, os
import numpy as np
import pandas as pd
import torch.backends.cudnn as cudnn
from typing import Literal, Optional, List
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

# For reproduction
def seed_everything(seed : int = 42, deterministic : bool = False):
    
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False

def preparing_0D_dataset(
    df : pd.DataFrame,
    state_col : List,
    control_col : List,
    scaler : Literal['Robust', 'Standard', 'MinMax'] = 'Robust',
    random_seed : int = 42
    ):

    total_col = state_col + control_col

    # float type
    for col in total_col:
        df[col] = df[col].astype(np.float32)
    
    # shot sampling
    shot_list = np.unique(df.shot.values)
    print("# of shot : {}".format(len(shot_list)))
    
    # train / valid / test data split
    shot_train, shot_test = train_test_split(shot_list, test_size = 0.2, random_state = random_seed)
    shot_train, shot_valid = train_test_split(shot_train, test_size = 0.25, random_state = random_seed)
    
    df_train = df[df.shot.isin(shot_train)]
    df_valid = df[df.shot.isin(shot_valid)]
    df_test = df[df.shot.isin(shot_test)]
    
    print("# of train shot : {}".format(len(shot_train)))
    print("# of valid shot : {}".format(len(shot_valid)))
    print("# of test shot : {}".format(len(shot_test)))
    
    if scaler == 'Standard':
        state_scaler = StandardScaler()
        control_scaler = StandardScaler()
    elif scaler == 'Robust':
        state_scaler = RobustScaler()
        control_scaler = RobustScaler()
    elif scaler == 'MinMax':
        state_scaler = MinMaxScaler()
        control_scaler = MinMaxScaler()
  
    # scaler training
    print("# Fitting scaler process..")
    state_scaler.fit(df_train[state_col].values)
    control_scaler.fit(df_train[control_col].values)
    print("# Fitting scaler process complete")
        
    return df_train, df_valid, df_test, state_scaler, control_scaler

# get range of each output
def get_range_of_output(df : pd.DataFrame, state_col : List, control_col:List):
    
    range_info = {
        "state":{},
        "control":{},
    }

    for col in state_col:
        min_val = df[col].min()
        max_val = df[col].max()
        range_info['state'][col] = [min_val, max_val]
    
    for col in control_col:
        min_val = df[col].min()
        max_val = df[col].max()
        range_info['control'][col] = [min_val, max_val]
    
    return range_info