import numpy as np
import pandas as pd
import torch
import random
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Optional, Dict, List, Union, Literal
from src.utility.compute import moving_avarage_smoothing

class KSTARDataset(Dataset):
    def __init__(
        self, 
        data : pd.DataFrame, 
        traj_len : int,
        state_col : List[str],
        control_col : List[str],
        state_scaler = None,
        control_scaler = None,
        multi_step : bool = False,
        dt:float = 0.01,
        ):
        
        self.data = data
        self.traj_len = traj_len
        self.state_col = state_col
        self.control_col = control_col
        
        self.state_scaler = state_scaler
        self.control_scaler = control_scaler
        
        self.multi_step = multi_step
        self.dt = dt
        
        # indice for getitem method
        self.input_indices = []
        self.target_indices = []
        
        # experiment list
        self.shot_list = np.unique(self.data.shot.values).tolist()
        
        # preprocessing
        self.preprocessing()
        
        # data - label index generation
        self._generate_index()
        
    def preprocessing(self):
        # ignore shot which have too many nan values
        shot_ignore = []
        for shot in tqdm(self.shot_list, desc = '# Extract null data'):
            df_shot = self.data[self.data.shot == shot]
            null_check = df_shot[self.state_col + self.control_col].isna().sum()
            
            for c in null_check:
                if c > 0.5 * len(df_shot):
                    shot_ignore.append(shot)
                    break
        
        # update shot list with ignoring the null data
        shot_list_new = [shot_num for shot_num in self.shot_list if shot_num not in shot_ignore]
        self.shot_list = shot_list_new
                 
        # scaling
        if self.state_scaler:
            self.data[self.state_col] = self.state_scaler.transform(self.data[self.state_col])
            
        if self.control_scaler:
            self.data[self.control_col] = self.control_scaler.transform(self.data[self.control_col])

    def _generate_index(self):
        
        # Index generation
        for shot in tqdm(self.shot_list, desc = "# Dataset indice generation"):
            
            df_shot = self.data[self.data.shot == shot]
            
            # Find flattop regime
            dvl = df_shot['\\RC03'].shift(1).fillna(method = 'bfill').values.reshape(-1,)
            dvr = df_shot['\\RC03'].shift(-1).fillna(method = 'ffill').values.reshape(-1,)

            idx_tftsrt = np.argmax(moving_avarage_smoothing(dvr-dvl, 12, 'center'))
            idx_tend = np.argmin(dvr-dvl)
            
            tftsrt = df_shot.time.iloc[idx_tftsrt]
            tend = df_shot.time.iloc[idx_tend]
            
            if tend - tftsrt < 1.0:
                continue
            
            # indexing
            input_indices = []
            
            idx = 0
            idx_last = len(df_shot.index) - 1
            
            if idx_last < 200:
                continue

            while(idx < idx_last):
                row = df_shot.iloc[idx]
                t = row['time']
                
                if t < tftsrt + self.traj_len * self.dt + 0.1:
                    idx += 1
                    continue
                
                input_indx = df_shot.index.values[idx]
                input_indices.append(input_indx)
                
                if t > tend:
                    break
                else:
                    idx += 1

            self.input_indices.extend(input_indices)

    def __getitem__(self, idx:int):
        
        input_idx = self.input_indices[idx]
        traj = self.data.loc[input_idx-self.traj_len:input_idx-1, self.state_col + self.control_col].values
        state = self.data.loc[input_idx, self.state_col].values
        control = self.data.loc[input_idx, self.control_col].values
        
        next_state = self.data.loc[input_idx+1, self.state_col].values
        next_control = self.data.loc[input_idx+1, self.control_col].values

        traj = torch.from_numpy(traj).float()
        state = torch.from_numpy(state).float()
        control = torch.from_numpy(control).float()
        
        next_state = torch.from_numpy(next_state).float()
        next_control = torch.from_numpy(next_control).float()
        
        return traj, state, control, next_state, next_control
                
    def __len__(self):
        return len(self.input_indices)