import torch
import pandas as pd
import numpy as np
import random
from typing import Optional, List
import matplotlib.pyplot as plt
from src.simulators.predictive_model.dataset import KSTARDataset
from src.simulators.predictive_model.DeepONet import DeepONet
from src.configuration.sim_config import *

# Tensorboard - validation for model prediction
def predict_tensorboard(
    model : DeepONet,
    test_data : KSTARDataset,
    device : str = 'cpu',
    ):
    
    shot_list = np.unique(test_data.data.shot.values)
    
    traj_len = test_data.traj_len
    state_col = test_data.state_col
    control_col = test_data.control_col
    
    is_shot_valid = False
    
    while(not is_shot_valid):
        
        shot_num = random.choice(shot_list)
        
        df_shot_origin = test_data.data[test_data.data.shot == shot_num].reset_index(drop = True)
        df_shot = df_shot_origin.copy(deep = True)
        
        idx_end = len(df_shot) - 1
        
        if idx_end < 30 + traj_len * 4:
            is_shot_valid = False
        else:
            is_shot_valid = True
    
    model.to(device)
    model.eval()
    
    time_x = df_shot['time']
    
    predictions = []
    
    idx_srt = 30 + traj_len
    idx = idx_srt
    idx_max = len(df_shot) - 1
    
    model.to(device)
    model.eval()
    
    traj = None
    state = None
    control = None
    next_state = None
    next_control = None
    
    while(idx < idx_max): 
        if idx == idx_srt:
            traj = torch.from_numpy(df_shot.loc[idx-traj_len:idx-1, state_col + control_col].values).unsqueeze(0).float()
            state = torch.from_numpy(df_shot.loc[idx, state_col].values).unsqueeze(0).float()
            control = torch.from_numpy(df_shot.loc[idx, control_col].values).unsqueeze(0).float()
            next_control = torch.from_numpy(df_shot.loc[idx + 1, control_col].values).unsqueeze(0).float()
            
        else:
            next_control = torch.from_numpy(df_shot.loc[idx + 1, control_col].values).unsqueeze(0).float()
                
        # next_state = model.predict(traj.to(device), state.to(device), control.to(device), next_control.to(device))
        next_state = model(traj.to(device), state.to(device), control.to(device))
        
        idx += 1
        
        # update previous state
        traj = torch.concat([traj, torch.concat([state.cpu(), control.cpu()], dim = 1).unsqueeze(1)], dim = 1)[:,1:,:]
        state = next_state.cpu()
        control = next_control.cpu()
        
        # update prediction value
        prediction = next_state.detach().cpu().numpy()
        predictions.append(prediction)
            
    predictions = np.concatenate(predictions, axis = 0)
    
    time_x = time_x.loc[idx_srt+1:idx_srt+len(predictions)].values
    actual = df_shot.loc[idx_srt+1:idx_srt+len(predictions), state_col].values
    
    fig, axes = plt.subplots(len(state_col), 1, figsize = (10,6), sharex=True, facecolor = 'white')
    plt.suptitle("shot : {}-running-process".format(shot_num))
    
    if test_data.state_scaler:
        predictions = test_data.state_scaler.inverse_transform(predictions)
        actual = test_data.state_scaler.inverse_transform(actual)
    
    for i, (ax, col) in enumerate(zip(axes.ravel(), state_col)):
        ax.plot(time_x, actual[:,i], 'k', label = "actual")
        ax.plot(time_x, predictions[:,i], 'b', label = "pred")
        ax.set_ylabel(col)
        ax.legend(loc = "upper right")

    fig.tight_layout()
    return fig