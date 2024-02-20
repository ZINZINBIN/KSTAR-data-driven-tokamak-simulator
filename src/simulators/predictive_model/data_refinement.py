import pandas as pd
import numpy as np
import warnings
from tqdm.auto import tqdm
from typing import Literal
from src.configuration.sim_config import *
from src.utility.compute import compute_dWdt, compute_tau, compute_tau_e

warnings.filterwarnings(action = 'ignore')

def bound(x, value : float):
    return x if abs(x) < value else value * x / abs(x)

def positive(x):
    return x if x >0 else 0

def moving_avarage_smoothing(X:np.array,k:int, method:Literal['backward', 'center'] = 'backward'):
    S = np.zeros(X.shape[0])
    if method == 'backward':
        for t in range(X.shape[0]):
            if t < k:
                S[t] = np.mean(X[:t+1])
            else:
                S[t] = np.sum(X[t-k:t])/k
    else:
        hw = k//2
        for t in range(X.shape[0]):
            if t < hw:
                S[t] = np.mean(X[:t+1])
            elif t  < X.shape[0] - hw:
                S[t] = np.mean(X[t-hw:t+hw])
            else:
                S[t] = np.mean(X[t-hw:])
    return S

if __name__ == "__main__":
    
    config = SimConfig()
    
    df = pd.read_pickle("./dataset/KSTAR_tokamak_rl_control_data_orig.pkl")
    feat_cols = config.EFIT + config.PCPF + config.ECH + config.NBH + config.LV + config.DL + config.RC + config.GAS + config.TCI + config.BOL
    
    # Handling inf and null value
    print('\n',"="*50)
    print("# process: covert inf to nan & remove nan value")
    df.replace([np.inf, -np.inf], np.nan)
    
    print("EFIT - NaN case: ", sum(df[config.EFIT].isna().sum().values))
    print("PCPF - NaN case: ", sum(df[config.PCPF].isna().sum().values))
    print("ECH - NaN case: ", sum(df[config.ECH].isna().sum().values))
    print("NBH - NaN case: ", sum(df[config.NBH].isna().sum().values))
    print("LV - NaN case: ", sum(df[config.LV].isna().sum().values))
    print("DL - NaN case: ", sum(df[config.DL].isna().sum().values))
    print("RC - NaN case: ", sum(df[config.RC].isna().sum().values))
    print("GAS - NaN case: ", sum(df[config.GAS].isna().sum().values))
    print("TCI - NaN case: ", sum(df[config.TCI].isna().sum().values))
    print("BOL - NaN case: ", sum(df[config.BOL].isna().sum().values))
    
    print("\n# remove nan values")
    df[config.ECH] = df[config.ECH].fillna(0)
    df[config.NBH] = df[config.NBH].fillna(0)
    df[config.GAS] = df[config.GAS].fillna(0)
    df[config.BOL] = df[config.BOL].fillna(0)
    
    # Selection for valid shot: 1st - loop voltage and diamagnetic loop
    print('\n',"="*50)
    print("# process: valid shot selection - 1 and 2 steps")
    remove_shot_list = df[df['\\LV01'].isna() == True].shot.unique()
    shot_list = [shot for shot in df.shot.unique() if shot not in remove_shot_list]
    
    # Selection for valid shot: 2nd - Bolometer signals
    shot_list = [shot for shot in shot_list if shot not in df[df['\\ax3_bolo02:FOO'].isna() == True].shot.unique()]
    
    # Selection for valid shot: 3rd - diamagnetic signals
    shot_list = [shot for shot in shot_list if shot not in df[df['\\BETAP_DLM03'].isna() == True].shot.unique()]

    df = df[df.shot.isin(shot_list)]
    
    print("# process: checking for remained nan values")
    print("EFIT - NaN case: ", sum(df[config.EFIT].isna().sum().values))
    print("PCPF - NaN case: ", sum(df[config.PCPF].isna().sum().values))
    print("ECH - NaN case: ", sum(df[config.ECH].isna().sum().values))
    print("NBH - NaN case: ", sum(df[config.NBH].isna().sum().values))
    print("LV - NaN case: ", sum(df[config.LV].isna().sum().values))
    print("DL - NaN case: ", sum(df[config.DL].isna().sum().values))
    print("RC - NaN case: ", sum(df[config.RC].isna().sum().values))
    print("GAS - NaN case: ", sum(df[config.GAS].isna().sum().values))
    print("TCI - NaN case: ", sum(df[config.TCI].isna().sum().values))
    print("BOL - NaN case: ", sum(df[config.BOL].isna().sum().values))
    
    print("\n# Overall statistics of KSTAR dataset")
    df[feat_cols].describe()
    
    # scaling process
    print('\n',"="*50)
    print("# process: re-scaling values")
    print("# EFIT data re-scaling...")
    df['\\ipmhd'] = df['\\ipmhd'].apply(lambda x : x / 1e6) # MA
    
    # Bound for outliers
    for col in config.EFIT:
        df[col]= df[col].apply(lambda x : bound(x,1e2))
    
    df['\\q95'] = df['\\q95'].apply(lambda x : positive(x))
    df['\\q95'] = df['\\q95'].apply(lambda x : bound(x,1e1))
    
    df['\\betap'] = df['\\betap'].apply(lambda x : positive(x))
    df['\\betap'] = df['\\betap'].apply(lambda x : bound(x,1e1))
    
    df['\\betan'] = df['\\betan'].apply(lambda x : positive(x))
    df['\\betan'] = df['\\betan'].apply(lambda x : bound(x,1e1))
    
    df['\\li'] = df['\\li'].apply(lambda x : positive(x))
    df['\\li'] = df['\\li'].apply(lambda x : bound(x,1e1))
    
    df['\\drsep'] = df['\\drsep'].apply(lambda x : bound(x,1.0))
        
    print('# Diagnostic data re-scaling...')     
    df['\\WTOT_DLM03'] = df['\\WTOT_DLM03'].apply(lambda x : x / 1e3) # kJ or keV
    df['\\WTOT_DLM03'] = df['\\WTOT_DLM03'].apply(lambda x : positive(x))      
    
    df['\\BETAP_DLM03']= df['\\BETAP_DLM03'].apply(lambda x : positive(x))
    df['\\BETAP_DLM03']= df['\\BETAP_DLM03'].apply(lambda x : bound(x, 1e1))
    
    df['\\DMF_DLM03'] = df['\\DMF_DLM03'].apply(lambda x : bound(x, 1e1))
            
    df[config.RC] = df[config.RC].apply(lambda x : x / 1e6) # MV / Mamp
    
    df[config.PCPF] = df[config.PCPF].apply(lambda x : x / 1e3) # kA
    
    for col in config.TCI:
        df[col] = df[col].apply(lambda x : bound(x, 1e1))
        df[col] = df[col].apply(lambda x : positive(x))
        
    df[config.ECH] = df[config.ECH].apply(lambda x : x / 1e3) # kJ / kV / keV
    
    for col in config.ECH:
        df[col] = df[col].apply(lambda x : positive(x))
    
    for col in config.NBH:
        df[col] = df[col].apply(lambda x : positive(x))
        
    df[feat_cols].describe()
    
    # Selection for valid shot: 3rd: Flattop regime
    print('\n',"="*50)
    print("# process: valid shot selection - 3rd steps")
    
    # middle value => interpolation (just in case)
    df[feat_cols] = df[feat_cols].interpolate(method = 'linear', limit_area = 'inside')

    # shot selection again (short length -> flattop region)
    shot_ignore = []
    shot_list = df.shot.unique()
    
    for shot in tqdm(shot_list, desc='remove non-stable operation shot'):
        
        df_shot = df[df.shot == shot]
        
        t_srt = df_shot.time.iloc[0]
        t_end = df_shot.time.iloc[-1]
        n_point = len(df_shot.time)
        
        # 1st filter: flattop condition
        is_short = False
        if t_end - t_srt < 4.0:
            print("shot : {} - time length issue / too short".format(shot))
            is_short = True
        
        if is_short:
            shot_ignore.append(shot)
            continue
        
        # 2nd filter: label error
        is_error = False

        if t_end < 8:
            print("Invalid shot : {} - operation time too short".format(shot))
            is_error = True
        
        elif n_point < 512:
            print("Invalid shot : {} - few data points".format(shot))
            is_error = True
        
        elif t_srt + 2.0 > t_end - 2.0:
            print("Invalid shot : {} - flat top regime X".format(shot))
            is_error = True
        
        # revision: flattop checker should eb added
        elif t_end  - t_srt < 5.0:
            print("Invalid shot : {} - flat top regime too short".format(shot))
            is_error = True
            
        if is_error:
            shot_ignore.append(shot)
        
    new_shot_list = [shot for shot in shot_list if shot not in shot_ignore]

    print("# of original shot: ", len(shot_list))
    print("# of valid shot: ", len(new_shot_list))
    
    df = df[df.shot.isin(new_shot_list)]
    
    # interpolation - time interval
    print("# Interpolation - time interval modification")
    dt0 = config.dt_ori
    dt = config.dt_set
    
    n_point = int(dt / dt0)
    
    df_interpolate = None
    
    for shot in tqdm(new_shot_list, desc = 'Interpolation process for time interval modification'):
        
        df_shot = df[df.shot == shot]
        df_shot_interpolate = {}
        
        n_res = len(df_shot) % n_point
        t = df_shot['time'].values[:-n_res].reshape(-1, n_point)
        t = np.mean(t, axis = 1, keepdims=False)
        
        df_shot_interpolate['shot'] = [shot for _ in range(len(t))]
        df_shot_interpolate['time'] = t
        
        for col in feat_cols:
            v = df_shot[col].values[:-n_res]
            v = moving_avarage_smoothing(v, k = 5, method = 'center').reshape(-1, n_point)
            v = np.mean(v, axis = 1, keepdims=False)
            
            df_shot_interpolate[col] = v
        
        df_shot_interpolate = pd.DataFrame(df_shot_interpolate)
        df_shot_interpolate['shot'] = df_shot_interpolate['shot'].astype(int)

        if df_interpolate is None:
            df_interpolate = df_shot_interpolate
        
        else:
            df_interpolate = pd.concat([df_interpolate, df_shot_interpolate], axis = 0)
            
    del df
    
    df = df_interpolate

    # Feature engineering
    print('\n',"="*50)
    print("# process: feature engineering")
    compute_dWdt(df)
    compute_tau_e(df)
    compute_tau(df)
     
    df.describe()
    
    # saving file
    print('\n',"="*50)
    print("# saving files...")
    df.to_pickle("./dataset/KSTAR_tokamak_rl_control_data.pkl")