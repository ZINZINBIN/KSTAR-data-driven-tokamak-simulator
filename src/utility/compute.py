import numpy as np
import pandas as pd
import math
from typing import Literal
from tqdm.auto import tqdm

def compute_tau(df:pd.DataFrame):
    
    print("# compute tau")
    
    def _pow(x, p : float):
        return np.power(x, p)
    
    Meff = 1.5
    
    tau_L_factors = {
        '\\RC03':0.85,
        '\\bcentr':0.2,
        '\\ne_avg':0.17,
        '\\aminor':0.3,
        '\\rsurf':1.2,
        '\\kappa':0.5,
        '\\PL':0.5 * (-1)
    }
    
    df['\\ne_avg'] = df[['\\ne_tci01', '\\ne_tci02', '\\ne_tci03', '\\ne_tci04', '\\ne_tci05']].mean(axis = 1)
    df['\\TAU89'] = 0.048 * _pow(Meff, 0.5)
    
    for key in tau_L_factors.keys():
        value = tau_L_factors[key]
        df['\\TAU89'] *= _pow(df[key].abs().values, value)
    
    df['\\H89'] = df['\\TAUE'] / df['\\TAU89']
    
    return

def compute_tau_e(df:pd.DataFrame):
    print("# compute taue")
    
    df['\\PIN'] = np.array([0 for _ in range(len(df))])
    
    ECH = df[['\\EC1_PWR','\\EC2_PWR','\\EC3_PWR','\\EC4_PWR','\\EC5_PWR']].sum(axis = 1).values * 10 ** 3
    NBH = df[['\\nb11_pnb','\\nb12_pnb','\\nb13_pnb']].sum(axis = 1).values * 10 ** 6
    OHMIC = df[['\\LV01','\\LV12','\\LV23','\\LV34','\\LV45']].mean(axis = 1).abs().values * df['\\RC03'].abs().values * 10 ** 6
 
    df['\\PIN'] = ECH + NBH + OHMIC
    df['\\PL'] = df['\\PIN'] - df['\\DWDT'] * 10 ** 3
    df['\\TAUE'] = df['\\WTOT_DLM03'] * 10 ** 3 / df['\\PL']
    
    return
    
def compute_dWdt(df:pd.DataFrame):
    
    shot_list = df.shot.unique()
    df['\\DWDT'] = np.array([0 for _ in range(len(df['\\WTOT_DLM03']))])
    
    dwdt = []
    
    for shot in tqdm(shot_list, "# compute dWdt"):
        
        df_shot = df[df.shot == shot]
        dvl = df_shot['\\WTOT_DLM03'].shift(1).fillna(method = 'bfill').values.reshape(-1,)
        dvr = df_shot['\\WTOT_DLM03'].shift(-1).fillna(method = 'ffill').values.reshape(-1,)
        
        dtl = df_shot.time.shift(1).fillna(method = 'bfill').values
        dtr = df_shot.time.shift(-1).fillna(method = 'ffill').values
        
        dv = dvr - dvl
        dt = dtr - dtl
        
        dwdt.extend((dv / dt).tolist())
        
    df['\\DWDT'] = np.array(dwdt)

    return

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
    
    S = np.clip(S, 0, 1)
    
    return S