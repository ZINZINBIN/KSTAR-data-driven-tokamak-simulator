class SimConfig:
    
    # time interval
    dt_ori = 0.01
    dt_set = 0.05
    
    # EFIT
    EFIT = ['\\q95', '\\ipmhd', '\\kappa', '\\tritop','\\tribot', '\\betap', '\\betan', '\\li', '\\bcentr', '\\rsurf', '\\aminor', '\\drsep','\\rxpt1','\\zxpt1', '\\rxpt2', '\\zxpt2']
    
    # Current source
    PCPF = ['\\PCPF1U','\\PCPF2U','\\PCPF3U','\\PCPF3L','\\PCPF4U','\\PCPF4L','\\PCPF5U','\\PCPF5L','\\PCPF6U','\\PCPF6L','\\PCPF7U']
    
    # Heating source
    ECH = ['\\EC1_PWR','\\EC2_PWR','\\EC3_PWR','\\EC4_PWR','\\EC5_PWR']                                         # EC heating 
    NBH = ['\\nb11_pnb','\\nb12_pnb','\\nb13_pnb']                                                              # NB heating

    # Magnetic signals
    LV = ['\\LV01','\\LV12','\\LV23','\\LV34','\\LV45']
    DL = ['\\BETAP_DLM03','\\DMF_DLM03','\\WTOT_DLM03']
    RC = ['\\RC03','\\VCM03'] # + ['\\RCPPU1','\\RCPPU2:FOO','\\RCPPU2B:FOO','\\RCPPL1','\\RCPPL2B:FOO']
    
    # Gas flow
    GAS = ['\\I_GFLOW_IN:FOO', '\\D_GFLOW_IN:FOO', '\\G_GFLOW_IN:FOO','\\G_NFLOW_IN:FOO']
    
    # ECE
    ECE = [
        '\\ECE04','\\ECE05','\\ECE06','\\ECE07','\\ECE08', 
        '\\ECE09', '\\ECE10', '\\ECE11', '\\ECE12', '\\ECE13', '\\ECE14', '\\ECE15',
        '\\ECE16', '\\ECE17', '\\ECE18', '\\ECE19', '\\ECE20', '\\ECE21', '\\ECE22',
        '\\ECE23', '\\ECE24', '\\ECE25', '\\ECE26', '\\ECE27', '\\ECE29', '\\ECE30',
        '\\ECE31', '\\ECE32', '\\ECE33', '\\ECE34', '\\ECE35', '\\ECE36', '\\ECE37',
        '\\ECE38', '\\ECE39', '\\ECE40', '\\ECE41', '\\ECE42', '\\ECE43', '\\ECE44',
        '\\ECE45', '\\ECE46', '\\ECE47', '\\ECE50', '\\ECE51', '\\ECE53', '\\ECE54',
        '\\ECE55', '\\ECE56', '\\ECE57', '\\ECE58', '\\ECE60', '\\ECE61', '\\ECE62',
        '\\ECE63', '\\ECE64', '\\ECE65', '\\ECE66', '\\ECE67', '\\ECE68', '\\ECE69',
        '\\ECE70', '\\ECE71', '\\ECE72', '\\ECE73', '\\ECE74', '\\ECE75', '\\ECE76'
    ]
    
    # TCI
    TCI = ['\\ne_tci01', '\\ne_tci02', '\\ne_tci03', '\\ne_tci04', '\\ne_tci05']
    
    # Bolometer signals
    BOL = ['\\ax3_bolo02:FOO']
    
    feat_predicive = {
        "state": [
            '\\q95','\\betap', '\\betan', '\\li', 
            '\\kappa', '\\tritop', '\\tribot', '\\rsurf', 
            '\\aminor', # '\\drsep','\\rxpt1','\\zxpt1','\\rxpt2','\\zxpt2'
            ],
        "control": ['\\RC03'] + PCPF + ECH + NBH + GAS + LV,
    }
    
    # Model architecture
    model_config = {
        "branch_hidden_dim":64,
        "branch_layers":4,
        "trunk_hidden_dim":64,
        "trunk_layers":4,
        "enc_hidden_dim":64,
        "enc_n_channel":32,
        "enc_kernel_size":3,
        "enc_depth":4,
        "dropout":0.2,
        "dilation_size":2,
        "q":128
    }