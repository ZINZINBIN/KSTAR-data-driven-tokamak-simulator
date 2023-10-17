class SimConfig:
    
    # EFIT
    EFIT = ['\\q0', '\\q95', '\\ipmhd', '\\kappa', '\\tritop','\\tribot', '\\betap', '\\betan', '\\li', '\\bcentr', '\\rsurf', '\\aminor',]
    
    # current source
    PCPF = ['\\PCPF1U','\\PCPF2U','\\PCPF3U','\\PCPF3L','\\PCPF4U', '\\PCPF4L','\\PCPF5U','\\PCPF5L','\\PCPF6U','\\PCPF6L','\\PCPF7U']
    
    # heating source
    HEAT = ['\\nb11_pnb', '\\nb12_pnb', '\\nb13_pnb','\\EC1_PWR', '\\EC2_PWR', '\\EC3_PWR','\\EC4_PWR', '\\ECSEC1TZRTN', '\\ECSEC2TZRTN', '\\ECSEC3TZRTN','\\ECSEC4TZRTN']
    
    
    feat_predicive = {
        "EFIT": EFIT,
        "PCPF": PCPF,
        "HEAT": HEAT
    }
    
    feat_flux_gen = {
        "EFIT": EFIT,
        "PCPF": PCPF,
        "HEAT": HEAT
    }
    
    feat_cont_reg = {
        "EFIT": EFIT,
        "PCPF": PCPF,
        "HEAT": HEAT
    }