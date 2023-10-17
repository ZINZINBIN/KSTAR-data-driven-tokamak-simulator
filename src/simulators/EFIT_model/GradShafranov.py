import numpy as np
from scipy import special
import torch, math
from torch.autograd import Function
import torch.nn.functional as F
from typing import Union, List, Dict, Optional
from src.GSsolver.KSTAR_setup import limiter_shape
import math

def compute_k(R0 : torch.Tensor, Z0:torch.Tensor, R:torch.Tensor, Z:torch.Tensor):
    k = np.sqrt(4 * R0 * R / ((R + R0) ** 2 + (Z - Z0) ** 2))
    k = np.clip(k, 1e-10, 1 - 1e-10)
    return k

def compute_ellipK_derivative(k, ellipK, ellipE):
    return ellipE / k / (1-k*k) - ellipK / k

def compute_ellipE_derivative(k, ellipK, ellipE):
    return (ellipE - ellipK) / k

def compute_Green_function(R0 : torch.Tensor, Z0:torch.Tensor, R:torch.Tensor, Z:torch.Tensor):
    k = compute_k(R0, Z0, R, Z)
    ellipK = special.ellipk(k)
    ellipE = special.ellipe(k)
    g = 0.5 / math.pi / k * ((2-k**2) * ellipK - 2 * ellipE) * 4 * math.pi * 10 **(-7)
    g *= np.sqrt(R0 * R)
    return g

def gradient(u : torch.Tensor, x : torch.Tensor):
    u_x = torch.autograd.grad(u, x, grad_outputs = torch.ones_like(u), retain_graph = True, create_graph=True)[0]
    return u_x

def compute_plasma_region(psi_s : torch.Tensor):
    mask = F.relu(1 - psi_s).ge(0).float()
    return mask

def compute_eliptic_operator(psi:torch.Tensor, R : torch.Tensor, Z : torch.Tensor):
    psi_r = gradient(psi, R)
    psi_z = gradient(psi, Z)
    psi_z2 = gradient(psi_z, Z)
    psi_r2 = gradient(psi_r, R)
    return psi_r2 - 1 / R * psi_r + psi_z2

def compute_Jphi(psi: torch.Tensor, R:torch.Tensor, Z:torch.Tensor, mu:float):
    GS = compute_eliptic_operator(psi, R, Z)
    Jphi = (-1) * GS / R / mu
    return Jphi

def compute_grad_shafranov_loss(psi : torch.Tensor, R : torch.Tensor, Z : torch.Tensor, Jphi : torch.Tensor, Rc : float, psi_s : float):
    loss = compute_eliptic_operator(psi, R, Z) * Rc ** 2 / psi_s + R * Jphi / Rc
    loss = torch.norm(loss)
    return loss

def compute_det(psi : torch.Tensor, R : torch.Tensor, Z : torch.Tensor):
    psi_r = gradient(psi, R)
    psi_z = gradient(psi, Z)
    psi_r2 = gradient(psi_r, R)
    psi_z2 = gradient(psi_z, Z)
    det = psi_r2 * psi_z2 - (psi_r * psi_z) ** 2
    return det

def gradient_sqare(psi : torch.Tensor, R : torch.Tensor, Z : torch.Tensor):
    psi_r = gradient(psi, R)
    psi_z = gradient(psi, Z)
    grad = psi_r ** 2 + psi_z ** 2
    grad = torch.sqrt(grad)
    return grad

def compute_KSTAR_limiter_mask(RR, ZZ, min_value : float= 5e-2):
    
    def convert_coord_index(RR, ZZ, points_arr):
        indices_arr = []
        for point in points_arr:
            x, y = point

            idx_x, idx_y = 0, 0
            nx,ny = RR.shape
            
            for idx in range(nx-1):
                if RR[0,idx] <= x and RR[0,idx+1] > x:
                    idx_x = idx
                    break
            
            for idx in range(ny-1):
                if ZZ[idx,0] <= y and ZZ[idx+1,0] > y:
                    idx_y = idx
                    break
            
            indices_arr.append([idx_x, idx_y])
        return np.array(indices_arr)
    
    from skimage.draw import polygon
    mask = np.ones_like(RR, dtype = np.float32) * min_value
    contour = convert_coord_index(RR, ZZ, limiter_shape)

    # Create an empty image to store the masked array
    rr, cc = polygon(contour[:, 0], contour[:, 1], mask.shape)
    mask[cc, rr] = 1

    return mask