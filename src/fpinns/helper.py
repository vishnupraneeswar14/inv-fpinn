import math
import torch


def smooth_ceil_exact(x, k=100, eps=1e-6):
    x = torch.as_tensor(x, dtype=torch.float32)
    pi_x = math.pi * x
    transition = (2 / (1 + torch.exp(-k * torch.sin(pi_x)))) - 1
    u = torch.cos(pi_x)
    u = torch.clamp(u, -1 + eps, 1 - eps)
    smooth_value = x + (transition * torch.arcsin(u)) / math.pi + 0.5
    return smooth_value


def smooth_floor_exact(x):
    k=100
    x = torch.as_tensor(x, dtype=torch.float32)
    pi_x = math.pi * x
    transition = (2 / (1 + torch.exp(-k * torch.sin(pi_x)))) - 1
    smooth_value = x + (transition * torch.arcsin(torch.cos(pi_x))) / math.pi - 0.5
    return smooth_value


def smooth_ceil(x, alpha=10.0):
    x = torch.as_tensor(x, dtype=torch.float32)
    pi_x = math.pi * x
    frac_smooth = 0.5 * (1 + torch.tanh(alpha * torch.sin(pi_x)))
    return x + frac_smooth
