import torch

def slerp(val, low, high):
    low_norm = low/torch.norm(low, dim=-1, keepdim=True)
    high_norm = high/torch.norm(high, dim=-1, keepdim=True)
    omega = torch.acos(torch.clamp((low_norm * high_norm).sum(-1), -1, 1))
    so = torch.sin(omega)
    if so < 1e-3:
        return (torch.ones_like(val) - val) * low + val * high
    res = (torch.sin((1 - val)*omega) / so) * low + (torch.sin(val * omega) / so) * high
    return res

def linterp(val, low, high):
    res = (1 - val) * low + val * high
    return res