import torch
import logging
import numpy as np
logger_py = logging.getLogger(__name__)

def vec2skew(v):
    """
    :param v:  (n_rays, 3) torch tensor
    :return:   (n_rays, 3, 3)
    """
    n_rays = v.shape[0]
    zero = torch.zeros(n_rays, dtype=torch.float32, device=v.device)[..., None]  # (n_rays, 1)
    skew_v0 = torch.cat([ zero,    -v[..., 2:3],   v[..., 1:2]], dim=-1)  # (n_rays, 3)
    skew_v1 = torch.cat([ v[..., 2:3],   zero,    -v[..., 0:1]], dim=-1)  # (n_rays, 3)
    skew_v2 = torch.cat([-v[..., 1:2],   v[..., 0:1],   zero], dim=-1)  # (n_rays, 3)
    skew_v = torch.stack([skew_v0, skew_v1, skew_v2], dim=1)  # (n_rays, 3, 3)
    return skew_v  # (n_rays, 3, 3)


def Exp(r):
    """so(3) vector to SO(3) matrix
    :param r: (n_rays, 3)   torch tensor
    :return:  (n_rays, 3, 3)
    """
    n_rays = r.shape[0]
    skew_r = vec2skew(r)  # (n_rays, 3, 3)
    norm_r = r.norm(dim=-1,keepdim=True).unsqueeze(-1) + 1e-15  # (n_rays, 1, 1)
    eye = torch.eye(3, dtype=torch.float32, device=r.device).repeat(n_rays, 1, 1)  # (n_rays, 3, 3)
    R = eye + (torch.sin(norm_r) / norm_r) * skew_r + ((1 - torch.cos(norm_r)) / norm_r**2) * (skew_r @ skew_r)
    return R

def make_c2w(r, t):
    """
    :param r:  (n_rays, 3)  torch tensor
    :param t:  (n_rays, 3)  torch tensor
    :return:   (n_rays, 3, 4)
    """
    R = Exp(r)  # (n_rays, 3, 3)
    c2w = torch.cat([R, t.unsqueeze(-1)], dim=-1)  # (n_rays, 3, 4)
    c2w = convert3x4_4x4(c2w)
    return c2w
 
def convert3x4_4x4(input):
    """
    :param input:  (N, 3, 4) or (3, 4) torch or np
    :return:       (N, 4, 4) or (4, 4) torch or np
    """
    if input.shape[-2] == 4:
        return input
    if torch.is_tensor(input):
        if len(input.shape) == 3:
            output = torch.cat([input, torch.zeros_like(input[:, 0:1])], dim=1)  # (N, 4, 4)
            output[:, 3, 3] = 1.0
        else:
            output = torch.cat([input, torch.tensor([[0,0,0,1]], dtype=input.dtype, device=input.device)], dim=0)  # (4, 4)
    else:
        if len(input.shape) == 3:
            output = np.concatenate([input, np.zeros_like(input[:, 0:1])], axis=1)  # (N, 4, 4)
            output[:, 3, 3] = 1.0
        else:
            output = np.concatenate([input, np.array([[0,0,0,1]], dtype=input.dtype)], axis=0)  # (4, 4)
            output[3, 3] = 1.0
    return output