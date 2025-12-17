import torch
import numpy as np
import logging
from matplotlib import pyplot as plt
import os
import shutil
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R
logger_py = logging.getLogger(__name__)

def vec2skew(v):
    """
    :param v:  (n_rays, 3) torch tensor
    :return:   (n_rays, 3, 3)
    """
    n_rays = v.shape[0]
    zero = torch.zeros(n_rays, dtype=torch.float32, device=v.device)[..., None]
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
    return c2w
 