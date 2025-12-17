
import numpy as np

import ATE.trajectory_utils as tu
import ATE.transformations as tf
from utils_poses.align_traj import align_ate_c2b_use_a2b
from utils_poses.lie_group_helper import SO3_to_quat
from ATE.compute_trajectory_errors import compute_absolute_error
from ATE.results_writer import compute_statistics

def rotation_error(pose_error):
    """Compute rotation error
    Args:
        pose_error (4x4 array): relative pose error
    Returns:
        rot_error (float): rotation error
    """
    a = pose_error[0, 0]
    b = pose_error[1, 1]
    c = pose_error[2, 2]
    d = 0.5*(a+b+c-1.0)
    rot_error = np.arccos(max(min(d, 1.0), -1.0))
    return rot_error

def translation_error(pose_error):
    """Compute translation error
    Args:
        pose_error (4x4 array): relative pose error
    Returns:
        trans_error (float): translation error
    """
    dx = pose_error[0, 3]
    dy = pose_error[1, 3]
    dz = pose_error[2, 3]
    trans_error = np.sqrt(dx**2+dy**2+dz**2)
    return trans_error

def compute_rpe(gt, pred):
    trans_errors = []
    rot_errors = []
    for i in range(len(gt)-1):
        gt1 = gt[i]
        gt2 = gt[i+1]
        gt_rel = np.linalg.inv(gt1) @ gt2

        pred1 = pred[i]
        pred2 = pred[i+1]
        pred_rel = np.linalg.inv(pred1) @ pred2
        rel_err = np.linalg.inv(gt_rel) @ pred_rel
        
        trans_errors.append(translation_error(rel_err))
        rot_errors.append(rotation_error(rel_err))
    rpe_trans = np.mean(np.asarray(trans_errors))
    rpe_rot = np.mean(np.asarray(rot_errors))
    return rpe_trans, rpe_rot

def compute_ATE(gt, pred):
    """Compute RMSE of ATE
    Args:
        gt: ground-truth poses
        pred: predicted poses
    """
    errors = []

    for i in range(len(pred)):
        # cur_gt = np.linalg.inv(gt_0) @ gt[i]
        cur_gt = gt[i]
        gt_xyz = cur_gt[:3, 3] 

        # cur_pred = np.linalg.inv(pred_0) @ pred[i]
        cur_pred = pred[i]
        pred_xyz = cur_pred[:3, 3]

        align_err = gt_xyz - pred_xyz

        errors.append(np.sqrt(np.sum(align_err ** 2)))
    ate = np.sqrt(np.mean(np.asarray(errors) ** 2)) 
    return ate

def compute_ate_sim3(c2ws_a, c2ws_b):    
    """Compuate ate between a and b.
    :param c2ws_a: (N, 3/4, 4) torch
    :param c2ws_b: (N, 3/4, 4) torch
    :param align_a2b: None or 'sim3'. Set to None if a and b are pre-aligned.
    """
    c2ws_a_aligned = align_ate_c2b_use_a2b(c2ws_a, c2ws_b)
    # c2ws_a_aligned = c2ws_a
    R_a_aligned = c2ws_a_aligned[:, :3, :3].cpu().numpy()
    t_a_aligned = c2ws_a_aligned[:, :3, 3].cpu().numpy()

    R_b = c2ws_b[:, :3, :3].cpu().numpy()
    t_b = c2ws_b[:, :3, 3].cpu().numpy()
    quat_a_aligned = SO3_to_quat(R_a_aligned)
    quat_b = SO3_to_quat(R_b)

    e_trans, e_trans_vec, e_rot, e_ypr, e_scale_perc = compute_absolute_error(t_a_aligned,quat_a_aligned,
                                                                              t_b, quat_b)
    stats_tran = compute_statistics(e_trans)
    stats_rot = compute_statistics(e_rot)
    stats_scale = compute_statistics(e_scale_perc)

    return stats_tran, stats_rot, stats_scale  # dicts

def compute_ate(c2ws_a, c2ws_b):    
    """Compuate ate between a and b.
    :param c2ws_a: (N, 3/4, 4) torch
    :param c2ws_b: (N, 3/4, 4) torch
    :param align_a2b: None or 'sim3'. Set to None if a and b are pre-aligned.
    """
    # c2ws_a_aligned = align_ate_c2b_use_a2b(c2ws_a, c2ws_b)
    c2ws_a_aligned = c2ws_a
    R_a_aligned = c2ws_a_aligned[:, :3, :3].cpu().numpy()
    t_a_aligned = c2ws_a_aligned[:, :3, 3].cpu().numpy()

    R_b = c2ws_b[:, :3, :3].cpu().numpy()
    t_b = c2ws_b[:, :3, 3].cpu().numpy()
    quat_a_aligned = SO3_to_quat(R_a_aligned)
    quat_b = SO3_to_quat(R_b)

    e_trans, e_trans_vec, e_rot, e_ypr, e_scale_perc = compute_absolute_error(t_a_aligned,quat_a_aligned,
                                                                              t_b, quat_b)
    stats_tran = compute_statistics(e_trans)
    stats_rot = compute_statistics(e_rot)
    stats_scale = compute_statistics(e_scale_perc)

    return stats_tran, stats_rot, stats_scale  # dicts