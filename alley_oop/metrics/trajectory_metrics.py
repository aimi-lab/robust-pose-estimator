import numpy as np
import torch


def absolute_trajectory_error(gt_poses, predicted_poses):
    assert len(gt_poses) == len(predicted_poses)
    lib = torch if torch.is_tensor(gt_poses[0]) else np
    ate_rot = []
    ate_pos = []
    for gt, pred in zip(gt_poses, predicted_poses):
        diff_angle = lib.arccos((lib.trace(gt[:3, :3].T@pred[:3, :3]) - 1)/2) # angle-axis representation
        ate_rot.append(diff_angle**2)
        ate_pos.append(lib.sum((gt[:3,3].T-pred[:3, 3])**2))
    ate_rot = 180.0/lib.pi*lib.sqrt(lib.mean(ate_rot))
    ate_pos = lib.sqrt(lib.mean(ate_pos))
    return ate_pos, ate_rot
