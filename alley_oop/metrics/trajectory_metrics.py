import numpy as np
import torch
from typing import Union, Tuple
from alley_oop.utils.lib_handling import get_lib


def _align(model, data):
    """Align two trajectories using the method of Horn (closed-form).

    Input:
    model -- first trajectory (3xn)
    data -- second trajectory (3xn)

    Output:
    rot -- rotation matrix (3x3)
    trans -- translation vector (3x1)
    trans_error -- translational error per point (1xn)

    """
    model_zerocentered = model - model.mean(1)[...,None]
    data_zerocentered = data - data.mean(1)[...,None]

    W = np.zeros((3, 3))
    for column in range(model.shape[1]):
        W += np.outer(model_zerocentered[:, column], data_zerocentered[:, column])
    U, d, Vh = np.linalg.linalg.svd(W.transpose())
    S = np.matrix(np.identity(3))
    if (np.linalg.det(U) * np.linalg.det(Vh) < 0):
        S[2, 2] = -1
    rot = U * S * Vh
    trans = data.mean(1)[...,None] - rot * model.mean(1)[...,None]
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = trans.squeeze()
    return T


def absolute_trajectory_error(gt_poses: Union[np.ndarray, torch.Tensor],
                              predicted_poses: Union[np.ndarray, torch.Tensor],
                              prealign: bool=True) -> Tuple[float, Union[np.ndarray, torch.Tensor]]:
    """
        Absolute Trajectory Error ATE-RMSE

    :param gt_poses: array of ground-truth poses, shape nx4x4
    :param predicted_poses: array of estimated poses, shape nx4x4
    :param prealign: if true, pre-align trajectories using method of Horn
    :return: ate-RMSE, translation errors
    :rtype: float
    """
    assert len(gt_poses) == len(predicted_poses)
    lib = get_lib(gt_poses)
    if prealign:
        T = _align(predicted_poses[:, :3, 3].T, gt_poses[:, :3, 3].T)
        predicted_poses = T[None,...] @ predicted_poses

    trans_err = []
    for gt, pred in zip(gt_poses, predicted_poses):
        trans_err.append(lib.sum((gt[:3,3].T-pred[:3, 3])**2))
    trans_err = np.asarray(trans_err)
    ate_pos = lib.sqrt(lib.mean(trans_err))
    return ate_pos, np.sqrt(trans_err)


def relative_pose_error(gt_poses: Union[np.ndarray, torch.Tensor],
                        predicted_poses: Union[np.ndarray, torch.Tensor],
                        delta: int=1):
    """
            Relative Pose Error RPE (mean)

        :param gt_poses: array of ground-truth poses, shape nx4x4
        :param predicted_poses: array of estimated poses, shape nx4x4
        :param delta: time interval to compute relative poses
        :return: rpe-trans, rpe-rot
        :rtype: float
        """
    assert len(gt_poses) == len(predicted_poses)
    lib = get_lib(gt_poses)
    trans_errors = []
    rot_errors = []
    for i in range(len(gt_poses)-delta):
        gt_rel = np.linalg.inv(gt_poses[i]) @ gt_poses[i+delta]
        pred_rel = np.linalg.inv(predicted_poses[i]) @ predicted_poses[i+delta]
        rel_err = np.linalg.inv(gt_rel) @ pred_rel

        trans_errors.append(lib.sqrt(lib.sum((rel_err[:3,3])**2)))
        d = 0.5*(lib.trace(rel_err[:3, :3]) - 1)
        rot_errors.append(np.arccos(max(min(d, 1.0), -1.0)))

    rpe_trans = np.mean(np.asarray(trans_errors))
    rpe_rot = np.mean(np.asarray(rot_errors))
    return rpe_trans, rpe_rot
