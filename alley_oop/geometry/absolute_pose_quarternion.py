import numpy as np
import torch
from alley_oop.geometry.lie_3d import lie_SE3_to_se3


def align(reference, query, estimate_scale=False, ret_homogenous=False):
    """Align two trajectories using the method of Horn (closed-form).
    B. K. P. Horn, “Closed-form solution of absolute orientation using unit quaternions,”
    Journal of the Optical Society of America A , vol. 4, no. 4, pp. 629–642, 1987

    Input:
    model -- first trajectory (3xn)
    data -- second trajectory (3xn)

    Output:
    rot -- rotation matrix (3x3)
    trans -- translation vector (3x1)
    trans_error -- translational error per point (1xn)

    """
    model_zerocentered = reference - reference.mean(1)[:,None]
    data_zerocentered = query - query.mean(1)[:,None]

    W = np.zeros((3, 3))
    for column in range(reference.shape[1]):
        W += np.outer(model_zerocentered[:, column], data_zerocentered[:, column])
    U, d, Vh = np.linalg.linalg.svd(W.transpose())
    S = np.matrix(np.identity(3))
    if (np.linalg.det(U) * np.linalg.det(Vh) < 0):
        S[2, 2] = -1
    rot = U * S * Vh

    rotmodel = rot * model_zerocentered
    dots = 0.0
    norms = 0.0

    for column in range(data_zerocentered.shape[1]):
        dots += np.dot(data_zerocentered[:, column].transpose(), rotmodel[:, column])
        normi = np.linalg.norm(model_zerocentered[:, column])
        norms += normi * normi

    scale = float(dots / norms) if estimate_scale else 1.0

    trans = query.mean(1)[:,None] - scale * rot * reference.mean(1)[:,None]

    model_aligned = scale * rot * reference + trans
    alignment_error = model_aligned - query

    residuals = np.sqrt(np.sum(np.multiply(alignment_error, alignment_error), 0)).A[0]
    if not ret_homogenous:
        return rot, trans, residuals, scale, alignment_error
    else:
        pose = np.eye(4)
        pose[:3,:3] = rot
        pose[:3,3] = trans.squeeze()
        return pose, residuals, scale, alignment_error


def align_torch(reference, query):
    """Align two trajectories using the method of Horn (closed-form).
    B. K. P. Horn, “Closed-form solution of absolute orientation using unit quaternions,”
    Journal of the Optical Society of America A , vol. 4, no. 4, pp. 629–642, 1987

    Input:
    model -- first trajectory (batchx3xn)
    data -- second trajectory (batchx3xn)

    Output:
    se(3) alignment (batchx6)

    """
    ref_mean = reference.mean(-1)[...,None]
    query_mean = query.mean(-1)[...,None]
    model_zerocentered = reference - ref_mean
    data_zerocentered = query - query_mean
    N = reference.shape[0]
    outer = torch.bmm(model_zerocentered.permute(0,2,1).reshape(-1, 3).unsqueeze(2), data_zerocentered.permute(0,2,1).reshape(-1, 3).unsqueeze(1))
    W = outer.reshape(N, -1, 3, 3).sum(dim=1)
    U, d, Vh = torch.linalg.svd(W.transpose(1,2))
    S = torch.eye(3, device=reference.device).reshape((1, 3, 3)).repeat(N, 1, 1)
    det = torch.linalg.det(U) * torch.linalg.det(Vh)
    S[:, 2, 2] = torch.sign(det)
    rot = U @ S @ Vh

    trans = query_mean - rot @ ref_mean
    se3 = torch.zeros((N,6), device=reference.device)
    for b in range(N):
        T = torch.eye(4, device=reference.device)
        T[:3,:3] = rot[b]
        T[:3,3] = trans[b].squeeze()
        se3[b] = lie_SE3_to_se3(T)
    return se3, rot, trans
