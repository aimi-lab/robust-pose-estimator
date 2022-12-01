# Lie group for so3, SO3 and se3, SE3 spaces
# Small angle version for better numerical stability
# implementations follow https://ethaneade.com/lie.pdf

from core.geometry.lie_3d import *


def small_angle_lie_se3_to_SE3(
        pvec: Union[np.ndarray, torch.Tensor] = None,
        tol: float = 1e-12,
    ) -> Union[np.ndarray, torch.Tensor]:
    """
    create 4x4 projection matrix in SE(3) from Lie input vector
    
    :param pvec: concatenated Lie angle 3-vector and Lie translation 3-vector
    :return: projection matrix in SE(3), translation vector R^3
    """

    lib = get_lib(pvec)

    # define identity matrix in advance to account for torch device
    eye_3 = lib.eye(3, dtype=pvec.dtype, device=pvec.device) if lib == torch else lib.eye(3, dtype=pvec.dtype)

    # compute scale from vector norm
    theta = (pvec[:3] @ pvec[:3])**.5

    # Taylor coefficients for rotation
    a_term = lib.sin(theta)
    b_term = (1-lib.cos(theta))

    # normalize vector
    wvec_norm = pvec[:3] / theta if theta > tol else pvec[:3]

    # construct hat-map which is so(3)
    wmat = lie_hatmap(wvec_norm)
    
    rmat = eye_3 + a_term * wmat + b_term * wmat @ wmat

    tvec = pvec[3:]

    pmat = lib.eye(4, dtype=pvec.dtype, device=pvec.device) if lib == torch else lib.eye(4, dtype=pvec.dtype)
    pmat[:3, :3] = rmat
    pmat[:3, -1] = tvec

    return pmat


def small_angle_lie_se3_to_SE3_batch(
        pvec: Union[np.ndarray, torch.Tensor] = None,
        tol: float = 10e-12,
    ) -> Union[np.ndarray, torch.Tensor]:
    """
    create 4x4 projection matrix in SE(3) from Lie input vector

    :param pvec: concatenated Lie angle 3-vector and Lie translation 3-vector
    :return: projection matrix in SE(3), translation vector R^3
    """

    if pvec.ndim == 1:
        pvec = pvec.unsqueeze(0)
    b = pvec.shape[0]
    # define identity matrix in advance to account for torch device
    eye_3 = beye((b, 3), dtype=pvec.dtype, device=pvec.device)

    # compute scale from vector norm
    theta = batched_dot_product(pvec[:, :3],pvec[:, :3]).squeeze(-1)**.5
    theta = torch.clamp(theta, tol)

    # Taylor coefficients for rotation
    a_term = torch.sin(theta).unsqueeze(-1)
    b_term = (1-torch.cos(theta)).unsqueeze(-1)

    # normalize vector
    wvec_norm = pvec[:, :3] / theta

    # construct hat-map which is so(3)
    wmat = lie_hatmap_batch(wvec_norm)
    rmat = eye_3 + a_term * wmat + b_term * torch.bmm(wmat,wmat)

    tvec = pvec[:, 3:, None]

    pmat = beye((b, 4), dtype=pvec.dtype, device=pvec.device)
    pmat[:, :3, :3] = rmat
    pmat[:, :3, -1] = tvec.squeeze(-1)

    return pmat


def small_angle_lie_se3_to_SE3_batch_lin(
        pvec: Union[np.ndarray, torch.Tensor] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
    """
    create 4x4 projection matrix in SE(3) from Lie input vector for very small angles

    :param pvec: concatenated Lie angle 3-vector and Lie translation 3-vector
    :return: projection matrix in SE(3), translation vector R^3
    """

    if pvec.ndim == 1:
        pvec = pvec.unsqueeze(0)
    b = pvec.shape[0]
    # define identity matrix in advance to account for torch device
    eye_3 = beye((b, 3), dtype=pvec.dtype, device=pvec.device)

    # construct hat-map which is so(3)
    wmat = lie_hatmap_batch(pvec[:, :3])
    rmat = eye_3 + wmat

    pmat = beye((b, 4), dtype=pvec.dtype, device=pvec.device)
    pmat[:, :3, :3] = rmat
    pmat[:, :3, -1] = pvec[:, 3:]

    return pmat


def small_angle_lie_SE3_to_se3(
        pmat: Union[np.ndarray, torch.Tensor] = None, 
        tol: float = 10e-12,
    ) -> Union[np.ndarray, torch.Tensor]:
    """
    create projection vector in se(3) from SE(3) projection matrix
    
    :param pmat: concatenated rotation matrix in SO(3) and translation vector R^3
    :return: concatenated Lie angle 3-vector and Lie translation 3-vector
    """

    lib = get_lib(pmat)

    wvec = lie_SO3_to_so3(pmat[:3, :3])
    uvec = pmat[:3, -1]
    pvec = lib.hstack([wvec, uvec])

    return pvec


def small_angle_lie_SE3_to_se3_batch(
        pmat: Union[np.ndarray, torch.Tensor] = None,
        tol: float = 10e-12,
) -> Union[np.ndarray, torch.Tensor]:
    """
    create projection vector in se(3) from SE(3) projection matrix

    :param pmat: concatenated rotation matrix in SO(3) and translation vector R^3
    :return: concatenated Lie angle 3-vector and Lie translation 3-vector
    """

    lib = get_lib(pmat)
    pvec = []
    for n in range(pmat.shape[0]):
        wvec = lie_SO3_to_so3(pmat[n, :3, :3])
        uvec = pmat[n, :3, -1]
        pvec.append(lib.hstack([wvec, uvec]))
    pvec = lib.stack(pvec)
    return pvec