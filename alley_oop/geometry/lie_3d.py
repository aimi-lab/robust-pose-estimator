# Lie group for so3, SO3 and se3, SE3 spaces
# implementations follow https://ethaneade.com/lie.pdf


import numpy as np
import torch
from typing import Union

from alley_oop.utils.lib_handling import get_lib, get_class


def lie_so3_to_SO3(
    wvec: Union[np.ndarray, torch.Tensor] = None,
    tol: float = 10e-12,
    ) -> Union[np.ndarray, torch.Tensor]:
    """ 
    create rotation matrix in SO(3) from Lie angle 3-vector
    
    :param wvec: Lie angle 3-vector
    :return: rotation matrix in SO(3)
    """

    lib = get_lib(wvec)
    
    # define identity matrix in advance to account for torch device
    eye_3 = lib.eye(3, dtype=wvec.dtype).to(wvec.device) if lib == torch else lib.eye(3, dtype=wvec.dtype)

    # check if vector of zeros
    if not wvec.any():
        return eye_3

    # compute scale from vector norm
    theta = (wvec.T @ wvec)**.5

    # normalize vector
    wvec = wvec / theta if theta > tol else wvec

    # construct hat-map which is so(3)
    wmat = lie_hatmap(wvec)

    # compute exponential of hat-map using Taylor expansion (known as Rodrigues formula)
    rmat = eye_3 + lib.sin(theta) * wmat + (1-lib.cos(theta)) * wmat @ wmat

    return rmat


def lie_SO3_to_so3(
        rmat: Union[np.ndarray, torch.Tensor] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
    """ 
    create Lie angle 3-vector from rotation matrix in SO(3)
    
    :param rmat: rotation matrix in SO(3)
    :return: Lie angle 3-vector
    """

    lib = get_lib(rmat)

    # check if trace = -1
    if (lib.trace(rmat)+1):
        #   rotation by +/- pi, +/- 3pi etc.
        pass
    
    # compute scale
    theta = lib.arccos((lib.trace(rmat)-1)/2)
    theta_term = theta/(2*lib.sin(theta)) if theta != 0 else 0.5

    # compute logarithm of rotation matrix
    ln_rmat = theta_term * (rmat-rmat.T)

    # obtain used array data type
    data_class = get_class(rmat)

    # extract elements from hat-map
    wvec = data_class([ln_rmat[2, 1]-ln_rmat[1, 2], ln_rmat[0, 2]-ln_rmat[2, 0], ln_rmat[1, 0]-ln_rmat[0, 1]]) / 2

    return wvec


def lie_se3_to_SE3(
        pvec: Union[np.ndarray, torch.Tensor] = None,
        tol: float = 10e-12,
    ) -> Union[np.ndarray, torch.Tensor]:
    """
    create 4x4 projection matrix in SE(3) from Lie input vector
    
    :param pvec: concatenated Lie angle 3-vector and Lie translation 3-vector
    :return: projection matrix in SE(3), translation vector R^3
    """

    lib = get_lib(pvec)

    # define identity matrix in advance to account for torch device
    eye_3 = lib.eye(3, dtype=pvec.dtype).to(pvec.device) if lib == torch else lib.eye(3, dtype=pvec.dtype)

    # compute scale from vector norm
    theta = (pvec[:3].T @ pvec[:3])**.5

    # Taylor coefficients for rotation
    a_term = lib.sin(theta)
    b_term = (1-lib.cos(theta))

    # normalize vector
    wvec_norm = pvec[:3] / theta if theta > tol else pvec[:3]

    # construct hat-map which is so(3)
    wmat = lie_hatmap(wvec_norm)
    
    rmat = eye_3 + a_term * wmat + b_term * wmat @ wmat

    # Taylor coefficients for translation (theta terms do not cancel out as for rotation)
    c_term = (1 - a_term / theta) if theta**2 > tol else theta**2/6-theta**4/120
    b_term2 = b_term/theta if theta**2 > tol else (0.5*theta - theta**3/24 + theta**5/720)
    vmat = eye_3 + b_term2 * wmat + c_term * wmat @ wmat

    tvec = vmat @ pvec[3:]

    pmat = lib.eye(4, dtype=pvec.dtype).to(pvec.device) if lib == torch else lib.eye(4, dtype=pvec.dtype)
    pmat[:3, :3] = rmat
    pmat[:3, -1] = tvec

    return pmat


def lie_SE3_to_se3(
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

    # compute scale from vector norm
    theta = (wvec.T @ wvec)**.5

    # Taylor 1 - A/(2B) -> directly compute value because thetas do not cancel out as for rotation
    mul_term = 1- (theta*lib.sin(theta)/(2*(1-lib.cos(theta)))) if theta > tol else theta**2/12+theta**4/720

    # normalize vector
    wvec_norm = wvec / theta if theta > tol else wvec

    # construct hat-map which is so(3)
    wmat = lie_hatmap(wvec_norm)
    Eye = lib.eye(3, dtype=wmat.dtype).to(wmat.device) if lib == torch else lib.eye(3, dtype=wmat.dtype)
    vmat_inv = Eye - .5*theta*wmat + mul_term * wmat @ wmat
    uvec = vmat_inv @ pmat[:3, -1]

    pvec = lib.hstack([wvec, uvec])

    return pvec


def lie_hatmap(
        wvec: Union[np.ndarray, torch.Tensor] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
    """ 
    create hat-map in so(3) from Euler vector in R^3
    
    :param wvec: Euler vector in R^3
    :return: hat-map in so(3)
    """

    assert len(wvec) == 3, 'argument must be a 3-vector'

    data_class = get_class(wvec)

    wmat = data_class([
        [0, -wvec[2], +wvec[1]],
        [+wvec[2], 0, -wvec[0]],
        [-wvec[1], +wvec[0], 0],
    ])

    # consider torch device
    if isinstance(wvec, torch.Tensor):
        wmat = wmat.to(wvec.device)

    return wmat


def is_SO3(
        rmat: Union[np.ndarray, torch.Tensor], 
        tol: float = 10e-6, 
    ) -> bool:
    """
    test whether provided rotation matrix is part of SO(3) group

    :param rmat: rotation matrix in SO(3)
    :return: True
    """

    lib = get_lib(rmat)

    assert rmat.shape[0] == 3 and rmat.shape[1] == 3, 'matrix must be 3x3'
    assert lib.linalg.det(rmat @ rmat.T) > 0, 'det(R @ R.T) must be greater than zero'
    assert lib.linalg.norm(rmat @ rmat.T - lib.eye(rmat.shape[0])) < tol, 'R @ R.T must yield identity'
    
    return True
