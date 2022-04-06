# Lie group for so3, SO3 and se3, SE3 spaces
# implementations follow https://ethaneade.com/lie.pdf


import numpy as np
import torch
from typing import Union

from alley_oop.utils.lib_handling import get_lib_type, get_data_class


def lie_so3_to_SO3(
    wvec: Union[np.ndarray, torch.Tensor] = None,
    tol: float = 10e-12
    ) -> Union[np.ndarray, torch.Tensor]:

    lib = get_lib_type(wvec)

    # check if vector of zeros
    if not wvec.any():
        return lib.eye(3)

    theta = (wvec.T @ wvec)**.5

    wvec = wvec / theta if theta > tol else wvec
    wmat = lie_hatmap(wvec)

    rmat = lib.eye(3) + wmat * (lib.sin(theta)) + wmat @ wmat *((1-lib.cos(theta)))

    return rmat


def lie_SO3_to_so3(
        rmat: Union[np.ndarray, torch.Tensor] = None
    ) -> Union[np.ndarray, torch.Tensor]:

    lib = get_lib_type(rmat)

    # check if trace = -1
    if (lib.trace(rmat)+1):
        #   rotation by +/- pi, +/- 3pi etc.
        pass
    
    theta = lib.arccos((lib.trace(rmat)-1)/2)
    theta_term = theta/(2*lib.sin(theta)) if theta != 0 else 0.5
    ln_rmat = theta_term * (rmat-rmat.T)

    data_class = get_data_class(rmat)

    wvec = data_class([ln_rmat[2, 1]-ln_rmat[1, 2], ln_rmat[0, 2]-ln_rmat[2, 0], ln_rmat[1, 0]-ln_rmat[0, 1]]) / 2

    return wvec


def lie_SE3_to_se3(
        rmat: Union[np.ndarray, torch.Tensor] = None, 
        tvec: Union[np.ndarray, torch.Tensor] = None
    ) -> Union[np.ndarray, torch.Tensor]:

    lib = get_lib_type(rmat)

    wvec = lie_SO3_to_so3(rmat)
    wmat = lie_hatmap(wvec)

    theta = (wvec.T @ wvec)**.5
    a_term = lib.sin(theta) / theta
    b_term = (1-lib.cos(theta)) / theta**2
    #c_term = (1-a_term) / theta**2

    vmat_inv = lib.eye(3) - .5*wmat + 1/theta**2*(1-a_term/(2*b_term)) * wmat @ wmat

    uvec = vmat_inv @ tvec

    return wvec, uvec


def lie_se3_to_SE3(
        wvec: Union[np.ndarray, torch.Tensor] = None,
        uvec: Union[np.ndarray, torch.Tensor] = None
    ) -> Union[np.ndarray, torch.Tensor]:

    lib = get_lib_type(wvec)

    wmat = lie_hatmap(wvec)
    rmat = lie_so3_to_SO3(wvec)

    theta = (wvec.T @ wvec)**.5
    a_term = lib.sin(theta) / theta
    b_term = (1-lib.cos(theta)) / theta**2
    c_term = (1-a_term) / theta**2
    
    #rmat = np.eye(3) - a_term*wmat + b_term*wmat**2
    vmat = lib.eye(3) - b_term*wmat + c_term*wmat @ wmat

    tvec = vmat @ uvec

    return rmat, tvec


def lie_hatmap(
        wvec: Union[np.ndarray, torch.Tensor] = None
    ) -> Union[np.ndarray, torch.Tensor]:
    """ 
    create hat-map in so(3) from Euler vector in R^3
    
    :param wvec: Euler vector in R^3
    :return: hat-map in so(3)
    """

    assert len(wvec) == 3, 'argument must be a 3-vector'

    data_class = get_data_class(wvec)

    wmat = data_class([
        [0, -wvec[2], +wvec[1]],
        [+wvec[2], 0, -wvec[0]],
        [-wvec[1], +wvec[0], 0],
    ])

    return wmat


def is_SO3(
        rmat: Union[np.ndarray, torch.Tensor], 
        tol: float = 10e-6, 
    ) -> bool:

    lib = get_lib_type(rmat)

    assert rmat.shape[0] == 3 and rmat.shape[1] == 3, 'matrix must be 3x3'
    assert lib.linalg.norm(rmat @ rmat.T - lib.eye(rmat.shape[0])) < tol, 'R @ R.T must yield identity'
    assert lib.linalg.det(rmat @ rmat.T) > 0, 'det(R @ R.T) must be greater than zero'
    
    return True
