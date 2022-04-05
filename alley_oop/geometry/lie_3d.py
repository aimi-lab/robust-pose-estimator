# Lie group for so3, SO3 and se3, SE3 spaces
# implementations follow https://ethaneade.com/lie.pdf


import numpy as np


def lie_so3_to_SO3(wvec: np.ndarray = None):

    # check if vector of zeros
    if not wvec.any():
        return np.eye(3)

    theta = (wvec.T @ wvec)**.5

    wvec = wvec / theta if theta > np.finfo(np.float64).eps else wvec
    wmat = lie_hatmap(wvec)

    rmat = np.eye(3) + wmat * (np.sin(theta)/theta) + wmat @ wmat.T *((1-np.cos(theta))/theta**2)

    return rmat


def lie_SO3_to_so3(rmat: np.ndarray = None):

    # check if trace = -1
    if (np.trace(rmat)+1):
        #   rotation by +/- pi, +/- 3pi etc.
        pass
    
    theta = np.arccos((np.trace(rmat)-1)/2)
    theta_term = theta/(2*np.sin(theta)) if theta != 0 else 0.5
    ln_rmat = theta_term * (rmat-rmat.T)

    wvec = np.array([ln_rmat[2, 1]-ln_rmat[1, 2], ln_rmat[0, 2]-ln_rmat[2, 0], ln_rmat[1, 0]-ln_rmat[0, 1]]) / 2

    return wvec


def lie_SE3_to_se3(rmat: np.ndarray = None, tvec: np.ndarray = None):

    wvec = lie_SO3_to_so3(rmat)
    wmat = lie_hatmap(wvec)

    theta = (wvec.T @ wvec)**.5
    a_term = np.sin(theta) / theta
    b_term = (1-np.cos(theta)) / theta**2
    #c_term = (1-a_term) / theta**2

    vmat_inv = np.eye(3) - .5*wmat + 1/theta**2*(1-a_term/(2*b_term))*wmat**2

    uvec = vmat_inv @ tvec

    return wvec, uvec


def lie_se3_to_SE3(wvec: np.ndarray = None, uvec: np.ndarray = None):

    wmat = lie_hatmap(wvec)
    rmat = lie_so3_to_SO3(wvec)

    theta = (wvec.T @ wvec)**.5
    a_term = np.sin(theta) / theta
    b_term = (1-np.cos(theta)) / theta**2
    c_term = (1-a_term) / theta**2
    
    #rmat = np.eye(3) - a_term*wmat + b_term*wmat**2
    vmat = np.eye(3) - b_term*wmat + c_term*wmat**2

    tvec = vmat @ uvec

    return rmat, tvec


def lie_hatmap(wvec: np.ndarray = None):
    """ 
    create hat-map in so(3) from Euler vector in R^3
    
    :param wvec: Euler vector in R^3
    :return: hat-map in so(3)
    """

    assert wvec.size == 3, 'argument must be a 3-vector'

    wmat = np.array([
        [0, -wvec[2], +wvec[1]],
        [+wvec[2], 0, -wvec[0]],
        [-wvec[1], +wvec[0], 0],
    ])

    return wmat


def is_SO3(rmat: np.ndarray, tol=100, eps: float = np.finfo(np.float64).eps):

    assert rmat.size == 9, 'matrix must have 9 elements'
    assert np.linalg.norm(rmat @ rmat.T - np.eye(rmat.shape[0])) < tol * eps, 'R @ R.T must yield identity'
    assert np.linalg.det(rmat @ rmat.T) > 0, 'det(R @ R.T) must be greater than zero'
    
    return True
