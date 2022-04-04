# Lie group for so3, SO3 and se3, SE3 spaces
# implementations follow https://ethaneade.com/lie.pdf


import numpy as np


def lie_alebra2group_rot(wvec: np.ndarray = None):

    assert wvec.size == 3

    wmat = lie_hatmap(wvec)

    rmat = np.exp(wmat)

    return rmat


def lie_group2algebra_rot(rmat: np.ndarray = None):

    assert rmat.size == 9
    
    theta = np.arccos((np.trace(rmat)-1)/2)

    ln_rmat = theta/(2*np.sin(theta)) * (rmat-rmat.T)

    wvec = np.zeros(3)
    wvec[0] = (ln_rmat[2, 1]-ln_rmat[1, 2]) / 2
    wvec[1] = (ln_rmat[0, 2]-ln_rmat[2, 0]) / 2
    wvec[2] = (ln_rmat[1, 0]-ln_rmat[0, 1]) / 2

    return wvec

def lie_group2algebra(rmat: np.ndarray = None, tvec: np.ndarray = None):

    wvec = lie_group2algebra_rot(rmat)
    wmat = lie_hatmap(wvec)

    theta = (wvec.T @ wvec)**.5
    a_term = np.sin(theta) / theta
    b_term = (1-np.cos(theta)) / theta**2
    #c_term = (1-a_term) / theta**2

    vmat_inv = np.eye(3) - .5*wmat + 1/theta**2*(1-a_term/(2*b_term))*wmat**2

    uvec = vmat_inv @ tvec

    return wvec, uvec

def lie_algebra2group(wvec: np.ndarray = None, uvec: np.ndarray = None):

    wmat = lie_hatmap(wvec)
    rmat = np.exp(wmat)

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

    assert wvec.size == 3

    wmat = np.array([
        [0, -wvec[2], +wvec[1]],
        [+wvec[2], 0, -wvec[0]],
        [-wvec[1], +wvec[0], 0],
    ])

    return wmat