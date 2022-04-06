import torch
import numpy as np
from typing import Union


from alley_oop.utils.lib_handling import get_lib_type


def forward_project(
        opts: Union[np.ndarray, torch.Tensor],
        kmat: Union[np.ndarray, torch.Tensor],
        rmat: Union[np.ndarray, torch.Tensor] = None,
        tvec: Union[np.ndarray, torch.Tensor] = None,
                    ):

    # determine library given input type
    lib = get_lib_type(opts)

    # init values potentially missing
    rmat = lib.eye(3) if rmat is None else rmat
    tvec = lib.zeros([3, 1]) if tvec is None else tvec
    opts = lib.vstack([opts, lib.ones(opts.shape[1])]) if opts.shape[0] == 3 else opts

    # compose projection matrix
    pmat = compose_projection_matrix(kmat, rmat, tvec)

    # pinhole projection
    ipts = pmat @ opts

    # inhomogenization
    ipts = ipts[:3] / ipts[-1]

    return ipts


def reverse_project(
        ipts: Union[np.ndarray, torch.Tensor],
        kmat: Union[np.ndarray, torch.Tensor],
        rmat: Union[np.ndarray, torch.Tensor] = None,
        tvec: Union[np.ndarray, torch.Tensor] = None,
        dept: Union[np.ndarray, torch.Tensor] = None,
        disp: Union[np.ndarray, torch.Tensor] = None,
        base: Union[float, int] = 1.,
                    ):

    # determine library given input type
    lib = get_lib_type(ipts)

    # init values potentially missing
    rmat = lib.eye(3) if rmat is None else rmat
    tvec = lib.zeros([3, 1]) if tvec is None else tvec
    ipts = lib.vstack([ipts, lib.ones(ipts.shape[1])]) if ipts.shape[0] == 2 else ipts
    dept = disp2depth(disp=disp, kmat=kmat, base=base) if disp is not None else dept
    dept = lib.ones(ipts.shape[1]) if dept is None else dept

    # pinhole projection
    opts = dept.flatten() * (lib.linalg.inv(kmat) @ ipts)

    # from camera to world coordinates
    opts = rmat @ opts + tvec

    return opts


def compose_projection_matrix(
        kmat: Union[np.ndarray, torch.Tensor],
        rmat: Union[np.ndarray, torch.Tensor] = None,
        tvec: Union[np.ndarray, torch.Tensor] = None,
                             ):

    # determine library given input type
    lib = get_lib_type(kmat)

    return kmat @ lib.hstack([rmat, tvec])


def decompose_projection_matrix(
        pmat: Union[np.ndarray, torch.Tensor],
        scale: bool = True,
                               ):
    """
    https://www.robots.ox.ac.uk/~vgg/hzbook/code/vgg_multiview/vgg_KR_from_P.m
    """

    # determine library given input type
    lib = get_lib_type(pmat)

    n = pmat.shape[0] if len(pmat.shape) == 2 else lib.sqrt(pmat.size)
    hmat = pmat.reshape(n, -1)[:, :n]
    rmat, kmat = decompose_rq(hmat)

    if scale:
        kmat = kmat / kmat[n-1, n-1]
        if kmat[0, 0] < 0:
            dmat = lib.diag([-1, -1, *lib.ones(n - 2)])
            kmat = lib.einsum('ij,jk -> ik', dmat, kmat)
            rmat = lib.einsum('ij,jk -> ik', rmat, dmat)

    tvec = -lib.linalg.lstsq(-pmat[:, :n], pmat[:, -1], rcond=None)[0][:, None] if pmat.shape[1] == 4 else lib.zeros(n)

    return kmat, rmat, tvec

def decompose_rq(hmat:Union[np.ndarray, torch.Tensor]):
    """
    https://github.com/petercorke/machinevision-toolbox-matlab/blob/master/vgg_rq.m
    """

    # determine library given input type
    lib = get_lib_type(hmat)

    hmat = hmat.T
    rmat, kmat = lib.linalg.qr(hmat[::-1, ::-1])    #, mode='reduced'
    rmat = rmat.T
    rmat = rmat[::-1, ::-1]
    kmat = kmat.T
    kmat = kmat[::-1, ::-1]

    if lib.linalg.det(rmat) < 0:
        kmat[:, 0] *= -1
        rmat[0, :] *= -1

    return rmat, kmat


def create_img_coords_t(
        y: int = 720,
        x: int = 1280,
        b: int = 1,
        ref_type: type = torch.Tensor,
                       ):

    # determine library given input type
    lib = get_lib_type(ref_type)

    # create 2-D coordinates
    x_mesh = lib.linspace(0, x-1, x).repeat(b, y, 1).type_as(ref_type) + .5
    y_mesh = lib.linspace(0, y-1, y).repeat(b, x, 1).transpose(1, 2).type_as(ref_type) + .5
    ipts = lib.vstack([x_mesh.flatten(), y_mesh.flatten(), lib.ones(x_mesh.flatten().shape[0])])

    return ipts


def create_img_coords_np(
        y: int = 720,
        x: int = 1280,
                     ):

    x_coords = np.arange(0, x) + .5
    y_coords = np.arange(0, y) + .5
    x_mesh, y_mesh = np.meshgrid(x_coords, y_coords)
    ipts = np.vstack([x_mesh.flatten(), y_mesh.flatten(), np.ones(len(x_mesh.flatten()))])

    return ipts


def disp2depth(
        disp: Union[np.ndarray, torch.Tensor],
        kmat: Union[np.ndarray, torch.Tensor],
        base: Union[float, int] = 1.,
               ):

    flen = (kmat[0][0] + kmat[1][1]) / 2
    dept = (base * flen) / disp.flatten()

    return dept
