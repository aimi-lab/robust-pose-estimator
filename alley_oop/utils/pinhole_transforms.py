import torch
import numpy as np


def forward_project(opts, kmat, rmat=None, tvec=None):

    # determine library given input type
    lib = np if isinstance(opts, np.ndarray) else torch

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


def reverse_project(ipts, kmat, rmat=None, tvec=None, disp=None, base=float(1)):

    # determine library given input type
    lib = np if isinstance(ipts, np.ndarray) else torch

    # init values potentially missing
    rmat = lib.eye(3) if rmat is None else rmat
    tvec = lib.zeros([3, 1]) if tvec is None else tvec
    ipts = lib.vstack([ipts, lib.ones(ipts.shape[1])]) if ipts.shape[0] == 2 else ipts

    # compose projection matrix
    pmat = compose_projection_matrix(kmat, rmat, tvec)

    # pinhole projection
    opts = lib.linalg.pinv(pmat) @ ipts

    # depth assignment
    if disp is not None:
        flen = [kmat[0][0], kmat[1][1], (kmat[0][0]+kmat[1][1])/2]
        fmat = lib.diag(flen) if isinstance(kmat, np.ndarray) else lib.diag(lib.Tensor(flen))
        opts = base * fmat @ opts[:3] / disp

    return opts


def compose_projection_matrix(kmat, rmat, tvec):

    # determine library given input type
    lib = np if isinstance(kmat, np.ndarray) else torch

    return kmat @ lib.hstack([rmat, tvec])


def decompose_projection_matrix(pmat, scale=False):
    """
    https://www.robots.ox.ac.uk/~vgg/hzbook/code/vgg_multiview/vgg_KR_from_P.m
    """

    # determine library given input type
    lib = np if isinstance(pmat, np.ndarray) else torch

    n = pmat.shape[0] if len(pmat.shape) == 2 else lib.sqrt(pmat.size)
    hmat = pmat.reshape(n, -1)[:, :n]
    rmat, kmat = lib.linalg.qr(hmat, mode='reduced')

    if scale:
        kmat = kmat / kmat[n - 1, n - 1]
        if kmat[0, 0] < 0:
            D = np.diag([-1, -1, *lib.ones(n - 2)])
            kmat = lib.dot(D, kmat)
            rmat = lib.dot(rmat, D)

    tvec = lib.linalg.lstsq(-pmat[:, :n], pmat[:, -1])[0][:, None] if pmat.shape[1] == 4 else lib.zeros(n)

    return rmat, kmat, tvec


def create_img_coords_t(y: int = 720, x: int = 1280, b: int = 1, lib_type: type = np.ndarray):

    # determine library given input type
    lib = np if isinstance(lib_type, np.ndarray) else torch

    # create 2-D coordinates
    x_mesh = lib.linspace(0, x-1, x).repeat(b, y, 1).type_as('int') + .5
    y_mesh = lib.linspace(0, y-1, y).repeat(b, x, 1).transpose(1, 2).type_as('int') + .5
    ipts = lib.vstack([x_mesh.flatten(), y_mesh.flatten(), lib.ones(x_mesh.flatten().shape[0])])

    return ipts


def create_img_coords(y: int = 720, x: int = 1280):

    x_coords = np.arange(0, x)
    y_coords = np.arange(0, y)
    x_mesh, y_mesh = np.meshgrid(x_coords, y_coords)
    ipts = np.vstack([x_mesh.flatten(), y_mesh.flatten(), np.ones(len(x_mesh.flatten()))])

    return ipts
