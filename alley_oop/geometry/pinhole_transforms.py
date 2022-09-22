import torch
import numpy as np
from typing import Union, Tuple


from alley_oop.utils.lib_handling import get_lib


def forward_project(
        opts: Union[np.ndarray, torch.Tensor],
        kmat: Union[np.ndarray, torch.Tensor],
        rmat: Union[np.ndarray, torch.Tensor] = None,
        tvec: Union[np.ndarray, torch.Tensor] = None,
        inhomogenize_opt: bool = True
                    ):

    # determine library given input type
    lib = get_lib(opts)

    # init values potentially missing
    if lib == torch:
        rmat = torch.eye(3, device=opts.device) if rmat is None else rmat
        tvec = torch.zeros([3, 1], device=opts.device) if tvec is None else tvec
        opts = torch.vstack([opts, lib.ones(opts.shape[1], device=opts.device)]) if opts.shape[0] == 3 else opts
    else:
        rmat = lib.eye(3) if rmat is None else rmat
        tvec = lib.zeros([3, 1]) if tvec is None else tvec
        opts = lib.vstack([opts, lib.ones(opts.shape[1])]) if opts.shape[0] == 3 else opts

    # compose projection matrix
    pmat = compose_projection_matrix(kmat, rmat, tvec)

    # pinhole projection
    ipts = pmat @ opts

    # inhomogenization
    ipts = ipts[:3] / ipts[-1] if inhomogenize_opt else ipts

    return ipts


def forward_project2image(
        opts: Union[np.ndarray, torch.Tensor],
        kmat: Union[np.ndarray, torch.Tensor],
        img_shape: Tuple,
        rmat: Union[np.ndarray, torch.Tensor] = None,
        tvec: Union[np.ndarray, torch.Tensor] = None,
                    ):
    assert len(img_shape) == 2
    ipts = forward_project(opts, kmat, rmat, tvec)
    # filter points that are not in the image
    valid = (ipts[1] < img_shape[0]) & (ipts[0] < img_shape[1]) & (ipts[1] >= 0) & (ipts[0] >= 0)
    return ipts, valid


def reverse_project(
        ipts: Union[np.ndarray, torch.Tensor],
        kmat: Union[np.ndarray, torch.Tensor],
        rmat: Union[np.ndarray, torch.Tensor] = None,
        tvec: Union[np.ndarray, torch.Tensor] = None,
        dpth: Union[np.ndarray, torch.Tensor] = None,
        disp: Union[np.ndarray, torch.Tensor] = None,
        base: Union[float, int] = 1.,
                    ):

    # determine library given input type
    lib = get_lib(ipts)

    # init values potentially missing
    rmat = lib.eye(3) if rmat is None else rmat
    tvec = lib.zeros([3, 1]) if tvec is None else tvec
    ipts = lib.vstack([ipts, lib.ones(ipts.shape[1])]) if ipts.shape[0] == 2 else ipts
    dpth = disp2depth(disp=disp, kmat=kmat, base=base) if disp is not None else dpth
    dpth = lib.ones(ipts.shape[1]) if dpth is None else dpth

    # pinhole projection
    opts = dpth.flatten() * (lib.linalg.inv(kmat) @ ipts)

    # from camera to world coordinates
    opts = rmat @ opts + tvec

    return opts


def compose_projection_matrix(
        kmat: Union[np.ndarray, torch.Tensor],
        rmat: Union[np.ndarray, torch.Tensor] = None,
        tvec: Union[np.ndarray, torch.Tensor] = None,
                             ):

    # determine library given input type
    lib = get_lib(kmat)

    return kmat @ lib.hstack([rmat, tvec])


def decompose_projection_matrix(
        pmat: Union[np.ndarray, torch.Tensor],
        scale: bool = True,
                               ):
    """
    https://www.robots.ox.ac.uk/~vgg/hzbook/code/vgg_multiview/vgg_KR_from_P.m
    """

    # determine library given input type
    lib = get_lib(pmat)

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
    lib = get_lib(hmat)

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
        device: torch.device=torch.device('cpu')
        ) -> torch.Tensor:

    # create 2-D coordinates
    x_mesh = torch.linspace(0, x-1, x, device=device).repeat(b, y, 1) + .5
    y_mesh = torch.linspace(0, y-1, y, device=device).repeat(b, x, 1).transpose(1, 2) + .5
    ipts = torch.vstack([x_mesh.flatten(), y_mesh.flatten(), torch.ones(x_mesh.flatten().shape[0], device=device)])

    return ipts


def create_img_coords_np(
        y: int = 720,
        x: int = 1280,
        step: int = 1
        ) -> np.ndarray:

    x_coords = np.arange(0, x, step) + .5
    y_coords = np.arange(0, y, step) + .5
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


def inv_transform(mat:torch.Tensor):
    mat_inv = torch.eye(4, device=mat.device, dtype=mat.dtype)
    mat_inv[:3, :3] = mat[:3, :3].T
    mat_inv[:3, 3] = -mat[:3, :3].T @ mat[:3, 3]
    return mat_inv


def homogenous(opts: torch.Tensor):
    n = opts.shape[0]
    opts = torch.cat((opts, torch.ones((n, 1, opts.shape[-1]), device=opts.device, dtype=opts.dtype)), dim=1)
    return opts


def transform(opts: torch.Tensor, T:torch.tensor):
    return torch.bmm(T, opts)


def reproject(depth: torch.Tensor, intrinsics: torch.Tensor, img_coords: torch.Tensor):
    # transform and project using pinhole camera model
    # pinhole projection
    n = depth.shape[0]
    repr = torch.linalg.inv(intrinsics) @ img_coords.view(3, -1)
    opts = depth.view(n, 1, -1) * repr

    opts = homogenous(opts)
    return opts


def project(opts: torch.Tensor, pmat:torch.tensor, kmat:torch.tensor):
    p = kmat @ pmat[:, :3]
    # pinhole projection
    ipts = torch.bmm(p, homogenous(opts))
    # inhomogenization
    ipts = ipts[:, :3] / (ipts[:, -1].unsqueeze(1) + 1e-12)
    return ipts[:, :2]
