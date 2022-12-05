import torch
from typing import Tuple, Union
from lietorch import SE3, LieGroupParameter


def create_img_coords_t(
        y: int,
        x: int,
        b: int = 1,
        device: torch.device=torch.device('cpu')
        ) -> torch.Tensor:

    # create 2-D coordinates
    x_mesh = torch.linspace(0, x-1, x, device=device).repeat(b, y, 1) + .5
    y_mesh = torch.linspace(0, y-1, y, device=device).repeat(b, x, 1).transpose(1, 2) + .5
    ipts = torch.vstack([x_mesh.flatten(), y_mesh.flatten(), torch.ones(x_mesh.flatten().shape[0], device=device)])

    return ipts


def homogeneous(opts: torch.Tensor):
    n = opts.shape[0]
    opts = torch.cat((opts, torch.ones((n, 1, opts.shape[-1]), device=opts.device, dtype=opts.dtype)), dim=1)
    return opts


def transform(opts: torch.Tensor, T:Union[SE3, LieGroupParameter]):
    if (isinstance(T, SE3) & (len(T.shape) == 1)) | (isinstance(T, LieGroupParameter) & (len(T.shape) == 2)):
        T = T[:, None, :]  # broadcast dim
    opts_tr = T * opts.permute(0, 2, 1)
    return opts_tr.permute(0, 2, 1)


def reproject(depth: torch.Tensor, intrinsics: torch.Tensor, img_coords: torch.Tensor):
    # transform and project using pinhole camera model
    # pinhole projection
    n = depth.shape[0]
    repr = torch.linalg.inv(intrinsics) @ img_coords.view(3, -1)
    opts = depth.view(n, 1, -1) * repr

    opts = homogeneous(opts)
    return opts


def project(opts: torch.Tensor, T:Union[SE3, LieGroupParameter], intrinsics:torch.tensor):
    # pinhole projection
    opts = transform(opts, T)
    ipts = torch.bmm(intrinsics, opts)
    # inhomogenization
    ipts = ipts[:, :3] / torch.clamp(ipts[:, -1], 1e-12, None).unsqueeze(1)
    return ipts[:, :2]


def project2image(
        opts: torch.Tensor,
        intrinsics: torch.Tensor,
        img_shape: Tuple,
        T:Union[SE3, LieGroupParameter] = None
                    ):
    assert len(img_shape) == 2
    if T is None:
        T = SE3.Identity(1)
    ipts = project(opts, T, intrinsics)
    # filter points that are not in the image
    valid = (ipts[:, 1] < img_shape[0]) & (ipts[:, 0] < img_shape[1]) & (ipts[:, 1] >= 0) & (ipts[:, 0] >= 0)
    return ipts, valid
