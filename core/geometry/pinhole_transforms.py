import torch
from typing import Tuple, Union
from lietorch import SE3, LieGroupParameter
from core.utils.pytorch import skewmat


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


def transform_forward(opts: torch.Tensor, T:Union[SE3, LieGroupParameter]):
    opts_tr = T * opts.permute(0, 2, 1)
    return opts_tr.permute(0, 2, 1)


def transform_backward(grad_out, out, opts_grad, T):
    # custom backward function to enable double backward
    n = grad_out.shape[0]
    grad_out = grad_out.movedim(1, -1).reshape(-1, 1,3)
    out = out.movedim(1, -1).reshape(-1, 3)
    if T.requires_grad:
        # (I | -out_x) \in R^(3x6)
        eye = torch.eye(3, device=out.device, dtype=out.dtype).repeat(grad_out.shape[0], 1, 1)
        grad_T = torch.bmm(grad_out,torch.cat((eye, skewmat(-out)), dim=-1))
        grad_T = grad_T.reshape(n, -1, 6)
    else:
        grad_T = None
    if opts_grad:
        # 3x3 rot matrix
        if isinstance(T, LieGroupParameter):
            T = T.group
        m = grad_out.shape[0]//n//T.shape[1]

        grad_opts = torch.bmm(grad_out,T.matrix().repeat(1,m,1,1).view(-1,4,4)[:, :3, :3])
        grad_opts = grad_opts.reshape(n, -1, 3).permute(0, 2, 1)
    else:
        grad_opts = None
    return grad_opts, grad_T


class Transform(torch.autograd.Function):
    @staticmethod
    def forward(ctx, opts, T):
        with torch.no_grad():
            out = transform_forward(opts, T)
        ctx.save_for_backward(opts, T, out)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        opts, T, out = ctx.saved_tensors
        return transform_backward(grad_out, out, opts.requires_grad, T)


def transform(opts: torch.Tensor, T:Union[SE3, LieGroupParameter], double_backward:bool=False):
    if double_backward:
        return Transform.apply(opts, T)
    else:
        return transform_forward(opts, T)


def reproject(depth: torch.Tensor, intrinsics: torch.Tensor, img_coords: torch.Tensor):
    # transform and project using pinhole camera model
    # pinhole projection
    n = depth.shape[0]
    repr = torch.linalg.inv(intrinsics) @ img_coords.view(3, -1)
    opts = depth.view(n, 1, -1) * repr

    opts = homogeneous(opts)
    return opts


def project(opts: torch.Tensor, intrinsics:torch.tensor, T:Union[SE3, LieGroupParameter]=None, double_backward:bool=False):
    # pinhole projection
    if T is not None:
        opts = transform(opts, T, double_backward=double_backward)
    ipts = torch.bmm(intrinsics, opts)
    # inhomogenization
    depth = torch.clamp(ipts[:, -1], 1e-12, None).unsqueeze(1)
    ipts = torch.cat((ipts[:,:2], torch.ones_like(ipts[:,None,2])), dim=1)
    ipts = ipts / depth
    return ipts


def project2image(
        opts: torch.Tensor,
        intrinsics: torch.Tensor,
        img_shape: Tuple,
        T:Union[SE3, LieGroupParameter] = None
                    ):
    assert len(img_shape) == 2
    if T is None:
        T = SE3.Identity(1)
    ipts = project(opts, intrinsics, T)
    # filter points that are not in the image
    valid = (ipts[:, 1] < img_shape[0]) & (ipts[:, 0] < img_shape[1]) & (ipts[:, 1] >= 0) & (ipts[:, 0] >= 0)
    return ipts, valid
