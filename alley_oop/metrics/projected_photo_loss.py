import numpy as np
import torch
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator, CloughTocher2DInterpolator

from alley_oop.utils.pinhole_transforms import reverse_project, forward_project


def dual_projected_photo_loss(img0, img1, dep0, dep1, rmat, tvec, kmat0, kmat1=None):

    # init values
    kmat1 = kmat0 if kmat1 is None else kmat1

    loss0 = projected_photo_loss(img0, img1, dep1, rmat, tvec, kmat0, kmat1)
    loss1 = projected_photo_loss(img1, img0, dep0, rmat.T, -tvec, kmat1, kmat0)

    return loss0 + loss1


def projected_photo_loss(rimg, qimg, dept, rmat, tvec, kmat0, kmat1=None, dbg_opt=False):

    # init values
    kmat1 = kmat0 if kmat1 is None else kmat1

    try:
        synth_view_torch(qimg, dept, rmat, tvec, kmat0, kmat1)
    except Exception:
        # channel-wise new image generation (given perspective and depth)
        nimg = synth_view_scipy(qimg, dept, rmat, tvec, kmat0, kmat1)

    # compute loss
    rmse = np.mean((rimg - nimg)**2)**.5

    if dbg_opt:
        import imageio
        imageio.imwrite('./photometric_loss_img.png', nimg/nimg.max())
        imageio.imwrite('./photometric_loss_ref.png', rimg/rimg.max())

    return rmse


def synth_view_scipy(img, dept, rmat, tvec, kmat0, kmat1=None, mode='bilinear'):

    # init values
    kmat1 = kmat0 if kmat1 is None else kmat1

    # create 2-D coordinates
    x_coords = np.arange(0, img.shape[1])
    y_coords = np.arange(0, img.shape[0])
    x_mesh, y_mesh = np.meshgrid(x_coords, y_coords)
    ipts = np.vstack([x_mesh.flatten(), y_mesh.flatten(), np.ones(x_mesh.flatten().shape[0])])

    # back-project coordinates into space
    opts = reverse_project(ipts, kmat1, disp=dept.flatten())

    # rotate, translate and forward-project points
    npts = forward_project(opts, kmat0, rmat, tvec)

    # interpolator settings
    if mode == 'bilinear':
        InterpolClass = LinearNDInterpolator 
    elif mode == 'clough':
        InterpolClass = CloughTocher2DInterpolator
    else:
        InterpolClass = NearestNDInterpolator

    nimg = np.zeros_like(img)
    for i in range(img.shape[-1]):
        interpolator = InterpolClass(ipts[:2].T, img[..., i].flatten(), rescale=False)
        nimg[..., i] = interpolator(npts[:2].T).reshape(img.shape[:2])

    return nimg


def synth_view_torch(img, dept, rmat, tvec, kmat0, kmat1=None, mode='bilinear'):

    # init values
    kmat1 = kmat0 if kmat1 is None else kmat1
    b, _, y, x = dept.unsqueeze(1).size() if len(dept.shape) == 3 else dept.size()

    # create 2-D coordinates
    x_mesh = torch.linspace(0, x-1, x).repeat(b, y, 1).type_as(img) + .5
    y_mesh = torch.linspace(0, y-1, y).repeat(b, x, 1).transpose(1, 2).type_as(img) + .5
    ipts = torch.vstack([x_mesh.flatten(), y_mesh.flatten(), torch.ones(x_mesh.flatten().shape[0])])

    # back-project coordinates into space
    opts = reverse_project(ipts, kmat1, disp=dept.flatten())

    # rotate, translate and forward-project points
    npts = forward_project(opts, kmat0, rmat, tvec)

    # create new sample grid
    grid = torch.swapaxes(npts[:2].reshape([2, *img.shape[-2:]]).transpose(1, 2), 0, -1) / torch.Tensor([x, y]) * 2 - 1

    # interpolate new image
    nimg = torch.nn.functional.grid_sample(img, grid[None, :], mode=mode, padding_mode='zeros', align_corners=False)

    return nimg
