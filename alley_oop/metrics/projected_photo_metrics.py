import numpy as np
import torch
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator, CloughTocher2DInterpolator

from alley_oop.geometry.pinhole_transforms import reverse_project, forward_project
from alley_oop.geometry.pinhole_transforms import create_img_coords_t, create_img_coords_np


def dual_projected_photo_loss(img0, img1, dep0, dep1, rmat, tvec, kmat0, kmat1=None):

    # init values
    kmat1 = kmat0 if kmat1 is None else kmat1

    loss0 = projected_photo_loss(img0, img1, dep1, rmat, tvec, kmat0, kmat1)
    loss1 = projected_photo_loss(img1, img0, dep0, rmat.T, -tvec, kmat1, kmat0)

    return loss0 + loss1


def projected_photo_loss(rimg, qimg, dept, rmat, tvec, kmat0, kmat1=None, dbg_opt=False):

    # init values
    kmat1 = kmat0 if kmat1 is None else kmat1

    # generate view at perspective
    nimg = synth_view(qimg, dept, rmat, tvec, kmat0, kmat1)

    # compute loss
    rmse = np.mean((rimg - nimg)**2)**.5

    if dbg_opt:
        import imageio
        imageio.imwrite('./photometric_loss_img.png', nimg/nimg.max())
        imageio.imwrite('./photometric_loss_ref.png', rimg/rimg.max())

    return rmse


def synth_view(img, dept, rmat, tvec, kmat0, kmat1=None, mode='bilinear'):

    # determine library given input type
    lib = np if isinstance(img, np.ndarray) else torch

    # init values
    kmat1 = kmat0 if kmat1 is None else kmat1

    # create 2-D coordinates
    if lib == np:
        b, _, y, x = dept[:, None, ...].shape if len(dept.shape) == 3 else dept.size()
        ipts = create_img_coords_np(y, x)
    else:
        b, _, y, x = dept.unsqueeze(1).size() if len(dept.shape) == 3 else dept.size()
        ipts = create_img_coords_t(y, x, b, ref_type=img)

    # back-project coordinates into space
    opts = reverse_project(ipts, kmat=kmat1, dept=dept.flatten())

    # rotate, translate and forward-project points
    npts = forward_project(opts, kmat=kmat0, rmat=rmat, tvec=tvec)

    if lib == np:
        nimg = img_map_scipy(img=img, ipts=ipts, npts=npts, mode=mode)
    else:
        nimg = img_map_torch(img=img, npts=npts, mode=mode)

    return nimg


def img_map_scipy(img, ipts, npts, mode='bilinear'):

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


def img_map_torch(img, npts, mode='bilinear'):

    # create new sample grid
    y, x = img.shape[-2:]
    grid = torch.swapaxes(npts[:2].reshape([2, *img.shape[-2:]]).transpose(1, 2), 0, -1) / torch.Tensor([x, y]) * 2 - 1

    # interpolate new image
    nimg = torch.nn.functional.grid_sample(img, grid[None, :], mode=mode, padding_mode='zeros', align_corners=False)

    return nimg
