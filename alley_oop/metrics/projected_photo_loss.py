import numpy as np
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator, CloughTocher2DInterpolator

from alley_oop.utils.pinhole_transforms import reverse_project, forward_project


def dual_projected_photo_loss(img0, img1, dep0, dep1, rmat, tvec, kmat0, kmat1=None):

    # init values
    kmat1 = kmat0 if kmat1 is None else kmat1

    loss0 = projected_photo_loss(img0, img1, dep1, rmat, tvec, kmat0, kmat1)
    loss1 = projected_photo_loss(img1, img0, dep0, np.linalg.pinv(rmat), -tvec, kmat1, kmat0)

    return loss0 + loss1


def projected_photo_loss(rimg, qimg, dept, rmat, tvec, kmat0, kmat1=None, dbg_opt=False):

    # init values
    kmat1 = kmat0 if kmat1 is None else kmat1

    # channel-wise new image generation (given perspective and depth)
    nimg = np.zeros_like(qimg)
    for i in range(qimg.shape[-1]):
        nimg[..., i] = synthesize_view(qimg[..., i], dept, rmat, tvec, kmat0, kmat1).reshape(qimg.shape[:2])

    # compute loss
    loss = np.mean((rimg - nimg)**2)**.5

    if dbg_opt:
        import imageio
        imageio.imwrite('./photometric_loss_img.png', nimg/nimg.max())
        imageio.imwrite('./photometric_loss_ref.png', rimg/rimg.max())

    return loss


def synthesize_view(img_ch, dept, rmat, tvec, kmat0, kmat1):

    # create 2-D coordinates
    x_coords = np.arange(0, img_ch.shape[1])
    y_coords = np.arange(0, img_ch.shape[0])
    x_mesh, y_mesh = np.meshgrid(x_coords, y_coords)
    ipts = np.vstack([x_mesh.flatten(), y_mesh.flatten(), np.ones(len(x_mesh.flatten()))])

    # back-project coordinates into space
    opts = reverse_project(ipts, kmat1, disp=dept.flatten())

    # rotate, translate and forward-project points
    npts = forward_project(opts, kmat0, rmat, tvec)

    # interpolate RGB values
    interpolator = CloughTocher2DInterpolator(npts[:2].T, img_ch.flatten(), rescale=False)
    imgn = interpolator(ipts[:2].T)

    return imgn