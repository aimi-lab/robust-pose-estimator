import numpy as np
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator, CloughTocher2DInterpolator

from alley_oop.utils.pinhole import reverse_project, forward_project


def projected_photo_loss(rimg, qimg, rmat, tvec, kmat0, kmat1=None, disp=None, dbg_opt=False):

    # init values
    kmat1 = kmat0 if kmat1 is None else kmat1

    # channel-wise new image generation (given perspective and depth)
    nimg = np.zeros_like(qimg)
    for i in range(qimg.shape[-1]):
        nimg[..., i] = synthesize_view(qimg[..., i], rmat, tvec, kmat0, kmat1, disp).reshape(qimg.shape[:2])

    # compute loss
    loss = np.sum((rimg - nimg)**2)

    if dbg_opt:
        import imageio
        imageio.imwrite('./photometric_loss_img.png', nimg)
        imageio.imwrite('./photometric_loss_ref.png', rimg)

    return loss


def synthesize_view(img_ch, rmat, tvec, kmat0, kmat1, disp):

    # create 2-D coordinates
    x_coords = np.arange(0, img_ch.shape[1])
    y_coords = np.arange(0, img_ch.shape[0])
    x_mesh, y_mesh = np.meshgrid(x_coords, y_coords)
    ipts = np.vstack([x_mesh.flatten(), y_mesh.flatten(), np.ones(len(x_mesh.flatten()))])

    # back-project coordinates
    opts = reverse_project(ipts, kmat1, disp=disp.flatten())

    # rotate, translate and forward-project points
    npts = forward_project(opts, kmat0, np.hstack([rmat, tvec]))

    # interpolate RGB values
    interpolator = CloughTocher2DInterpolator(npts[:2].T, img_ch.flatten(), fill_value=0, rescale=False)
    imgn = interpolator(ipts[:2].T)

    return imgn