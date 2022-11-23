from core.interpol.synth_view import synth_view, disp_shift_view_synth
from core.utils.lib_handling import get_lib
import torch

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

    # determine library given input type
    lib = get_lib(nimg)

    # compute loss
    rmse = lib.mean((rimg - nimg)**2)**.5

    if dbg_opt:
        import imageio
        imageio.imwrite('./photometric_loss_img.png', nimg/nimg.max())
        imageio.imwrite('./photometric_loss_ref.png', rimg/rimg.max())

    return rmse


def disparity_photo_loss(rimg, qimg, disp, alpha=1.0):
    assert rimg.ndim == 4
    assert qimg.ndim == 4
    rimg_synth = disp_shift_view_synth(qimg, disp, mode='lr')
    photo_diff = ((rimg - rimg_synth)**2).sum(dim=1, keepdims=True)
    return 2-2*torch.sigmoid(alpha*photo_diff)
