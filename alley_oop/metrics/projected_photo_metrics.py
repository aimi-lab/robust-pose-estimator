from alley_oop.interpol.synth_view import synth_view
from alley_oop.utils.lib_handling import get_lib_type

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
    lib = get_lib_type(nimg)

    # compute loss
    rmse = lib.mean((rimg - nimg)**2)**.5

    if dbg_opt:
        import imageio
        imageio.imwrite('./photometric_loss_img.png', nimg/nimg.max())
        imageio.imwrite('./photometric_loss_ref.png', rimg/rimg.max())

    return rmse
