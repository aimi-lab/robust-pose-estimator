import numpy as np
import torch

from alley_oop.geometry.pinhole_transforms import reverse_project, forward_project
from alley_oop.geometry.pinhole_transforms import create_img_coords_t, create_img_coords_np
from alley_oop.interpol.img_mappings import img_map_scipy, img_map_torch


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
