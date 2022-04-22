import numpy
import torch
from typing import Union

from alley_oop.geometry.pinhole_transforms import reverse_project, forward_project
from alley_oop.geometry.pinhole_transforms import create_img_coords_t, create_img_coords_np
from alley_oop.interpol.img_mappings import img_map_scipy, img_map_torch
from alley_oop.utils.lib_handling import get_lib


def synth_view(
        img: Union[numpy.ndarray, torch.Tensor], 
        dept: Union[numpy.ndarray, torch.Tensor], 
        rmat: Union[numpy.ndarray, torch.Tensor], 
        tvec: Union[numpy.ndarray, torch.Tensor], 
        kmat0: Union[numpy.ndarray, torch.Tensor], 
        kmat1: Union[numpy.ndarray, torch.Tensor] = None, 
        mode: str = 'bilinear',
    ) -> Union[numpy.ndarray, torch.Tensor]:

    # determine library given input type
    lib = get_lib(img)

    # init values
    kmat1 = kmat0 if kmat1 is None else kmat1

    # create 2-D coordinates
    if lib == numpy:
        b, _, y, x = dept[:, None, ...].shape if len(dept.shape) == 3 else dept.size()
        ipts = create_img_coords_np(y, x)
    else:
        b, _, y, x = dept.unsqueeze(1).size() if len(dept.shape) == 3 else dept.size()
        ipts = create_img_coords_t(y, x, b)

    # back-project coordinates into space
    opts = reverse_project(ipts, kmat=kmat1, dpth=dept.flatten())

    # rotate, translate and forward-project points
    npts = forward_project(opts, kmat=kmat0, rmat=rmat, tvec=tvec, inhomogenize_opt=True)

    if lib == numpy:
        nimg = img_map_scipy(img=img, ipts=ipts, npts=npts, mode=mode)
    else:
        nimg = img_map_torch(img=img, npts=npts, mode=mode)

    return nimg
