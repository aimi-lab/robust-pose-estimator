import numpy
import torch
from typing import Union

from core.geometry.pinhole_transforms import reverse_project, forward_project
from core.geometry.pinhole_transforms import create_img_coords_t, create_img_coords_np
from core.interpol.img_mappings import img_map_scipy, img_map_torch
from core.utils.lib_handling import get_lib


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


def disp_shift_view_synth(im, disp, mode):
    """ synthesize 2nd view from one RGB viewpoint image and corresponding disparities """

    # add color channel at dim 1 if necessary
    if im.dim() == 3:
        im = im.reshape(im.shape[0], 1, im.shape[1], im.shape[2])
    N, C, H, W = im.shape

    i_base = torch.linspace(0, H - 1, H).repeat(N, W, 1).transpose(1, 2).type_as(im) + .5
    j_base = torch.linspace(0, W - 1, W).repeat(N, H, 1).type_as(im) + .5
    i_base = i_base.to(disp.device)
    j_base = j_base.to(disp.device)

    # apply shift in j direction
    if mode == 'lr':
        j_new = j_base + disp
    elif mode == 'rl':
        j_new = j_base - disp

    # normalize with image height and width
    i_base = 2 * i_base / H - 1
    j_new = 2 * j_new / W - 1
    j_new = j_new.clamp(-1, 1)

    # create grid
    grid = torch.stack((j_new, i_base), dim=3)  # reversed, because (x, y) -> (j, i)

    # sample new image
    im_out = torch.nn.functional.grid_sample(im, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

    return im_out
