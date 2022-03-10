import numpy
import torch

from alley_oop.geometry.pinhole_transforms import forward_project, reverse_project


def feat_homo_rmse(rpts, qpts, hmat):

    # determine library given input type
    lib = numpy if isinstance(hmat, numpy.ndarray) else torch

    # map points from one view to another
    npts = hmat @ rpts

    # compute loss
    rsds = lib.mean((qpts - npts)**2, 0)**.5

    return rsds


def feat_proj_rmse(pts0, pts1, rmat, tvec, kmat0, kmat1, dep0, dep1, method: str = '3d'):

    # determine library given input type
    lib = numpy if isinstance(pts0, numpy.ndarray) else torch

    # project into space
    opt0 = reverse_project(pts0, kmat0, disp=dep0)
    opt1 = reverse_project(pts1, kmat1, disp=dep1)

    # map points to other camera coordinate system
    npts = rmat @ opt1 + tvec

    # compute loss
    if method == '2d':
        ipts = forward_project(npts, kmat0)
        rsds = lib.mean((pts0 - ipts) ** 2, 0) ** .5
    else:
        rsds = lib.mean((opt0 - npts) ** 2, 0) ** .5

    return rsds
