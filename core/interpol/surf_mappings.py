import numpy as np
import torch
from scipy.interpolate import LinearNDInterpolator, CloughTocher2DInterpolator
from scipy import spatial


def surf_interpol(rpts, qpts, method: str = 'bilinear', fill_val: float = 0):

    interpol_funs = create_surf_interpolator(rpts, method=method, fill_val=fill_val)
    interpol_arrs = surf_interpol_pts(qpts, interpol_funs)

    return interpol_arrs


def surf_interpol_pts(pts, interpol_funs):
    
    interpol_arrs = np.zeros([pts.shape[0] - 2, pts.shape[1]])
    for i, interpol_fun in enumerate(interpol_funs):
        interpol_arrs[i] = interpol_fun(pts[0], pts[1])
    
    return interpol_arrs


def create_surf_interpolator(pts, method: str = 'bilinear', fill_val: float = 0):
    """ create interpolator (linear or cubic) from reference points """

    tri, face = get_delaunay_triangles(pts[:2].copy().T)
    InterpolatorClass = LinearNDInterpolator if method.lower() == 'bilinear' else CloughTocher2DInterpolator

    interpolators = []
    # iterate through dimensions except x, y
    for i in range(pts.shape[0] - 2):
        interpolators.append(InterpolatorClass(
            points=tri,
            values=pts[i + 2].copy(),
            rescale=False,
            fill_value=fill_val,
            ))
        
    return interpolators


def get_delaunay_triangles(pts):

    assert pts.shape[0] > 2, 'minimum number of points at dim 0 is 3'

    # determine library given input type
    lib = np if isinstance(pts, np.ndarray) else torch

    if lib == torch:
        pts = pts.cpu().numpy()

    tri = spatial.Delaunay(pts, qhull_options='QJ')

    if lib == torch:
        face = torch.from_numpy(tri.simplices)
        face = face.t().contiguous().to(pts.device, torch.long)
    else:
        face = tri.simplices

    return tri, face


