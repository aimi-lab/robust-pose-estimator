import numpy as np
from scipy.interpolate import LinearNDInterpolator, CloughTocher2DInterpolator
from scipy import spatial


def surf_interpol(rpts, qpts, method: str = 'bilinear', fill_value: float = 0):

    interpol_funs = create_surf_interpolator(rpts, method=method, fill_value=fill_value)
    interpol_arrs = surf_interpol_pts(qpts, interpol_funs)

    return interpol_arrs


def surf_interpol_pts(qpts, interpol_funs):
    
    interpol_arrs = np.zeros([qpts.shape[0]-2, qpts.shape[1]])
    for i, interpol_fun in enumerate(interpol_funs):
        interpol_arrs[i] = interpol_fun(qpts[0], qpts[1])
    
    return interpol_arrs


def create_surf_interpolator(rpts, method: str = 'bilinear', fill_value: float = 0):
    """ create interpolator (linear or cubic) from reference points """

    interpolators = []
    for i in range(rpts.shape[0]-2):
        tri = spatial.Delaunay(rpts[:2].copy().T)
        InterpolatorClass = LinearNDInterpolator if method.lower() == 'bilinear' else CloughTocher2DInterpolator
        interpolators.append(InterpolatorClass(
            points=tri,
            values=rpts[i+2].copy(),
            rescale=False,
            fill_value=fill_value,
            ))
        
    return interpolators
