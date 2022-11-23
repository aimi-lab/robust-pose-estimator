from scipy.spatial import KDTree
from sklearn.decomposition import PCA
import numpy as np
import torch
from typing import Union, Tuple
from core.utils.lib_handling import get_lib


def normals_from_regular_grid(oarr: Union[np.ndarray, torch.Tensor], pad_opt: bool = False) -> Union[np.ndarray, torch.Tensor]:
    """
    compute normal for each point in a regular grid

    #TODO pad normals to be consistent with input dimensions

    """

    assert len(oarr.shape) == 3, 'normal computation from regular grid requires 3 (x, y, z) dimensions'
    lib = get_lib(oarr)

    # compute difference in vertical and horizontal direction
    vdif = (oarr[:-1, :, :] - oarr[1:, :, :])
    hdif = (oarr[:, :-1, :] - oarr[:, 1:, :])

    if pad_opt:
        vdif = lib.vstack((vdif, vdif[-1, :, :][None, :, :]))
        hdif = lib.hstack((hdif, hdif[:, -1, :][:, None, :]))
    else:
        vdif = vdif[:, :-1, :]
        hdif = hdif[:-1, :, :]

    # compute normals
    narr = lib.cross(hdif, vdif)

    # normalize vector length
    norm = lib.sum(narr**2, axis=-1)**.5
    norm[norm == 0.0] = 1.0
    narr /= norm[..., None]

    return narr


def normals_from_pca(opts: np.ndarray, distance: float=10, leafsize: int = 10) -> np.ndarray:
    """
    compute normal for each point in a pointcloud using PCA on its KD-tree neighbours
    """

    kdt = KDTree(opts.T, leafsize=leafsize)
    pca = PCA(n_components=3)

    naxs = []
    for pt in opts.T:

        idxs = kdt.query_ball_point(pt, r=distance)
        apts = opts[:, idxs].T

        if apts.shape[0] > pca.n_components-1:
            pca.fit(apts)
            eigv = pca.components_
            cidx = np.argmin(pca.singular_values_)
            naxs.append(eigv[cidx])
        else:
            naxs.append(np.ones([3]) * float('NaN'))

    naxs = np.array(naxs).T
    angs = get_ray_surfnorm_angle(opts, naxs)

    # flip normals not facing to camera
    naxs[:, angs/np.pi > .5] *= -1

    return naxs


def get_ray_surfnorm_angle(opts: np.ndarray, naxs: np.ndarray, tvec: np.ndarray = None) -> np.ndarray:
    """
    compute radian angles between surface normals and rays
    """

    # get ray vectors from camera pose if provided
    rays = tvec-opts if tvec is not None else -opts

    # matrix contraction
    avec = np.einsum('ji,ji->i', rays, naxs)

    # normalization term
    dnom = np.linalg.norm(rays, axis=0) * np.linalg.norm(naxs, axis=0)

    # nan-safe normalization
    anit = np.divide(avec.astype('float'), dnom.astype('float'), out=np.zeros_like(avec, dtype='float'), where=dnom!=0)

    # compute angles in radian
    angs = np.arccos(anit)

    return angs


def resize_normalmap(normals: torch.tensor, img_shape: Tuple) -> torch.tensor:
    """
    Consistent resizing of normal map. The map can be sparse where undefined normals have value (0,0,0).
    """

    assert normals.ndim == 4
    assert normals.shape[1] == 3

    normals_lowscale = torch.nn.functional.adaptive_avg_pool2d(normals, output_size=img_shape)
    norm = torch.linalg.norm(normals_lowscale, dim=1, keepdim=True)
    norm[norm == 0.0] = 1.0
    normals_lowscale /= norm
    return normals_lowscale
