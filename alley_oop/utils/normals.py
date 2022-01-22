from scipy.spatial import KDTree
from sklearn.decomposition import PCA
import numpy as np


def normals_from_regular_grid(oarr):

    assert len(oarr.shape) == 3, 'normal computation from regular grid requires 3 (x, y, z) dimensions'

    # compute difference in vertical and horizontal direction
    vdif = (oarr[:-1, :, :]-oarr[1:, :, :])[:, :-1, :]
    hdif = (oarr[:, :-1, :]-oarr[:, 1:, :])[:-1, :, :]

    # compute normals
    narr = np.cross(hdif, vdif)

    # normalize vector length
    narr /= np.linalg.norm(narr, axis=-1)[..., np.newaxis]

    return narr


def normals_from_pca(opts, distance:float=10, leafsize:int=10):
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
    aidx = angs/np.pi < 0.5
    naxs[:, aidx] *= -1

    return naxs


def get_ray_surfnorm_angle(opts, naxs, tvec=None):

    # get ray vectors from camera pose if provided
    rays = tvec-opts if tvec is not None else -opts

    # matrix contraction
    avec = np.einsum('ij,ji->i', rays.T, naxs)

    # normalization term
    dnom = np.linalg.norm(rays, axis=0) * np.linalg.norm(naxs, axis=0)

    # nan-safe normalization
    anit = np.divide(avec.astype('float'), dnom.astype('float'), out=np.zeros_like(avec, dtype='float'), where=dnom!=0)

    # compute angles in radian
    angs = np.arccos(anit)

    return angs
