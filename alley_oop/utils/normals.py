from scipy.spatial import KDTree
from sklearn.decomposition import PCA
import numpy as np


def get_normals(opts, distance:float=10, leafsize:int=10, plot_opt=False):
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

    if plot_opt:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # plot all norms and points
        ax.scatter(*opts, c='k')
        ax.quiver(opts[0], opts[1], opts[2], naxs[0], naxs[1], naxs[2], length=distance/2, normalize=True, color='r')
        ax.scatter(0, 0, 0, 'kx')
        plt.show()

    return naxs


def get_ray_surfnorm_angle(opts, naxs, tvec=None):

    # get ray vectors from camera pose if provided
    rays = tvec-opts if tvec is not None else -opts

    # unit length
    rays = rays / np.linalg.norm(rays)
    opts = opts / np.linalg.norm(opts)

    # matrix contraction
    avec = np.einsum('ij,ji->i', opts.T, naxs)

    # compute angles in radian (omit normalization term for unit vectors)
    angs = np.arccos(avec)

    return angs
