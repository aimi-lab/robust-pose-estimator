import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter


class DeformablePhantom(object):
    def __init__(self, size: tuple=(10,10), scale=1.0, n_deformation_nodes=1):
        # generate 3D points that lie on xy-plane
        xx, yy = np.meshgrid(range(size[0]), range(size[1]))
        self.plane_pts = scale*np.column_stack((xx.flatten(),yy.flatten(), np.zeros_like(xx.flatten()))).T
        self.affine_t = np.eye(4)
        self.size = size
        self.deform_param = 0.0
        random_locs = np.arange(len(self.plane_pts))
        np.random.shuffle(random_locs)
        self.random_locs = random_locs[:n_deformation_nodes]

    def transform_affine(self, pts=None, rmat=np.eye(3), tvec=np.zeros((3,1)), update=True):
        affine_t = np.eye(4)
        affine_t[:3,:3] = rmat
        affine_t[:3,3] = tvec.squeeze()
        affine_t = affine_t @ self.affine_t
        if update:
            self.affine_t = affine_t
        if pts is None:
            pts = self.plane_pts
        plane_pts = affine_t[:3,:3] @ pts + affine_t[:3,3,None]
        return plane_pts

    def deform(self, pts=None, deformation_param=0.0, update=True):
        # add a bumps add random locations of height max deform_param at the center of the plane
        # and then smooth using Gaussian kernel
        deformation_param = deformation_param+self.deform_param
        if update:
            self.deform_param = deformation_param
        plane_pts_img = np.zeros(self.size)
        plane_pts_img[np.unravel_index(self.random_locs, plane_pts_img.shape)] = deformation_param
        plane_pts_img = gaussian_filter(plane_pts_img, sigma=self.size[0]/2)
        if pts is None:
            pts = self.plane_pts
        plane_pts_deformed = pts.copy()
        plane_pts_deformed[2] = plane_pts_img.reshape(len(plane_pts_deformed[2]))
        return plane_pts_deformed

    def add_noise(self, noise_level, update=True):
        raise NotImplementedError

        #query + noise_level * np.random.randn(*query.shape)

    def plot(self, ax=None):
        if ax is None:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
        plane_pts = self.deform(update=False)
        plane_pts = self.transform_affine(pts=plane_pts, update=False)
        ax.scatter(plane_pts[0,:],plane_pts[1,:], plane_pts[2,:])

        return ax

    def pts(self, original=False):
        if original:
            return self.plane_pts
        else:
            plane_pts = self.deform(update=False)
            plane_pts = self.transform_affine(pts=plane_pts, update=False)
            return plane_pts

plane = DeformablePhantom()
plane2 = DeformablePhantom()
from scipy.spatial.transform import Rotation as R
R_true = R.from_euler('x', 20, degrees=True).as_matrix()
t_true = np.array([1,1,1])[:,None]
plane.transform_affine(rmat=R_true, tvec=t_true)
plane.deform(deformation_param=100.0)
ax = plane2.plot()
plane.plot(ax)
plt.legend(['original', 'deformed'])
plt.show()