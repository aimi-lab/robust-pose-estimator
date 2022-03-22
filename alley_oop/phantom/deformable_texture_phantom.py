import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from alley_oop.geometry.pinhole_transforms import create_img_coords_np
from viewer.view_render import Render
from alley_oop.geometry.camera import PinholeCamera
import open3d
import time
from scipy.spatial.transform import Rotation as R

class DeformableTexturePhantom(object):
    def __init__(self, img, depth, camera:PinholeCamera, n_deformation_nodes: int=1):
        np.random.seed(123456)
        self.camera = camera
        # generate 3D points by backprojection
        ipts = create_img_coords_np(depth.shape[0], depth.shape[1])
        self.pcl_loc = self.camera.project3d(ipts, depth.flatten()).T
        self.pcl_rgb = img.reshape(-1, 3)
        self.affine_t = np.eye(4)
        self.deform_param = 0.0
        self.size = depth.shape
        random_locs = np.arange(len(self.pcl_rgb))
        np.random.shuffle(random_locs)
        self.random_locs = random_locs[:n_deformation_nodes]
        self.random_deform_rot = 20*np.random.rand(1)

    def transform_affine(self, pts=None, rmat=np.eye(3), tvec=np.zeros((3,1)), update=True):
        affine_t = np.eye(4)
        affine_t[:3,:3] = rmat
        affine_t[:3,3] = tvec.squeeze()
        affine_t = affine_t @ self.affine_t
        if update:
            self.affine_t = affine_t
        if pts is None:
            pts = self.pcl_loc
        plane_pts = pts @ affine_t[:3,:3].T  + affine_t[None, :3,3]
        return plane_pts

    def deform(self, pts=None, deformation_param=0.0, update=True):
        # add bumps add random locations of height max deform_param
        # and then smooth using Gaussian kernel
        deformation_param = deformation_param+self.deform_param
        if update:
            self.deform_param = deformation_param
        deformation_depth = np.zeros(self.size)
        deformation_depth[np.unravel_index(self.random_locs, deformation_depth.shape)] = deformation_param*self.size[0]**2
        deformation_depth = gaussian_filter(deformation_depth, sigma=self.size[0]/5)
        if pts is None:
            pts = self.pcl_loc
        ipts = create_img_coords_np(deformation_depth.shape[0], deformation_depth.shape[1])
        deformation_pcl = self.camera.project3d(ipts, deformation_depth.flatten()).T
        # apply small affine transform to deformation such that it is not only in z-direction of the camera
        rmat = R.from_euler('x', self.random_deform_rot, degrees=True).as_matrix()
        tvec = np.array([0, 0, 0])[:, None]
        deformation_pcl = self.transform_affine(deformation_pcl, rmat, tvec, update=False)
        return pts + deformation_pcl

    def add_noise(self, noise_level, update=True):
        raise NotImplementedError

        #query + noise_level * np.random.randn(*query.shape)

    def render(self, apply_deform=True):
        if apply_deform:
            pcl_loc = self.deform(update=False)
            pcl_loc = self.transform_affine(pts=pcl_loc, update=False)
        else:
            pcl_loc = self.pcl_loc
        img, depth = self.camera.render(pcl_loc.T, self.pcl_rgb, self.size)
        return img, depth

    def plot(self, ax=plt.gca(), apply_deform=True):
        img, depth = self.render(apply_deform)
        ax.imshow(img)
        return ax, img

    def animate(self, steps=10, repeats=1):
        pcd = open3d.geometry.PointCloud()
        self.render3d = Render(pcd)
        for i in range(repeats):
            for j in range(steps):
                pcl_loc = self.deform(deformation_param=self.deform_param*(j/steps-1), update=False)
                pcl_loc = self.transform_affine(pts=pcl_loc, update=False)
                pcd.points = open3d.utility.Vector3dVector(pcl_loc)
                pcd.colors = open3d.utility.Vector3dVector(self.pcl_rgb / 255.0)
                _ = self.render3d.render(np.eye(4), pcd)
                time.sleep(0.01)
            for j in range(steps):
                pcl_loc = self.deform(deformation_param=self.deform_param*(-j/steps), update=False)
                pcl_loc = self.transform_affine(pts=pcl_loc, update=False)
                pcd.points = open3d.utility.Vector3dVector(pcl_loc)
                pcd.colors = open3d.utility.Vector3dVector(self.pcl_rgb / 255.0)
                _ = self.render3d.render(np.eye(4), pcd)
                time.sleep(0.01)

    def pts(self, original=False):
        if original:
            return self.pcl_loc
        else:
            pcl_loc = self.deform(update=False)
            pcl_loc = self.transform_affine(pts=pcl_loc, update=False)
            return pcl_loc

import cv2
img = cv2.cvtColor(cv2.resize(cv2.imread('../../tests/test_data/000000l.png'), (640, 480)), cv2.COLOR_BGR2RGB)
disparity = cv2.resize(cv2.imread('../../tests/test_data/000000l.pfm', cv2.IMREAD_UNCHANGED), (640, 480))/2
depth = 2144.878173828125 / disparity
camera = PinholeCamera(np.array([[517.654052734375, 0, 298.4775085449219],
                                         [0, 517.5438232421875, 244.20501708984375],
                                         [0,0,1]]))
plane = DeformableTexturePhantom(img, depth, camera)
plane2 = DeformableTexturePhantom(img, depth, camera)

plane.deform(deformation_param=10.0)
fig, ax = plt.subplots(1,2)
#ax = plane2.plot()
plane.animate(20, 2)
plane.plot(ax[0])
plane2.plot(ax[1])
print('dd')
#plt.legend(['original', 'deformed'])
plt.show()