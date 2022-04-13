import unittest
import numpy as np
import imageio
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from alley_oop.geometry.pinhole_transforms import forward_project, reverse_project, compose_projection_matrix, decompose_projection_matrix, create_img_coords_np
from alley_oop.geometry.quaternions import quat2rmat, euler2quat
from alley_oop.interpol.img_mappings import img_map_scipy


class PinholeTransformTester(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(PinholeTransformTester, self).__init__(*args, **kwargs)

    def setUp(self):

        # settings
        self.plot_opt = True

        # intrinsics
        self.resolution = (180, 180)
        self.kmat = np.diag([150, 150, 1])
        self.kmat[0, -1] = self.resolution[1]//2
        self.kmat[1, -1] = self.resolution[0]//2
        self.ipts = create_img_coords_np(*self.resolution)

        # extrinsics
        self.rmat = np.eye(3)
        self.tvec = np.zeros([3, 1])
        self.zpts = 0.1 * np.random.randn(np.multiply(*self.resolution))[np.newaxis] + 1
        self.ball = imageio.imread(Path.cwd() / 'tests' / 'test_data' / 'bball.jpeg')

    def test_ortho_plane_projection(self):

        self.setUp()

        for plane_dist in range(1, int(1e5), 10):

            zpts = plane_dist * np.ones(np.multiply(*self.resolution))[np.newaxis]

            opts = reverse_project(self.ipts, self.kmat, self.rmat, self.tvec, disp=zpts)

            npts = forward_project(opts, self.kmat, self.rmat, self.tvec, inhomogenize_opt=True)

            self.assertTrue(np.allclose(self.ipts, npts))

    def test_translated_projection(self):

        self.setUp()

        dist = 100
        dept = dist * np.ones(np.multiply(*self.resolution))[np.newaxis]

        dofs = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])

        for dof in dofs:

            self.tvec = dof[:3][:, None]

            opts = reverse_project(self.ipts, self.kmat, rmat=np.eye(3), tvec=np.zeros([3, 1]), depth=dept)

            npts = forward_project(opts, kmat=self.kmat, rmat=self.rmat, tvec=self.tvec)

            disp = self.tvec * self.kmat[0, 0] / dist
            ret_val = np.allclose(self.ipts+disp, npts)
            self.assertTrue(ret_val)

    def test_rotated_projection(self):

        self.setUp()

        dist = 100
        anch = np.array([0, 0, dist])
        rmat = quat2rmat(euler2quat(0, np.pi/4, 0))
        npos = np.abs(anch - rmat @ anch)

        dept = dist * np.ones(np.multiply(*self.resolution))[np.newaxis]
        opts = reverse_project(self.ipts, self.kmat, rmat=np.eye(3), tvec=np.zeros([3, 1]), depth=dept)

        npts = forward_project(opts, kmat=self.kmat, rmat=rmat.T, tvec=npos[:, None])

        nimg = img_map_scipy(self.ball, ipts=self.ipts, npts=npts)

        if self.plot_opt:
            plt.figure()
            plt.imshow(nimg)
            plt.show()

    def test_KR_decomposition(self):

        # intrinsics
        self.kmat = np.diag([525.8345947265625, 525.7257690429688, 1])
        self.kmat[0, -1] = 320
        self.kmat[1, -1] = 240
        self.tvec = np.array([[1], [2], [.5]])
        self.rmat = quat2rmat(euler2quat(np.pi/4, 0, 1))

        pmat = compose_projection_matrix(self.kmat, self.rmat, self.tvec)

        kmat, rmat, tvec = decompose_projection_matrix(pmat, scale=True)

        # assertion
        self.assertTrue(np.allclose(self.kmat, abs(kmat)))

        # todo: consider sign results and make sure tvec is correct

    def plot_img_comparison(self):

            if isinstance(self.rimg, torch.Tensor):
                self.rimg = self.rimg.detach().cpu().numpy()

            if isinstance(self.nimg, torch.Tensor):
                self.nimg = self.nimg.detach().cpu().numpy()

            _, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
            axs[0].imshow(self.rimg)
            axs[0].set_title('before transformation')
            axs[1].imshow(self.nimg)
            axs[1].set_title('after transformation')
            plt.show()

    def test_all(self):

        self.test_KR_decomposition()


if __name__ == '__main__':
    unittest.main()
