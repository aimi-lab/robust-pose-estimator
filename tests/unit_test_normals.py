from os import stat
import unittest
from pathlib import Path
import numpy as np

from alley_oop.utils.normals import normals_from_pca, normals_from_regular_grid


class NormalsTester(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(NormalsTester, self).__init__(*args, **kwargs)

    def setUp(self):
        
        self.data_path = Path.cwd() / 'tests' / 'test_data'
        self.name_list = sorted((self.data_path).rglob('*rgbd.npz'))

        npz_obj = np.load(self.name_list[0])
        self.rgbd_init = npz_obj[npz_obj.files[0]]

        # create plane
        from alley_oop.utils.pinhole_transforms import create_img_coords
        ipts = create_img_coords(480, 640)[:2]
        self.wall_init = np.vstack([ipts, np.repeat(np.arange(480), 640)-1000])
        self.wall_init = self.wall_init.T.reshape(480, 640, 3)
        self.wall_init /= 100

    def test_normals_from_regular_grid(self, plot_opt=True):

        oarr = self.rgbd_init[..., :3]
        oarr = self.wall_init

        narr = normals_from_regular_grid(oarr)

        naxs = narr[1:, 1:, :].reshape(-1, 3).T
        opts = oarr[1:-1, 1:-1, :].reshape(-1, 3).T

        if plot_opt: self.plot_normals(naxs[:, ::1000], opts[:, ::1000])

    def test_normals_from_pca(self, plot_opt=False):
        
        # var init
        opts = self.rgbd_init[..., :3].reshape(-1, 3)[::1000, :].T
        distance = 10
        leafsize = 10

        # normals computation based on PCA
        naxs = normals_from_pca(opts, distance=distance, leafsize=leafsize)
        
        # plot
        if plot_opt: self.plot_normals(naxs, opts)
    
    @staticmethod
    def plot_normals(naxs, opts):
            
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = plt.axes(projection='3d')

            # plot all norms and points
            ax.scatter(*opts, c='k', label='object points')
            ax.quiver(opts[0], opts[1], opts[2], naxs[0], naxs[1], naxs[2], length=5, normalize=True, color='r')
            ax.scatter(0, 0, 0, 'kx', label='camera center')

            x_limits = ax.get_xlim3d()
            y_limits = ax.get_ylim3d()
            z_limits = ax.get_zlim3d()

            x_range = abs(x_limits[1] - x_limits[0])
            x_middle = np.mean(x_limits)
            y_range = abs(y_limits[1] - y_limits[0])
            y_middle = np.mean(y_limits)
            z_range = abs(z_limits[1] - z_limits[0])
            z_middle = np.mean(z_limits)

            # The plot bounding box is a sphere in the sense of the infinity
            # norm, hence I call half the max range the plot radius.
            plot_radius = 0.5*max([x_range, y_range, z_range])

            ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
            ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
            ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

            plt.show()


    def test_all(self):

        self.test_normals_from_pca()


if __name__ == '__main__':
    unittest.main()
