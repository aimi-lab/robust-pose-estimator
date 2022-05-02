import unittest
import torch
import imageio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from alley_oop.interpol.gauss_pyramid import GaussPyramid


class GaussPyramidTester(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(GaussPyramidTester, self).__init__(*args, **kwargs)

    def setUp(self):

        # settings
        self.plot_opt = False

        self.ball_img = imageio.imread(Path.cwd() / 'tests' / 'test_data' / 'bball.jpeg')
        self.porc_img = imageio.imread(Path.cwd() / 'tests' / 'test_data' / '000000l.png')

        # convert to b/w
        self.ball_img = np.mean(self.ball_img, axis=-1)[..., None]
        self.porc_img = np.mean(self.porc_img, axis=-1)[..., None]

    def test_gaussian_pyramid(self):

        # create batch of images
        self.t_img = torch.from_numpy(np.stack([self.ball_img, self.porc_img[:180, :180, :]]))

        # align to torch image style dimensions
        self.b_img = torch.swapaxes(self.t_img, 1, -1)

        kernel_size = 5
        kernel_std = 1
        level_num = 2

        gauss_pyramid = GaussPyramid(
            img=self.b_img, 
            kernel_size=kernel_size, 
            kernel_std=kernel_std,
            level_num = level_num,
            )

        pyramid_levels, _ = gauss_pyramid.forward()

        # validate input parameters
        self.assertEqual(gauss_pyramid.gauss_kernel.shape[2], kernel_size, 'Unexpected kernel size')
        self.assertEqual(gauss_pyramid.gauss_kernel.shape[3], kernel_size, 'Unexpected kernel size')
        self.assertEqual(gauss_pyramid._kernel_std, kernel_std, 'Unexpected kernel standard deviation')
        self.assertEqual(len(pyramid_levels)-1, level_num, 'Unexpected level number')

        # validate types
        self.assertTrue(isinstance(pyramid_levels, list), 'Unexpected pyramid type %s'  % type(pyramid_levels))
        self.assertTrue(isinstance(pyramid_levels[0], torch.Tensor), 'Unexpected pyramid level type %s'  % type(pyramid_levels[0]))

        for i in range(1, gauss_pyramid._level_num+1):
            level = gauss_pyramid.levels[i]
            # validate image dimensions
            self.assertEqual(self.b_img.shape[0], gauss_pyramid.levels[i].shape[0], 'Unexpected batch number')
            self.assertEqual(self.b_img.shape[1], gauss_pyramid.levels[i].shape[1], 'Unexpected channel number')
            self.assertEqual(level.shape[2], gauss_pyramid.top_level.shape[2]//(2*i), 'Unexpected height')
            self.assertEqual(level.shape[3], gauss_pyramid.top_level.shape[3]//(2*i), 'Unexpected width')

        if self.plot_opt:
            _, axs = plt.subplots(nrows=self.b_img.shape[0], ncols=gauss_pyramid._level_num+1)
            for i in range(0, gauss_pyramid._level_num+1):
                img_batch_level = gauss_pyramid.levels[i].cpu().numpy()
                for j in range(self.b_img.shape[0]):
                    img = np.swapaxes(img_batch_level[j], axis1=0, axis2=-1)
                    axs[j, i].imshow(img, cmap='gray')
                    axs[j, i].set_title('Level %s' % str(i))
            plt.show()

    def test_intrinsics(self):
        
        intrinsics = torch.eye(3)
        intrinsics[0, 0] *= 100
        intrinsics[1, 1] *= 101
        intrinsics[0, -1] = 480
        intrinsics[1, -1] = 640

        # create batch of images
        self.t_img = torch.from_numpy(np.stack([self.porc_img, self.porc_img]))

        # align to torch image style dimensions
        self.b_img = torch.swapaxes(self.t_img, 1, -1)

        gauss_pyramid = GaussPyramid(
            img=self.b_img, 
            intrinsics=intrinsics,
            )

        # validate input parameters
        self.assertTrue(torch.allclose(gauss_pyramid.top_instrinsics, intrinsics), 'Unexpected top intrinsics')

        _, intrinsics_levels = gauss_pyramid.forward()

        # validate types
        self.assertTrue(isinstance(intrinsics_levels, list), 'Unexpected pyramid type %s'  % type(intrinsics_levels))
        self.assertTrue(isinstance(intrinsics_levels[0], torch.Tensor), 'Unexpected pyramid level type %s'  % type(intrinsics_levels[0]))

        for i in range(1, gauss_pyramid._level_num+1):
            intrinsics_level = gauss_pyramid.intrinsics_levels[i]
            # validate intrinsic values
            self.assertEqual(intrinsics_level.shape, torch.Size([3, 3]), 'Unexpected intrinsics dimensions')
            self.assertEqual(intrinsics_level[0, 0], intrinsics[0, 0]/(2*i), 'Unexpected focal length')
            self.assertEqual(intrinsics_level[1, 1], intrinsics[1, 1]/(2*i), 'Unexpected focal length')
            self.assertEqual(intrinsics_level[0, -1], intrinsics[0, -1]/(2*i), 'Unexpected center')
            self.assertEqual(intrinsics_level[1, -1], intrinsics[1, -1]/(2*i), 'Unexpected center')
            self.assertEqual(intrinsics_level[-1, -1], 1., 'Last element in intrinsics not 1')

    def test_all(self):

        self.test_gaussian_pyramid()
        self.test_intrinsics()


if __name__ == '__main__':
    unittest.main()
