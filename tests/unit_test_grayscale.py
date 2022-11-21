import unittest
import torch
import imageio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from core.utils.rgb2gray import rgb2gray_t


class GrayscaleTester(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(GrayscaleTester, self).__init__(*args, **kwargs)

    def setUp(self):

        # settings
        self.plot_opt = False

        self.ball_img = imageio.imread(Path.cwd() / 'tests' / 'test_data' / 'bball.jpeg')
        self.porc_img = imageio.imread(Path.cwd() / 'tests' / 'test_data' / '000000l.png')

    def test_rgb2gry(self):

        # create batch of images
        self.t_img = torch.as_tensor(np.stack([self.ball_img, self.porc_img[:180, :180, :]]), dtype=torch.float32)

        # align to torch image style dimensions
        self.b_img = torch.swapaxes(self.t_img, 1, -1)

        # trigger exception
        try:
            gray = rgb2gray_t(self.b_img)
        except AttributeError:
            pass

        # convert using custom values
        gray = rgb2gray_t(self.b_img, vec=(0, 0, 0), ax0=1)
        self.assertEqual(torch.sum(gray), 0)
        gray = rgb2gray_t(self.b_img, vec=(1/3, 1/3, 1/3), ax0=1)
        self.assertTrue(torch.allclose(gray[:, 0, ...], torch.mean(self.b_img, 1)))

        # convert to grayscale as intended
        gray = rgb2gray_t(self.b_img, ax0=1)

        self.assertEqual(len(gray.shape), len(self.b_img.shape), 'Output dimension mismatch')
        self.assertEqual(gray.shape[1], 1, 'Grayscale channel amounts to %s ' % gray.shape[1])

        if self.plot_opt:
            _, axs = plt.subplots(nrows=1, ncols=self.b_img.shape[0])
            for i in range(self.b_img.shape[0]):
                img = np.swapaxes(gray[i], axis1=0, axis2=-1)
                axs[i].imshow(img, cmap='gray')
                axs[i].set_title('batch #%s' % str(i))
            plt.show()


    def test_all(self):

        self.test_rgb2gry()


if __name__ == '__main__':
    unittest.main()
