import unittest
import numpy
import torch
import matplotlib.pyplot as plt

from core.interpol.synth_view import synth_view


class PhotoLossTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(PhotoLossTest, self).__init__(*args, **kwargs)

    def setUp(self):

        self.plot_opt = False

        self.kmat = numpy.eye(3)
        self.rmat = numpy.eye(3)
        self.tvec = numpy.zeros([3, 1])

        self.rimg = numpy.ones([480, 640, 3])
        self.disp = numpy.ones([480, 640])

    def test_synth_view_shift(self):

        for lib in [numpy, torch]:

            # set inputs
            self.kmat = lib.eye(3)
            self.rmat = lib.eye(3)
            self.tvec = lib.zeros([3, 1])
            self.sgap = 100
            self.disp = lib.ones([1, 480, 640]) * self.sgap
            self.tvec[0] = 1
            if lib == torch:
                self.rimg = torch.rand([1, 3, 48, 64]).repeat_interleave(10, axis=-2).repeat_interleave(10, axis=-1)
            else:
                self.rimg = numpy.random.rand(48, 64, 3).repeat(10, axis=0).repeat(10, axis=1)

            # uut
            self.nimg = synth_view(self.rimg, 1./self.disp, rmat=self.rmat, tvec=self.tvec, kmat0=self.kmat, mode='bilinear')

            # plot result
            if self.plot_opt:
                self.plot_img_comparison()

            # assertion
            sad = lib.sum(lib.abs(self.rimg[..., self.sgap:] - self.nimg[..., :-self.sgap]))
            ret = lib.allclose(self.rimg[..., self.sgap:], self.nimg[..., :-self.sgap], atol=1e-4)

            self.assertTrue(sad < 1, msg='sum of absolute differences too large using %s' % lib)
            self.assertTrue(ret, msg='pixel-wise similarity assertion failed for %s' % lib)

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
        
        self.test_synth_view_shift()


if __name__ == '__main__':
    unittest.main()
