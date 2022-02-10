import unittest
import numpy
import torch

from alley_oop.metrics.projected_photo_loss import synth_view_scipy, synth_view_torch


class PhotoLossTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(PhotoLossTest, self).__init__(*args, **kwargs)

    def setUp(self):
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
            self.tvec[0] = 1
            if lib == torch:
                self.rimg = torch.rand([1, 3, 48, 64]).repeat_interleave(10, axis=-2).repeat_interleave(10, axis=-1)
            else:
                self.rimg = numpy.random.rand(48, 64, 3).repeat(10, axis=0).repeat(10, axis=1)
            self.disp = lib.ones([1, 480, 640]) * 100

            # uut
            synth_view_fun = synth_view_torch if lib == torch else synth_view_scipy
            nimg = synth_view_fun(self.rimg, self.disp, self.rmat, self.tvec, kmat0=self.kmat, mode='bilinear')

            # assertion
            sad = lib.sum(lib.abs(self.rimg[..., 100:] - nimg[..., :-100]))
            self.assertTrue(sad < 1, msg='sum of absolute differences too large using %s' % lib)
            ret = lib.allclose(self.rimg[..., 100:], nimg[..., :-100], atol=1e-4)
            self.assertTrue(ret, msg='pixel-wise similarity assertion failed for %s' % lib)

    def test_all(self):
        self.test_synth_view_shift()


if __name__ == '__main__':
    unittest.main()
