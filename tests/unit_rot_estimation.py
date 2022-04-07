import unittest
import numpy as np

from alley_oop.pose.rotation_estimation import RotationEstimator
from scipy.spatial.transform import Rotation
import cv2
import torch



class Lie3DTester(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(Lie3DTester, self).__init__(*args, **kwargs)

    def setUp(self):

        np.random.seed(3008)

    def test_estimator(self):
        # generate dummy intrinsics and dummy images
        f = 1200.0
        cx = 79.5
        cy = 63.5
        intrinsics = torch.tensor([[f, 0, cx], [0, f, cy], [0, 0, 1.0]]).float()
        R_true = torch.tensor(Rotation.from_euler('xyz', 2*np.random.randn(3), degrees=True).as_matrix()).float()
        img1 = torch.tensor(cv2.resize(cv2.imread('test_data/000000l.png', cv2.IMREAD_GRAYSCALE),
                                       (160, 128))).float() / 255.0

        img2cv = RotationEstimator._warp_img(img1, R_true, intrinsics)

        estimator = RotationEstimator(img1.shape, intrinsics)

        R, residuals, warped_img = estimator.estimate(img2cv, img1)

        # assertion
        self.assertTrue(np.allclose(R, R_true, atol=10e-2))

        self.assertTrue(np.allclose(img2cv, warped_img, atol=10e-3))


    def test_all(self):

        self.test_estimator()


if __name__ == '__main__':
    unittest.main()
