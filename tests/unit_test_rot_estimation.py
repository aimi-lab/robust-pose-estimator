import unittest
import numpy as np
from pathlib import Path

from alley_oop.pose.rotation_estimation import RotationEstimator, FrameClass
from scipy.spatial.transform import Rotation
import cv2
import torch


class RotEstimatorTester(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(RotEstimatorTester, self).__init__(*args, **kwargs)

    def setUp(self):

        np.random.seed(3008)

    def test_estimator(self):

        # load test image
        test_img = cv2.imread(str(Path.cwd() / 'tests' / 'test_data' / '000000l.png'))
        img1 = torch.tensor(cv2.resize(test_img,(160, 128))).float() / 255.0

        # generate dummy intrinsics and dummy images
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        f = 1200.0
        cx = 79.5
        cy = 63.5
        intrinsics = torch.tensor([[f, 0, cx], [0, f, cy], [0, 0, 1.0]]).float()
        R_true = torch.tensor(Rotation.from_euler('xyz', np.array([1.0,1,5])*np.random.rand(3), degrees=True).as_matrix()).float()

        # opencv has inverted coordinate system
        img2cv = cv2.warpPerspective(img1.numpy(),(intrinsics @ R_true@ torch.linalg.inv(intrinsics)).squeeze().numpy(), (img1.shape[1], img1.shape[0]))
        img2cv = torch.tensor(img2cv)
        mask = (img2cv[...,0] != 0) # need to have a valid mask, otherwise gradients are ill-defined

        img1 = img1.permute(2,0,1).unsqueeze(0)
        img2cv = img2cv.permute(2, 0, 1).unsqueeze(0)
        dummy_depth = torch.zeros((img1.shape[0], 1, img1.shape[2], img1.shape[3]), dtype=img1.dtype)
        frame1 = FrameClass(img1, dummy_depth, intrinsics=intrinsics)
        frame2 = FrameClass(img2cv, dummy_depth, intrinsics=intrinsics)

        with torch.no_grad():
            estimator = RotationEstimator(frame1.shape, intrinsics).to(device)
            R, residuals = estimator.estimate(frame1.to(device), frame2.to(device), mask=mask.to(device))

        # assertion
        self.assertTrue(np.allclose(R.cpu(), R_true, atol=1e-3))

    def test_all(self):

        self.test_estimator()


if __name__ == '__main__':
    unittest.main()
