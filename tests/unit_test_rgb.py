import unittest
import numpy as np
from pathlib import Path

from alley_oop.pose.rgb_pose_estimation import RGBPoseEstimator
import cv2
import torch
from scipy.spatial.transform import Rotation as R
from alley_oop.geometry.lie_3d import lie_SE3_to_se3
from alley_oop.interpol.synth_view import synth_view

class RGBEstimatorTester(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(RGBEstimatorTester, self).__init__(*args, **kwargs)

    def setUp(self):

        np.random.seed(3008)

    def test_estimator(self):
        # load test data
        scale = 16
        disparity = cv2.imread(str(Path.cwd() / 'tests' / 'test_data' / '000000l.pfm'), cv2.IMREAD_UNCHANGED)
        h, w = (int(disparity.shape[0]/scale), int(disparity.shape[1]/scale))
        disparity = cv2.resize(disparity, (w, h))/scale
        depth = torch.tensor(4289.756 / disparity/2).double()
        img = torch.tensor(cv2.resize(cv2.imread(str(Path.cwd() / 'tests' / 'test_data' / '000000l.png'), cv2.IMREAD_GRAYSCALE),
                                       (w, h))).float() / 255.0

        # generate dummy intrinsics and dummy images
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        R_true = torch.tensor(R.from_euler('xyz', (1.0, 2.0, 5.0), degrees=True).as_matrix()).double()
        t_true = torch.tensor([0.0, 3.0, 30.0]).double()
        T_true = torch.eye(4).double()
        T_true[:3, :3] = R_true
        T_true[:3, 3] = t_true
        intrinsics = torch.tensor([[1035.3/scale, 0, 596.955/scale],
                                   [0, 1035.3/scale, 488.41/scale],
                                   [0, 0, 1]]).double()

        target_img = synth_view(img.unsqueeze(0).unsqueeze(0), depth.unsqueeze(0).float(), R_true.float(),
                          t_true.unsqueeze(1).float(), intrinsics.float()).squeeze()
        mask = (target_img != 0)
        estimator = RGBPoseEstimator(img.shape[:2], intrinsics).to(device)
        with torch.no_grad():
            T, cost = estimator.estimate_lm(img.double().to(device), depth.to(device), target_img.double().to(device), mask=mask.to(device))

        # assertion
        self.assertTrue(np.allclose(T[:3,:3].cpu(), R_true.cpu(), atol=1e-1))
        self.assertTrue(np.allclose(T[:3,3].cpu(), t_true.cpu(), atol=5))
        self.assertTrue(np.allclose(cost.cpu(),0.0, atol=1e-2))

    def test_all(self):

        self.test_estimator()


if __name__ == '__main__':
    unittest.main()
