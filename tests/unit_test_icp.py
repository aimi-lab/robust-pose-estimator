import unittest
import numpy as np
from pathlib import Path

from alley_oop.pose.icp_estimation import ICPEstimator
import cv2
import torch
from scipy.spatial.transform import Rotation as R
from alley_oop.geometry.lie_3d import lie_SE3_to_se3
from alley_oop.geometry.point_cloud import PointCloud

class RotEstimatorTester(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(RotEstimatorTester, self).__init__(*args, **kwargs)

    def setUp(self):

        np.random.seed(3008)

    def test_estimator(self):

        # load test data
        scale = 16
        disparity = cv2.imread(str(Path.cwd() / 'tests' / 'test_data' / '000000l.pfm'), cv2.IMREAD_UNCHANGED)
        disparity = cv2.resize(disparity, (int(disparity.shape[1]/scale), int(disparity.shape[0]/scale)))/scale
        depth = torch.tensor(4289.756 / disparity/2).double()
        # generate dummy intrinsics and dummy images
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        R_true = torch.tensor(R.from_euler('xyz', (0.0, 0.0, 2), degrees=True).as_matrix()).double()
        t_true = torch.tensor([0, 0.0, 1.0]).double()
        T_true = torch.eye(4).double()
        T_true[:3, :3] = R_true
        T_true[:3, 3] = t_true
        intrinsics = torch.tensor([[1035.3/scale, 0, 596.955/scale],
                                   [0, 1035.3/scale, 488.41/scale],
                                   [0, 0, 1]]).double()
        ref_pcl = PointCloud()
        ref_pcl.from_depth(depth, intrinsics)
        target_pcl = ref_pcl.transform_cpy(T_true)

        estimator = ICPEstimator(depth.shape[:2], intrinsics, Ftol=1e-1, association_mode='projective').to(device)
        with torch.no_grad():
            T, cost = estimator.estimate_lm(depth.to(device), target_pcl.to(device))

        # assertion
        self.assertTrue(np.allclose(T.cpu(), T_true.cpu(), atol=1e-1))
        self.assertTrue(np.allclose(cost.cpu(),0.0, atol=1e-3))

    def test_all(self):

        self.test_estimator()


if __name__ == '__main__':
    unittest.main()
