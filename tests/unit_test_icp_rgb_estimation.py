import unittest
import numpy as np
from pathlib import Path

from core.pose.icp_rgb_pose_estimation import RGBICPPoseEstimator, FrameClass
from core.fusion.surfel_map import SurfelMap
import cv2
import torch
from scipy.spatial.transform import Rotation as R
from core.utils.pfm_handler import load_pfm
from core.geometry.lie_3d import lie_se3_to_SE3

class RGBICPPoseEstimatorTester(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(RGBICPPoseEstimatorTester, self).__init__(*args, **kwargs)

    def setUp(self):
        # load test data
        scale = 4
        depth_scale = 80
        disparity, _ = load_pfm(str(Path.cwd() / 'tests' / 'test_data' / '0006.pfm'))
        h, w = (int(disparity.shape[0] / scale), int(disparity.shape[1] / scale))
        disparity = cv2.resize(disparity, (w, h))
        # background is very far, make it appear closer
        rand_background = 17 + 6 * np.random.rand(int((disparity < 20).sum()))
        disparity[disparity < 20] = rand_background
        depth = torch.tensor(1050.0 / disparity) / depth_scale
        img = torch.tensor(
            cv2.cvtColor(cv2.resize(cv2.imread(str(Path.cwd() / 'tests' / 'test_data' / '0006.png')),
                                    (w, h)), cv2.COLOR_RGB2BGR)).float() / 255.0

        # generate dummy intrinsics and dummy images
        R_true = torch.tensor(R.from_euler('xyz', (0.0, 0.5, 1.0), degrees=True).as_matrix())
        t_true = torch.tensor([0.0, 0.0, 1.0]) / depth_scale
        self.T_true = torch.eye(4)
        self.T_true[:3, :3] = R_true
        self.T_true[:3, 3] = t_true
        self.intrinsics = torch.tensor([[1050.0 / scale, 0, 479.5 / scale],
                                        [0, 1050.0 / scale, 269.5 / scale],
                                        [0, 0, 1]])
        img = img.permute(2, 0, 1).unsqueeze(0)
        self.ref_frame = FrameClass(img, depth.unsqueeze(0).unsqueeze(0), intrinsics=self.intrinsics)

        self.ref_pcl = SurfelMap(frame=self.ref_frame, kmat=self.intrinsics)

        self.target_pcl = self.ref_pcl.transform_cpy(torch.linalg.inv(self.T_true))
        self.target_frame = self.target_pcl.render(self.intrinsics)
        np.random.seed(3008)

    def test_icp(self):
        with torch.inference_mode():
            estimator = RGBICPPoseEstimator(self.ref_frame.shape, self.intrinsics, icp_weight=10000000.0, n_iter=100, dist_thr=0.5, association_mode='euclidean')
            T_se3, *_ = estimator.estimate_gn(self.ref_frame, self.target_frame,
                                              self.target_pcl)
            T = lie_se3_to_SE3(T_se3.squeeze())

        self.assertTrue(np.allclose(T[:3, :3], self.T_true[:3, :3], atol=1e-2))
        self.assertTrue(np.allclose(T[:3, 3], self.T_true[:3, 3], atol=1e-2))

    def test_rgb(self):
        with torch.inference_mode():
            estimator = RGBICPPoseEstimator(self.ref_frame.shape, self.intrinsics, icp_weight=0.0, n_iter=100,
                                            dist_thr=0.5)
            T_se3, *_ = estimator.estimate_gn(self.ref_frame, self.target_frame,
                                              self.target_pcl)
            T = lie_se3_to_SE3(T_se3.squeeze())

        self.assertTrue(np.allclose(T[:3, :3], self.T_true[:3, :3], atol=1e-2))
        self.assertTrue(np.allclose(T[:3, 3], self.T_true[:3, 3], atol=1e-2))

    def test_combined(self):
        with torch.inference_mode():
            estimator = RGBICPPoseEstimator(self.ref_frame.shape, self.intrinsics, icp_weight=10.0, n_iter=100,
                                            dist_thr=0.5)
            T_se3, *_ = estimator.estimate_gn(self.ref_frame, self.target_frame,
                                              self.target_pcl)
            T = lie_se3_to_SE3(T_se3.squeeze())

        self.assertTrue(np.allclose(T[:3, :3], self.T_true[:3, :3], atol=1e-2))
        self.assertTrue(np.allclose(T[:3, 3], self.T_true[:3, 3], atol=1e-2))


    def test_all(self):

        self.test_icp()
        self.test_rgb()
        self.test_combined()


if __name__ == '__main__':
    unittest.main()
