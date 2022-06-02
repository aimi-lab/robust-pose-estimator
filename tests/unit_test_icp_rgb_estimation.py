import unittest
import numpy as np
from pathlib import Path

from alley_oop.pose.icp_rgb_pose_estimation import RGBICPPoseEstimator, FrameClass
from alley_oop.fusion.surfel_map import SurfelMap
import cv2
import torch
from scipy.spatial.transform import Rotation as R
from alley_oop.utils.pfm_handler import load_pfm
from alley_oop.geometry.lie_3d import lie_se3_to_SE3

class RGBICPPoseEstimatorTester(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(RGBICPPoseEstimatorTester, self).__init__(*args, **kwargs)

    def setUp(self):

        np.random.seed(3008)

    def test_estimator(self):
        with torch.no_grad():
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
            device = torch.device('cpu')

            R_true = torch.tensor(R.from_euler('xyz', (0.0, 0.5, 1.0), degrees=True).as_matrix())
            t_true = torch.tensor([0.0, 0.0, 1.0]) / depth_scale
            T_true = torch.eye(4)
            T_true[:3, :3] = R_true
            T_true[:3, 3] = t_true
            intrinsics = torch.tensor([[1050.0 / scale, 0, 479.5 / scale],
                                       [0, 1050.0 / scale, 269.5 / scale],
                                       [0, 0, 1]])
            img = img.permute(2, 0, 1).unsqueeze(0)
            ref_frame = FrameClass(img, depth.unsqueeze(0).unsqueeze(0), intrinsics=intrinsics)

            ref_pcl = SurfelMap(frame=ref_frame, kmat=intrinsics)

            target_pcl = ref_pcl.transform_cpy(torch.linalg.inv(T_true))
            target_frame = target_pcl.render(intrinsics)

            estimator = RGBICPPoseEstimator(img.shape[-2:], intrinsics, icp_weight=10.0, n_iter=100, dist_thr=0.5).to(device)
            T_se3, *_ = estimator.estimate_gn(ref_frame.to(device), target_frame.to(device),
                                          target_pcl.to(device))
            T = lie_se3_to_SE3(T_se3.squeeze())

            self.assertTrue(np.allclose(T[:3, :3].cpu(), T_true[:3,:3].cpu(), atol=1e-2))
            self.assertTrue(np.allclose(T[:3, 3].cpu(), T_true[:3,3].cpu(), atol=1e-2))

    def test_all(self):

        self.test_estimator()


if __name__ == '__main__':
    unittest.main()
