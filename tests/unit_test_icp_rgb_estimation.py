import unittest
import numpy as np
from pathlib import Path

from alley_oop.pose.icp_rgb_pose_estimation import RGBICPPoseEstimator, FrameClass
from alley_oop.fusion.surfel_map import SurfelMap
import cv2
import torch
from scipy.spatial.transform import Rotation as R
from alley_oop.geometry.lie_3d import lie_SE3_to_se3
from alley_oop.interpol.synth_view import synth_view

class RGBICPPoseEstimatorTester(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(RGBICPPoseEstimatorTester, self).__init__(*args, **kwargs)

    def setUp(self):

        np.random.seed(3008)

    def test_estimator(self):
        # load test data
        scale = 16
        disparity = cv2.imread(str(Path.cwd() / 'tests' / 'test_data' / '000000l.pfm'), cv2.IMREAD_UNCHANGED)
        h, w = (int(disparity.shape[0]/scale), int(disparity.shape[1]/scale))
        disparity = cv2.resize(disparity, (w, h))/scale
        depth = torch.tensor(4289.756 / disparity)
        img = torch.tensor(cv2.resize(cv2.imread(str(Path.cwd() / 'tests' / 'test_data' / '000000l.png')),
                                       (w, h))).float() / 255.0

        # generate dummy intrinsics and dummy images
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        R_true = torch.tensor(R.from_euler('xyz', (0.0, 2.0, 20.0), degrees=True).as_matrix())
        t_true = torch.tensor([0.0, 5.0, 30.0])
        T_true = torch.eye(4)
        T_true[:3, :3] = R_true
        T_true[:3, 3] = t_true
        intrinsics = torch.tensor([[1035.3/scale, 0, 596.955/scale],
                                   [0, 1035.3/scale, 488.41/scale],
                                   [0, 0, 1]])
        img = img.permute(2, 0, 1).unsqueeze(0)
        ref_frame = FrameClass(img, depth.unsqueeze(0).unsqueeze(0), intrinsics=intrinsics)
        target_img = synth_view(ref_frame.img.float(), ref_frame.depth.float(), R_true.float(),
                                t_true.unsqueeze(1).float(), intrinsics.float())
        mask = (target_img[0, 0] != 0)
        target_frame = FrameClass(target_img, depth.unsqueeze(0).unsqueeze(0), intrinsics=intrinsics)
        ref_pcl = SurfelMap(frame=ref_frame, kmat=intrinsics)
        target_pcl = ref_pcl.transform_cpy(T_true)

        estimator = RGBICPPoseEstimator(img.shape[-2:], intrinsics, icp_weight=0.001, n_iter=100).to(device)
        with torch.no_grad():
            T, cost = estimator.estimate_gn(ref_frame.to(device), target_frame.to(device),
                                      target_pcl.to(device), target_mask=mask)
        # assertion
        self.assertTrue(np.allclose(T[:3,:3].cpu(), R_true.cpu(), atol=1e-1))
        self.assertTrue(np.allclose(T[:3,3].cpu(), t_true.cpu(), atol=5))
        self.assertTrue(np.allclose(cost.cpu(),0.0, atol=1e-2))

    def test_all(self):

        self.test_estimator()


if __name__ == '__main__':
    unittest.main()
