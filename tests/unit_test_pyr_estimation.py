import unittest
import numpy as np
from pathlib import Path

from alley_oop.pose.pyramid_pose_estimator import PyramidPoseEstimator, FrameClass
from alley_oop.geometry.point_cloud import PointCloud
from alley_oop.utils.pfm_handler import load_pfm
from scipy.spatial.transform import Rotation as R
import cv2
import torch


class PyramidPoseEstimatorTester(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(PyramidPoseEstimatorTester, self).__init__(*args, **kwargs)

    def setUp(self):

        np.random.seed(3008)

    def test_estimator(self):

        with torch.no_grad():
            config = {
                'pyramid_levels': 3,
                'rot': {'n_iter': 10, 'Ftol': 1e-3},
                'icp_weight': 0.0000001,
                'n_iter': [3, 4, 10]
            }
            scale = 2
            disparity, _ = load_pfm(str(Path.cwd() / 'tests' / 'test_data' / '0006.pfm'))
            h, w = (int(disparity.shape[0] / scale), int(disparity.shape[1] / scale))
            disparity = cv2.resize(disparity, (w, h))
            # background is very far, make it appear closer
            rand_background = 17 + 6 * np.random.rand(int((disparity < 20).sum()))
            disparity[disparity < 20] = rand_background
            depth = torch.tensor(1050.0 / disparity).double()
            img = torch.tensor(
                cv2.cvtColor(cv2.resize(cv2.imread(str(Path.cwd() / 'tests' / 'test_data' / '0006.png')),
                                        (w, h)), cv2.COLOR_RGB2BGR)).float() / 255.0

            # generate dummy intrinsics and dummy images
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            R_true = torch.tensor(R.from_euler('xyz', (0.0, 2.0, 5.0), degrees=True).as_matrix()).double()
            t_true = torch.tensor([0.0, 1.0, 3.0]).double()
            T_true = torch.eye(4).double()
            T_true[:3, :3] = R_true
            T_true[:3, 3] = t_true
            intrinsics = torch.tensor([[1050.0 / scale, 0, 479.5 / scale],
                                       [0, 1050.0 / scale, 269.5 / scale],
                                       [0, 0, 1]]).double()
            img = img.permute(2, 0, 1).unsqueeze(0)
            ref_frame = FrameClass(img.double(), depth.unsqueeze(0).unsqueeze(0), intrinsics=intrinsics)

            ref_pcl = PointCloud()
            ref_pcl.from_depth(depth, intrinsics, normals=ref_frame.normals)
            ref_pcl.set_colors(ref_frame.img)

            target_pcl = ref_pcl.transform_cpy(T_true)
            target_frame = target_pcl.render(intrinsics)
            mask = (target_frame.img_gray[0, 0] != 0)
            estimator = PyramidPoseEstimator(intrinsics, config).to(device)

            T = estimator.estimate(target_frame.to(device), target_pcl.to(device))
            T = estimator.estimate(ref_frame.to(device), target_pcl.to(device))
            T_true_inv = torch.linalg.inv(T_true)
            # assertion
            self.assertTrue(np.allclose(T[:3, :3].cpu(), T_true_inv[:3,:3].cpu(), atol=1e-1))
            self.assertTrue(np.allclose(T[:3, 3].cpu(), T_true_inv[:3,3].cpu(), atol=5))


    def test_all(self):

        self.test_estimator()


if __name__ == '__main__':
    unittest.main()
