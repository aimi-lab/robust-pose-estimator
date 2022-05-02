import unittest
import numpy as np
from pathlib import Path
import cv2
import torch

from alley_oop.pose.icp_estimation import ICPEstimator
from scipy.spatial.transform import Rotation as R
from alley_oop.fusion.surfel_map import SurfelMap
from alley_oop.utils.pfm_handler import load_pfm
from alley_oop.pose.frame_class import FrameClass

class IcpEstimationTester(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(IcpEstimationTester, self).__init__(*args, **kwargs)

    def setUp(self):

        np.random.seed(3008)

    def test_estimator(self):

        # load test data
        scale = 16
        disparity, _ = load_pfm(str(Path.cwd() / 'tests' / 'test_data' / '0006.pfm'))
        h, w = (int(disparity.shape[0] / scale), int(disparity.shape[1] / scale))
        img = torch.tensor(cv2.resize(cv2.imread(str(Path.cwd() / 'tests' / 'test_data' / '0006.png')),
                                      (w, h))).float() / 255.0
        disparity = cv2.resize(disparity, (w, h))
        # background is very far, make it appear closer
        rand_background = 17+ 6*np.random.rand(int((disparity < 20).sum()))
        disparity[disparity < 20] = rand_background

        depth = torch.tensor(1050.0 / disparity).double()
        # generate dummy intrinsics and dummy images
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        R_true = torch.tensor(R.from_euler('xyz', (0.0, 1.0, 2.0), degrees=True).as_matrix()).double()
        t_true = torch.tensor([0, 0.0, 3.0]).double()
        T_true = torch.eye(4).double()
        T_true[:3, :3] = R_true
        T_true[:3, 3] = t_true
        intrinsics = torch.tensor([[1050.0/scale, 0, 479.5/scale],
                                   [0, 1050.0/scale, 269.5/scale],
                                   [0, 0, 1]]).double()
        img = img.permute(2, 0, 1).unsqueeze(0)
        frame = FrameClass(img.double(), depth.unsqueeze(0).unsqueeze(0), intrinsics=intrinsics)

        ref_pcl = SurfelMap(dept=frame.depth, kmat=intrinsics, normals=frame.normals.view(3,-1), img_shape=frame.shape)

        target_pcl = ref_pcl.transform_cpy(T_true)

        estimator = ICPEstimator(depth.shape[:2], intrinsics, Ftol=1e-3, dist_thr=5, association_mode='projective').to(device)
        with torch.no_grad():
            T, cost = estimator.estimate_lm(frame.to(device), target_pcl.to(device))

        # assertion
        self.assertTrue(np.allclose(T.cpu(), T_true.cpu(), atol=1e-1))
        self.assertTrue(np.allclose(cost.cpu(),0.0, atol=1e-3))

    def test_all(self):

        self.test_estimator()


if __name__ == '__main__':
    unittest.main()
