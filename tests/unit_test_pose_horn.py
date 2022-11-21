import unittest
import torch
from core.geometry.absolute_pose_quarternion import align_torch
from scipy.spatial.transform import Rotation as R


class PoseHornTester(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(PoseHornTester, self).__init__(*args, **kwargs)

    def setUp(self):
        self.pcl1 = torch.rand((2, 3, 10))
        # generate dummy intrinsics and dummy images
        self.R_true = torch.tensor(R.from_euler('xyz', (3.0, 5, 1.0), degrees=True).as_matrix()).float()[None, ...]
        self.t_true = torch.tensor([3.0, 2.0, 1.0]).float()[..., None]

        self.pcl2 = self.R_true @ self.pcl1 + self.t_true

    def test_align(self):
        se3, rot, trans = align_torch(self.pcl1, self.pcl2)
        self.assertTrue(torch.allclose(rot, self.R_true))
        self.assertTrue(torch.allclose(trans, self.t_true))

    def test_all(self):
        self.test_align()


if __name__ == '__main__':
    unittest.main()
