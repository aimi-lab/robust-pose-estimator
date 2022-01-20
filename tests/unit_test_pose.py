import unittest
import numpy as np

from alley_oop.pose.feat_pose_estimation import FeatPoseEstimator


class PoseTester(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(PoseTester, self).__init__(*args, **kwargs)

    def setUp(self):
        pass
    
    def test_feat_pose(self):
        
        # synthetic points
        refer = np.array([[0,0,0], [1,0,0], [0,0,1], [1,1,1], [1,1,0], [1,0,1], [0,1,1], [10,4,1], [100,100,100]]).T

        # translation and rotation definition
        from scipy.spatial.transform import Rotation as R
        r_true = R.from_euler('z', 30, degrees=True).as_matrix()
        t_true = np.array([1, 1, 1])[:,None]

        # transform points
        query = r_true @ refer + t_true

        # estimate translation and rotation
        estimator = FeatPoseEstimator(feat_refer=refer, feat_query=query)
        estimator.estimate()

        # assert output estimates
        t_bool = np.allclose(t_true, estimator.tvec, atol=1e-3)
        r_bool = np.allclose(r_true, estimator.rmat, atol=1e-3)
        self.assertTrue(t_bool)
        self.assertTrue(r_bool)

    def test_all(self):

        self.test_feat_pose()


if __name__ == '__main__':
    unittest.main()
