import unittest
import numpy as np
import matplotlib.pyplot as plt

from alley_oop.pose.feat_pose_estimation import FeatPoseEstimator
from alley_oop.geometry.quaternions import quat2rmat, euler2quat


class PoseTester(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(PoseTester, self).__init__(*args, **kwargs)

    def setUp(self):

        self.plt_opt = False
    
    def test_feat_pose(self):
        
        # synthetic points
        refer = np.array([[0,0,0], [1,0,0], [0,0,1], [1,1,1], [1,1,0], [1,0,1], [0,1,1], [10,4,1], [10,10,10]]).T*10
        refer += 100

        for a in np.arange(0, 90, 2):
            # translation and rotation definition
            from scipy.spatial.transform import Rotation as R
            r_true = R.from_euler('z', a, degrees=True).as_matrix()
            t_true = np.array([1, 1, 1])[:, None]
            t_true *= a if a != 0 else 1

            # transform points
            query = r_true @ refer + t_true

            # estimate translation and rotation
            estimator = FeatPoseEstimator(feat_refer=refer, feat_query=query, quat_opt=False)
            estimator.estimate()
            solve = estimator.rmat @ refer + estimator.tvec

            if self.plt_opt:
                fig = plt.figure(figsize=(12, 12))
                ax = fig.add_subplot(projection='3d')
                ax.scatter(refer[0], refer[1], refer[2], 'bx')
                ax.scatter(query[0], query[1], query[2], 'r.')
                ax.scatter(solve[0], solve[1], solve[2], 'go')
                plt.show()

            # assert output estimates
            t_bool = np.allclose(np.round(t_true, 3), np.round(estimator.tvec, 3), atol=1e-3)
            r_bool = np.allclose(np.round(r_true, 2), np.round(estimator.rmat, 2), atol=1e-3)
            self.assertTrue(t_bool, msg='failed for angle %s' % a)
            self.assertTrue(r_bool, msg='failed for angle %s' % a)

    def test_all(self):

        self.test_feat_pose()


if __name__ == '__main__':
    unittest.main()
