import unittest
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

from alley_oop.pose.feat_pose_estimation import FeatPoseEstimator


class PoseTester(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(PoseTester, self).__init__(*args, **kwargs)

    def setUp(self):

        self.plt_opt = False
    
    def test_feat_pose(self):
        
        # synthetic points
        refer = np.array([[0,0,0], [1,0,0], [0,0,1], [1,1,1], [1,1,0], [1,0,1], [0,1,1], [10,4,1], [10,10,10]]).T*10
        refer += 200

        for dim in ['x', 'y', 'z']:
            for a in np.arange(0, 90, 2):
                # translation and rotation definition
                r_true = Rotation.from_euler(dim, a, degrees=True).as_matrix()
                t_true = np.array([1, 1, 1])[:, None]
                t_true *= a if a != 0 else 1

                # transform points
                query = r_true @ refer + t_true

                # estimate translation and rotation
                estimator = FeatPoseEstimator(feat_refer=refer, feat_query=query, quat_opt=False)
                estimator.estimate()
                query_solve = estimator.rmat @ refer + estimator.tvec
                refer_solve = estimator.rmat.T @ (query - estimator.tvec)

                # assert output estimates
                dec_pos = 2
                tvec_bool = np.allclose(t_true, estimator.tvec, atol=10**-dec_pos)
                rmat_bool = np.allclose(r_true, estimator.rmat, atol=10**-dec_pos)

                # point residuals
                refer_mse = np.mean((refer_solve - refer) ** 2)
                query_mse = np.mean((query_solve - query) ** 2)

                try:
                    self.assertTrue(tvec_bool, msg='failed for angle %s at dim %s' % (a, dim))
                    self.assertTrue(rmat_bool, msg='failed for angle %s at dim %s' % (a, dim))
                    self.assertTrue(refer_mse < 1**-dec_pos, msg='failed for angle %s at dim %s' % (a, dim))
                    self.assertTrue(query_mse < 1**-dec_pos, msg='failed for angle %s at dim %s' % (a, dim))
                except AssertionError as e:
                    if self.plt_opt:
                        fig = plt.figure(figsize=(12, 12))
                        ax = fig.add_subplot(projection='3d')
                        ax.scatter(refer[0], refer[1], refer[2], color='b', marker='x', label='refer')
                        ax.scatter(query[0], query[1], query[2], color='r', marker='x', label='query')
                        ax.scatter(query_solve[0], query_solve[1], query_solve[2], color='k', marker='.',
                                   label='refer2query mapping')
                        ax.scatter(refer_solve[0], refer_solve[1], refer_solve[2], color='g', marker='.',
                                   label='query2refer mapping')
                        plt.legend()
                        plt.show()
                    raise e

    def test_all(self):

        self.test_feat_pose()


if __name__ == '__main__':
    unittest.main()
