import unittest
import numpy as np
import cv2

from other_slam_methods.emdq_slam import EmdqSLAM
from core.geometry.camera import PinholeCamera


class EMDQSlamTester(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(EMDQSlamTester, self).__init__(*args, **kwargs)

    def setUp(self):
        pass

    def test_slam_affine(self):
        img = cv2.resize(cv2.imread('test_data/000000l.png'), (640, 480))
        disparity = cv2.resize(cv2.imread('test_data/000000l.pfm', cv2.IMREAD_UNCHANGED), (640, 480))/2
        depth = 2144.878173828125 / disparity

        (h, w) = img.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((cX, cY), 35, 1.0)
        img_tr = cv2.warpAffine(img, M, (w, h))
        depth_tr = cv2.warpAffine(depth, M, (w, h))
        camera = PinholeCamera(np.array([[517.654052734375, 0, 298.4775085449219],
                                         [0, 517.5438232421875, 244.20501708984375],
                                         [0,0,1]]))

        slam = EmdqSLAM(camera)

        pose, _ = slam(img, depth)
        pose, inliers = slam(img_tr, depth_tr)

        true_pose = np.eye(4)
        true_pose[:2,:2] = M[:,:2]
        #print(pose, 180*np.arccos(pose[0,0])/np.pi)
        #print(true_pose, 180*np.arccos(true_pose[0,0])/np.pi)
        self.assertTrue(np.allclose(pose[:3,:3], true_pose[:3,:3], atol=0.1, rtol=0.1))
        self.assertTrue(np.allclose(pose[:3,3], true_pose[:3,3], rtol=0.1, atol=2))

    def test_all(self):
        self.test_slam_affine()


if __name__ == '__main__':
    unittest.main()
