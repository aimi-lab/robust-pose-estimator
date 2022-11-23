import unittest
import numpy as np
import torch
import cv2
import os
from core.pose.pose_head import DeclarativePoseHead3DNode
from core.utils.trajectory import read_freiburg
from core.geometry.lie_3d_small_angle import small_angle_lie_SE3_to_se3
from core.geometry.pinhole_transforms import create_img_coords_t

import matplotlib.pyplot as plt

class PoseHeadTester(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(PoseHeadTester, self).__init__(*args, **kwargs)

    def setUp(self):
        self.pose_head = DeclarativePoseHead3DNode()
        basepath = 'test_data/tartan_air'
        self.img1 = cv2.cvtColor(cv2.imread(os.path.join(basepath, '000000_left.png')), cv2.COLOR_BGR2RGB)
        self.img2 = cv2.cvtColor(cv2.imread(os.path.join(basepath, '000001_left.png')), cv2.COLOR_BGR2RGB)
        scale = 4000
        self.depth1 = torch.tensor(np.load(os.path.join(basepath, '000000_left_depth.npy')))*1000/scale
        self.depth2 = torch.tensor(np.load(os.path.join(basepath, '000001_left_depth.npy')))*1000/scale
        self.flow = torch.tensor(np.load(os.path.join(basepath, '000000_000001_flow.npy'))).permute(2,0,1).unsqueeze(0)
        self.mask = torch.tensor(np.load(os.path.join(basepath, '000000_000001_mask.npy'))) == 0
        poses = np.asarray(read_freiburg(os.path.join(basepath, 'pose_left.txt'), no_stamp=True))

        def NED2opencv(pose_ned):
            rot = pose_ned[[1,2,0], :3]
            rot = rot[:, [1, 2, 0]]
            pose_cv = np.eye(4)
            pose_cv[:3, :3] = rot
            pose_cv[0,3] = pose_ned[1,3]
            pose_cv[1,3] = pose_ned[2,3]
            pose_cv[2,3] = pose_ned[0,3]
            return pose_cv


        poses[:,:3,3] /=scale
        rel_pose = torch.tensor(NED2opencv(np.linalg.inv(poses[0]) @ poses[1])).float()
        self.pose_se3 = small_angle_lie_SE3_to_se3(rel_pose)
        self.pose_SE3 = rel_pose
        self.intrinsics = torch.tensor([[320.0, 0, 320], [0, 320, 240], [0,0,1]]).unsqueeze(0)  # focal length x

        def proj(depth, intrinsics):
            n = depth.shape[0]
            img_coords = create_img_coords_t(y=480, x=640)
            repr = torch.linalg.inv(intrinsics) @ img_coords.view(1, 3, -1)
            opts = depth.view(n, 1, -1) * repr
            return opts.view(n, 3, *depth.shape[-2:])
        self.pcl1 = proj(self.depth1.unsqueeze(0), self.intrinsics)
        self.pcl2 = proj(self.depth2.unsqueeze(0), self.intrinsics)
        self.weights1 = torch.ones((1, 1, 480, 640)) * self.mask[None, None, ...]
        self.weights2 = torch.ones((1, 1, 480, 640))
        #self.plt(self.flow[:,0])

    def plt(self, img):
        plt.subplot(321)
        plt.imshow(self.img1*self.mask.squeeze().numpy()[...,None])
        plt.subplot(322)
        plt.imshow(self.img2)
        plt.subplot(323)
        plt.imshow(self.depth1.squeeze().numpy())
        plt.subplot(324)
        plt.imshow(self.depth1.squeeze().numpy())
        plt.subplot(325)
        plt.imshow(self.flow[:,0].squeeze().numpy())
        plt.subplot(326)
        plt.imshow(img.squeeze().numpy())
        plt.show()

    def pose_loss(self, y):
        return torch.sum((y- self.pose_se3)**2)

    def test_3d(self):
        loss3d_gt = self.pose_head.depth_objective(self.flow, self.pcl1, self.pcl2, self.weights1, self.weights2, self.pose_se3)
        self.assertAlmostEqual(loss3d_gt.item(), 0.0,delta=1e-4)
        loss_weight = torch.tensor([[10.0, 0.0]])
        y = self.pose_head.solve(self.flow, self.pcl1, self.pcl2, self.weights1, self.weights2, loss_weight, self.intrinsics)[0]
        loss3d = self.pose_head.depth_objective(self.flow, self.pcl1, self.pcl2, self.weights1, self.weights2,y)
        self.assertAlmostEqual(loss3d.item(), 0.0, delta=1e-4)
        print("3d loss:", self.pose_loss(y))
        self.assertTrue(torch.allclose(y, self.pose_se3.float(), atol=5e-4))

    def test_2d(self):
        loss2d_gt, residuals, flow2 = self.pose_head.reprojection_objective(self.flow, self.pcl1, self.pcl2, self.weights1, self.intrinsics, self.pose_se3, True)
        #self.plt(residuals.view(480, 640))
        self.assertAlmostEqual(loss2d_gt.item(), 0.0, delta=1e-4)
        loss_weight = torch.tensor([[0.0, 1.0]])
        y = self.pose_head.solve(self.flow, self.pcl1, self.pcl2, self.weights1, self.weights2, loss_weight,
                                 self.intrinsics)[0]
        loss2d = self.pose_head.reprojection_objective(self.flow, self.pcl1, self.pcl2, self.weights1, self.intrinsics, y)
        self.assertAlmostEqual(loss2d.item(), 0.0, delta=1e-4)
        print("2d loss:", self.pose_loss(y))
        self.assertTrue(torch.allclose(y, self.pose_se3.float(), atol=5e-4))

    def test_head(self):
        loss_weight = torch.tensor([[1.0, 1.0]])
        y = self.pose_head.solve(self.flow, self.pcl1, self.pcl2, self.weights1, self.weights2, loss_weight,
                                 self.intrinsics)[0]
        loss = self.pose_head.objective(self.flow, self.pcl1, self.pcl2, self.weights1, self.weights2, loss_weight,
                                 self.intrinsics, y=y)
        self.assertAlmostEqual(loss.item(), 0.0, delta=1e-4)
        print("combined loss:", self.pose_loss(y))
        self.assertTrue(torch.allclose(y, self.pose_se3.float(), atol=5e-4))

    def test_all(self):
        self.test_3d()
        self.test_2d()
        self.test_head()


if __name__ == '__main__':
    unittest.main()
