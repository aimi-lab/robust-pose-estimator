import unittest
import numpy as np
import torch
import cv2
import os
from alley_oop.pose.PoseN import PoseN
from alley_oop.utils.trajectory import read_freiburg
from alley_oop.geometry.lie_3d import lie_SE3_to_se3, lie_se3_to_SE3
from alley_oop.geometry.pinhole_transforms import create_img_coords_t, transform, homogenous

import matplotlib.pyplot as plt

class PoseNetTester(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(PoseNetTester, self).__init__(*args, **kwargs)

    def setUp(self):
        config = {"pretrained" : "../alley_oop/photometry/raft/pretrained/raft-things.pth",
                  "iters": 12, "dropout": 0.0, "small": False, "pose_scale": 1.0, "mode": "lbgfs", "image_shape": (480, 640)}

        self.pose_net = PoseN(config)
        self.pose_net, _ = self.pose_net.init_from_raft(config['pretrained'])
        basepath = 'test_data/tartan_air'

        def read_img(path):
            img = torch.tensor(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB))[None, ...]
            return img.permute(0,3,1,2).float()
        self.img1 = read_img(os.path.join(basepath, '000000_left.png'))
        self.img2 = read_img(os.path.join(basepath, '000001_left.png'))
        self.img1_r = read_img(os.path.join(basepath, '000000_right.png'))
        self.img2_r = read_img(os.path.join(basepath, '000001_right.png'))
        scale = 4000
        self.depth1 = torch.tensor(np.load(os.path.join(basepath, '000000_left_depth.npy')))[None,None,...]*1000/scale
        self.depth2 = torch.tensor(np.load(os.path.join(basepath, '000001_left_depth.npy')))[None,None,...]*1000/scale
        self.flow = torch.tensor(np.load(os.path.join(basepath, '000000_000001_flow.npy'))).permute(2,0,1).unsqueeze(0)
        self.mask = torch.tensor(np.load(os.path.join(basepath, '000000_000001_mask.npy'))) == 0
        poses = np.asarray(read_freiburg(os.path.join(basepath, 'pose_left.txt'), no_stamp=True))
        self.pose_net.eval()

        # the transparent windows are tricking the flow network, so we should mask them
        self.mask[..., 137:332, 5:190] = False

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
        self.pose_se3 = lie_SE3_to_se3(rel_pose)
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
        plt.imshow(self.img1.squeeze().permute(1,2,0).to(torch.uint8)*self.mask.squeeze().numpy()[...,None])
        plt.subplot(322)
        plt.imshow(self.img2.squeeze().permute(1,2,0).to(torch.uint8))
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

    def test_with_depth(self):

        with torch.no_grad():
            flow_predictions, y, depth1, depth2 = self.pose_net(self.img1, self.img2 ,self.intrinsics, torch.tensor(80.0), depth1=self.depth1,
                                                                             depth2=self.depth2,
                                                                             iters=12)
        print("with depth loss:", self.pose_loss(y))
        self.plt(flow_predictions[-1].squeeze()[0])
        #self.assertTrue(torch.allclose(flow_predictions[-1], self.flow, atol=0.1))
        self.assertTrue(torch.allclose(y, self.pose_se3.float(), atol=2e-3))

    def test_with_stereo(self):

        with torch.no_grad():
            flow_predictions, y, depth1, depth2 = self.pose_net(self.img1, self.img2 ,self.intrinsics, torch.tensor(80.0), image1_r=self.img1_r,
                                                                             image2_r=self.img2_r,
                                                                             iters=12)
        print("without depth loss:", self.pose_loss(y))
        self.plt(flow_predictions[-1].squeeze()[0])
        #self.assertTrue(torch.allclose(flow_predictions[-1], self.flow, atol=0.1))
        self.assertTrue(torch.allclose(y, self.pose_se3.float(), atol=2e-3))

    def test_all(self):
        self.test_with_depth()
        self.test_with_stereo()

if __name__ == '__main__':
    unittest.main()
