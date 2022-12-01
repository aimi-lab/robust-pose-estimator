import unittest
import torch
from lietorch import SE3
from core.geometry.pinhole_transforms import transform, project, project2image, reproject, create_img_coords_t
from core.pose.pose_head import DeclarativePoseHead3DNode


class PoseHeadTester(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(PoseHeadTester, self).__init__(*args, **kwargs)

    def setUp(self):
        n = 20
        # intrinsics
        self.resolution = (180, 180)
        self.kmat = torch.diag(torch.tensor([150.0, 150, 1]))
        self.kmat[0, -1] = self.resolution[1]//2
        self.kmat[1, -1] = self.resolution[0]//2
        self.kmat = self.kmat.repeat((n,1,1))

        depth = 100*torch.clamp(torch.rand((n, 1, 180, 180)), 0.01, 1)
        self.pcl = reproject(depth, self.kmat, create_img_coords_t(180,180))[:, :3].view(n, 3, 180, 180)

        self.pose_head = DeclarativePoseHead3DNode(create_img_coords_t(180,180))
        torch.random.manual_seed(12345)
        self.poses = SE3.Random(n, sigma=0.01)
        # compute induced flow
        flow_off = project(self.pcl.view(n,3,-1), self.poses.inv(), intrinsics=self.kmat).view(n,2,180,180)
        self.valid = (flow_off[:,0] >= 0) & (flow_off[:,0] < 180) & (flow_off[:,1] >= 0) & (flow_off[:,1] < 180)
        self.valid = self.valid.unsqueeze(1)
        self.flow = flow_off - create_img_coords_t(180,180)[:2].reshape(1,2, 180, 180)
        self.pcl_transformed = transform(self.pcl.view(n,3,-1), self.poses.inv()).view(n,3,180,180)
        self.masks = torch.ones((n,1,180,180), dtype=torch.bool)
        self.weights = torch.ones((n,1,180,180))

    def test_transform(self):

        n = self.flow.shape[0]
        loss_weight = torch.tensor([[0.01, 1.0]]).repeat((n,1))
        xs = (self.flow, self.pcl, self.pcl_transformed, self.weights, self.weights, self.valid, self.masks, loss_weight, self.kmat)
        loss_gt = self.pose_head.objective(*xs, y=self.poses)
        self.assertTrue(torch.allclose(loss_gt, torch.zeros_like(loss_gt)))

        poses = SE3.InitFromVec(self.pose_head.solve(*xs)[0].float())
        loss_pred = self.pose_head.objective(*xs, y=poses)
        self.assertTrue(torch.allclose(loss_pred, torch.zeros_like(loss_gt), atol=1.5))
        supervised_loss = (poses.log() - self.poses.log()).abs().sum()/n
        self.assertTrue(torch.allclose(supervised_loss, torch.zeros_like(supervised_loss), atol=0.5))
        print("gt-loss: ", loss_gt)
        print("predicted loss: ", loss_pred)
        print("supervised loss: ", supervised_loss)


    def test_all(self):

        self.test_transform()


if __name__ == '__main__':
    unittest.main()
