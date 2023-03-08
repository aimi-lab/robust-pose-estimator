import unittest
import torch
from lietorch import SE3, LieGroupParameter
from core.geometry.pinhole_transforms import transform, project, project2image, reproject, transform_forward, Transform


class PinholeTransformTester(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(PinholeTransformTester, self).__init__(*args, **kwargs)

    def setUp(self):

        # intrinsics
        self.resolution = (180, 180)
        self.kmat = torch.diag(torch.tensor([150, 150, 1]))
        self.kmat[0, -1] = self.resolution[1]//2
        self.kmat[1, -1] = self.resolution[0]//2

        self.pcl = torch.rand((20, 3, 180, 180)).view(20,3,-1)
        self.pcl[:, 2] = self.pcl[:, 2].abs()
        self.pcl = torch.clamp(self.pcl, 0.0001, 1)

    def test_transform(self):

        poses = SE3.Random(20)
        pcl2_transformed_lie = transform(self.pcl, poses)

        pcl2_transformed_back = transform(pcl2_transformed_lie, poses.inv())
        self.assertTrue(torch.allclose(pcl2_transformed_back, self.pcl, rtol=1e-3, atol=1e-6))

        pcl2_transformed_mat = torch.bmm(poses.matrix(), torch.cat((self.pcl, torch.ones((self.pcl.shape[0], 1, self.pcl.shape[-1]))), dim=1))
        self.assertTrue(torch.allclose(pcl2_transformed_mat[:,:3], pcl2_transformed_lie, rtol=1e-3, atol=1e-6))

    def test_transform_backward(self):
        with torch.enable_grad():
            poses = SE3.Random(20)
            poses = poses[:, None, :]
            param = LieGroupParameter(poses)
            pcl = torch.nn.Parameter(self.pcl)

            transformed_custom = transform(pcl, param)
            transformed_custom.sum().backward()
            pose_grad = param.grad.clone()
            pcl_grad = pcl.grad.clone()
            param.grad.detach_()
            param.grad.zero_()
            pcl.grad.detach_()
            pcl.grad.zero_()
            transformed = transform_forward(pcl, param)
            transformed.sum().backward()
            self.assertTrue(torch.allclose(pose_grad, param.grad))
            self.assertTrue(torch.allclose(pcl_grad, pcl.grad))
            # import torchviz
            # transformed_custom = transform(pcl, param)
            # grad_x, = torch.autograd.grad(transformed_custom.sum(), param, create_graph=True)
            # torchviz.make_dot((grad_x, param, transformed_custom), params={"grad_x": grad_x, "poses": param, "out": transformed_custom}).render('graph')

    def test_all(self):

        self.test_transform()
        self.test_transform_backward()

if __name__ == '__main__':
    unittest.main()
