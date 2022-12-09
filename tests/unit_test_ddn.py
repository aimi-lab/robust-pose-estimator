import unittest
import torch
from lietorch import SE3, LieGroupParameter
from core.geometry.pinhole_transforms import transform, project, project2image, reproject, create_img_coords_t
from core.pose.pose_head import DeclarativePoseHead3DNode, AbstractDeclarativeNode, DeclarativeLayer
import torchviz


class DummyDeclarativeHead(DeclarativePoseHead3DNode):

    def objective(self, *xs, y, backward=False):
        pcl1, = xs
        n = pcl1.shape[0]
        return torch.sum((transform(pcl1, y, double_backward=backward)**2).reshape(n, -1), dim=-1)/10


class DummyDeclarativeHead2(AbstractDeclarativeNode):

    def objective(self, *xs, y):
        pcl1, = xs
        return torch.mean((pcl1-y)**2, dim=-1)

    def solve(self, *xs):
        xs = [x.detach().clone() for x in xs]
        with torch.enable_grad():
            # Solve using LBFGS optimizer:
            y = torch.zeros_like(xs[0], requires_grad=True)
            # don't use strong-wolfe with lietorch SE3, it does not converge
            optimizer = torch.optim.LBFGS([y], lr=1.0, max_iter=10, line_search_fn=None, )

            def fun():
                optimizer.zero_grad()
                loss = self.objective(*xs, y=y).sum()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(y, 10)
                return loss

            optimizer.step(fun)
        return y.detach(), None


class DDNLieTester(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(DDNLieTester, self).__init__(*args, **kwargs)

    def setUp(self):
        n = 2
        # intrinsics
        torch.random.manual_seed(12345)
        self.pcl = torch.nn.Parameter(torch.rand((n, 3, 10)), requires_grad=True)
        self.pose_head = DeclarativeLayer(DummyDeclarativeHead2(1e-4))
        self.pose_head_lie = DeclarativeLayer(DummyDeclarativeHead(torch.zeros(1)))

    # def test_backward(self):
    #     with torch.enable_grad():
    #         xs = (self.pcl, )
    #         y = self.pose_head(*xs)
    #
    #         loss = torch.sum(y**2)
    #         loss.backward()

    def test_backwardlie(self):
        with torch.enable_grad():
            xs = (self.pcl, )
            y = self.pose_head_lie(*xs)

            loss = torch.sum((SE3(y).log()-1)**2)
            grad_x, = torch.autograd.grad(loss, self.pcl, create_graph=True)
            torchviz.make_dot((grad_x, self.pcl, y), params={"grad_x": grad_x, "x": self.pcl, "out": y}).render("graph")
            #loss.backward()
            #print(xs.grad)


    def test_all(self):

        self.test_backwardlie()


if __name__ == '__main__':
    unittest.main()
