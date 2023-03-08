from core.geometry.pinhole_transforms import transform, project
from core.optimization.declerative_node_lie import *


class DPoseSE3Head(DeclarativeNodeLie):
    def __init__(self, img_coordinates, lbgfs_iters=100, dbg=False):
        super(DPoseSE3Head, self).__init__(eps=1e-3, dbg=dbg)
        self.img_coordinates = img_coordinates
        self.lbgfs_iters = lbgfs_iters
        self.losses = []

    def objective(self, *xs, y, backward=False):
        flow, pcl1, pcl2, weights, mask1, mask2, intrinsics, _= xs

        n, _, h, w = flow.shape
        # project 3D-pcl to image plane
        warped_pts = project(pcl1.view(n, 3, -1), intrinsics, y, double_backward=backward)[0]  #(x,y, 1/depth(x,y)
        inv_depth2 = 1.0/torch.clamp(pcl2.view(n,3,-1)[:, None, 2], min=1e-12)
        flow_off = torch.cat((self.img_coordinates[None, :2] + flow.view(n, 2, -1), inv_depth2), dim=1)
        # compute residuals
        residuals = torch.sum((warped_pts- flow_off) ** 2, dim=1)
        residuals *= weights.view(n, -1)
        # mask out invalid residuals
        valid = (flow_off[:, 0] > 0) & (flow_off[:, 1] > 0) & (flow_off[:, 0] < w) & (flow_off[:, 1] < h)
        valid = torch.isinf(residuals) | torch.isnan(residuals) | ~valid.view(n, -1) | ~mask1.view(n, -1) | ~mask2.view(n, -1)

        # weight residuals by confidences
        residuals[valid] = 0.0
        loss = torch.mean(residuals, dim=1) / (h * w)  # normalize with width and height
        return loss

    def solve(self, *xs):
        self.losses = []
        self.img_coordinates = self.img_coordinates.to(xs[0].device)
        xs = [x.detach().clone()for x in xs]
        xs = [x.double() if x.dtype==torch.float32 else x for x in xs]
        with torch.enable_grad():
            n = xs[0].shape[0]
            # Solve using LBFGS optimizer:
            y = LieGroupParameter(SE3.Identity(n, 1, device=xs[0].device, requires_grad=True, dtype=torch.float64))
            # don't use strong-wolfe with lietorch SE3, it does not converge
            optimizer = torch.optim.LBFGS([y], lr=1.0, max_iter=self.lbgfs_iters, line_search_fn=None, )

            def fun():
                optimizer.zero_grad()
                loss = self.objective(*xs, y=y).sum()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(y, 10)
                return loss
            optimizer.step(fun)
        return y, None