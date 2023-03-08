from core.geometry.pinhole_transforms import transform, project
from core.optimization.declerative_node_lie import *


class DPoseSE3Head(DeclarativeNodeLie):
    def __init__(self, img_coordinates, lbgfs_iters=100, dbg=False):
        super(DPoseSE3Head, self).__init__(eps=1e-3, dbg=dbg)
        self.img_coordinates = img_coordinates
        self.lbgfs_iters = lbgfs_iters
        self.losses = []

    def reprojection_objective(self, flow, pcl1, weights1, mask1, intrinsics, y, backward=False, ret_res=False):
        """
            r2D - reprojection residuals
        """
        n, _, h, w = flow.shape
        # project 3D-pcl to image plane
        warped_pts = project(pcl1.view(n,3,-1), intrinsics, y, double_backward=backward)[:,:2]
        flow_off = self.img_coordinates[None, :2] + flow.view(n, 2, -1)
        # compute residuals
        residuals = torch.sum((flow_off - warped_pts)**2, dim=1)
        residuals *= weights1.view(n, -1)
        # mask out invalid residuals
        valid = (flow_off[:, 0] > 0) & (flow_off[:, 1] > 0) & (flow_off[:, 0] < w) & (flow_off[:, 1] < h)
        valid = torch.isinf(residuals) | torch.isnan(residuals) | ~valid.view(n,-1) | ~mask1.view(n,-1)

        # weight residuals by confidences
        residuals[valid] = 0.0
        loss = torch.mean(residuals, dim=1) / (h*w)  # normalize with width and height
        if ret_res:
            flow = warped_pts - self.img_coordinates[None, :2]
            return loss, residuals, flow
        return loss

    def depth_objective(self, pcl1, pcl2, weights2, mask1, mask2, y, backward=False, ret_res=False):
        """
            r3D - point-to-point 3D residuals
        """
        n, _, h, w = pcl1.shape
        # transform point cloud given the pose
        pcl1_aligned = transform(pcl1.view(n, 3, -1), y, double_backward=backward).reshape(n, 3, h, w)
        # compute residuals
        residuals = torch.sum((pcl1_aligned.view(n, 3, -1) - pcl2.view(n, 3, -1)) ** 2, dim=1)
        # reweighing residuals
        residuals *= weights2.view(n, -1)
        # mask out invalid residuals
        valid = mask1 & mask2
        residuals[~valid.view(n, -1)] = 0.0
        if ret_res:
            return torch.mean(residuals, dim=-1), residuals
        return torch.mean(residuals, dim=-1)

    def objective(self, *xs, y, backward=False):
        pose = y[0]
        flow, pcl1, pcl2, weights1, weights2, mask1, mask2, intrinsics, loss_weight = xs
        loss3d = self.depth_objective(pcl1, pcl2, weights2, mask1, mask2, pose, backward)
        loss2d = self.reprojection_objective(flow, pcl1, weights1,mask1, intrinsics, pose, backward)
        return loss_weight[:, 1]*loss2d + loss_weight[:, 0]*loss3d

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
                loss = self.objective(*xs, y=(y,)).sum()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(y, 10)
                return loss
            optimizer.step(fun)
        return y, None