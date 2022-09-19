import torch
import torch.nn as nn

from alley_oop.geometry.pinhole_transforms import create_img_coords_t, transform, homogenous, project
from alley_oop.geometry.absolute_pose_quarternion import align_torch
from alley_oop.ddn.ddn.pytorch.node import AbstractDeclarativeNode
from alley_oop.photometry.raft.core.utils.flow_utils import remap_from_flow
from alley_oop.geometry.lie_3d import lie_se3_to_SE3_batch, lie_se3_to_SE3_batch_small
from alley_oop.utils.pytorch import batched_dot_product
from torchimize.functions import lsq_gna_parallel, lsq_lma_parallel
from alley_oop.geometry.normals import normals_from_regular_grid


class MLPPoseHead(nn.Module):
    def __init__(self, input_dims, apply_mask=False):
        super(MLPPoseHead, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=136, out_channels=32, kernel_size=(3, 3), padding='same'),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(3, 3), padding='same'))
        self.mlp = nn.Sequential(nn.Linear(in_features=input_dims+6,out_features=64),
                                    nn.ReLU(),
                                    nn.Linear(in_features=64, out_features=6))
        self.apply_mask = apply_mask

    def forward(self, net, flow, pcl1, pcl2, pose):
        n, _, h, w = flow.shape
        pcl_aligned, valid = remap_from_flow(pcl2, flow)
        if self.apply_mask:
            pcl_aligned.view(n, 3, -1)[~valid] = 0.0
            pcl1.view(n, 3, -1)[~valid] = 0.0
        out = self.convs(torch.cat((net, flow, pcl1, pcl_aligned), dim=1)).view(net.shape[0], -1)
        return self.mlp(torch.cat((out, pose), dim=1))


class HornPoseHead(MLPPoseHead):
    def __init__(self):
        super(HornPoseHead, self).__init__(1)

    def forward(self, net, flow, pcl1, pcl2, pose):
        n = pcl1.shape[0]
        pcl_aligned, valid = remap_from_flow(pcl2, flow)
        # if we mask it here, each batch has a different size
        pcl_aligned.view(n,3, -1)[~valid] = torch.nan
        pcl1.view(n, 3, -1)[~valid] = torch.nan
        direct_se3 = align_torch(pcl_aligned.view(n, 3, -1), pcl1.view(n,3,-1))[0]
        return direct_se3


class DeclarativePoseHead3DNode(AbstractDeclarativeNode):
    def __init__(self, intrinsics: torch.tensor, loss_weight: dict={"3d": 10.0, "2d": 1.0}):
        super(DeclarativePoseHead3DNode, self).__init__()
        self.intrinsics = intrinsics
        self.loss_weight= loss_weight

    def reprojection_objective(self, flow, pcl1, pcl2, weights2, y):
        # this is generally better for rotation
        n, _, h, w = flow.shape
        img_coordinates = create_img_coords_t(y=pcl1.shape[-2], x=pcl1.shape[-1]).to(pcl1.device)
        pose = lie_se3_to_SE3_batch_small(-y)  # invert transform to be consistent with other pose estimators #ToDo check if this is ok
        # project to image plane
        warped_pts = project(pcl2.view(n,3,-1), pose, self.intrinsics[None, ...])
        flow_off = img_coordinates[None, :2] + flow.view(n, 2, -1)
        residuals = torch.sum((flow_off - warped_pts) ** 2, dim=1)

        valid = (flow_off[:, 0] > 0) & (flow_off[:, 1] > 0) & (flow_off[:, 0] < w) & (flow_off[:, 1] < h)
        valid = torch.isnan(residuals) | ~valid.view(n,-1)
        # weight residuals by confidences
        residuals *= weights2.view(n, -1)
        residuals[valid] = 0.0
        loss = torch.mean(residuals, dim=1) / (h*w)  # normalize with width and height
        return loss

    def depth_objective(self, flow, pcl1, pcl2, weights1, weights2, y):
        # this is generally better for translation (essentially in z-direction)
        # 3D geometric L2 loss
        n, _, h, w = pcl1.shape
        # se(3) to SE(3)
        pose = lie_se3_to_SE3_batch_small(y)
        # # transform point cloud given the pose
        pcl2_aligned = transform(homogenous(pcl2.view(n, 3, -1)), pose).reshape(n, 4, h, w)[:, :3]
        # resample point clouds given the optical flow
        pcl2_aligned, _ = remap_from_flow(pcl2_aligned, flow)
        weights2_aligned, valid = remap_from_flow(weights2, flow)
        # define objective loss function
        residuals = torch.sum((pcl2_aligned.view(n, 3, -1) - pcl1.view(n, 3, -1)) ** 2, dim=1)
        # reweighing residuals
        residuals *= torch.sqrt(weights2_aligned.view(n,-1)*weights1.view(n,-1))
        residuals[~valid[:, 0]] = 0.0
        return torch.mean(residuals, dim=-1)

    def objective(self, *xs, y):
        flow, pcl1, pcl2, weights1, weights2 = xs
        loss3d = self.depth_objective(flow, pcl1, pcl2, weights1, weights2, y)
        loss2d = self.reprojection_objective(flow, pcl1, pcl2, weights2, y)
        return self.loss_weight["2d"]*loss2d + self.loss_weight["3d"]*loss3d

    def solve(self, *xs):
        flow, pcl1, pcl2, weights1, weights2 = xs
        # solve using method of Horn
        flow = flow.detach()
        pcl1 = pcl1.detach().clone()
        pcl2 = pcl2.detach().clone()
        weights1 = weights1.detach().clone()
        weights2 = weights2.detach().clone()
        with torch.enable_grad():
            n = pcl1.shape[0]
            # Solve using LBFGS optimizer:
            y = torch.zeros((n,6), device=flow.device, requires_grad=True)
            optimizer = torch.optim.LBFGS([y], lr=1.0, max_iter=100, line_search_fn="strong_wolfe", )
            def fun():
                optimizer.zero_grad()
                loss = self.objective(flow, pcl1, pcl2, weights1, weights2, y=y).sum()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(y, 100)
                return loss
            optimizer.step(fun)
        return y.detach(), None


class DeclarativeRGBD(AbstractDeclarativeNode):
    def __init__(self, intrinsics: torch.tensor, loss_weight: dict={"3d": 10.0, "2d": 1.0}):
        super(DeclarativeRGBD, self).__init__()
        self.intrinsics = intrinsics
        self.wvec = torch.nn.Parameter(torch.tensor([loss_weight["3d"], loss_weight["2d"]], dtype=torch.float64), requires_grad=False)
        self.losses = []

    def reprojection_residuals(self, flow, pcl1, pcl2, weights1, weights2, y):
        # this is generally better for rotation
        n, _, h, w = flow.shape
        img_coordinates = create_img_coords_t(y=pcl1.shape[-2], x=pcl1.shape[-1]).to(pcl1.device)
        pose = lie_se3_to_SE3_batch_small(-y)  # invert transform to be consistent with other pose estimators
        # project to image plane
        warped_pts = project(pcl2.view(n,3,-1), pose, self.intrinsics[None, ...])
        flow_off = img_coordinates[None, :2] + flow.view(n, 2, -1)
        residuals = torch.linalg.norm((flow_off - warped_pts), dim=1, ord=2)

        valid = (flow_off[:, 0] > 0) & (flow_off[:, 1] > 0) & (flow_off[:, 0] < w) & (flow_off[:, 1] < h)
        valid = torch.isnan(residuals) | ~valid.view(n,-1)
        # weight residuals by confidences
        residuals *= weights2.view(n, -1)
        residuals[valid] = 0.0
        residuals /= (h*w)  # normalize with width and height
        return residuals

    def depth_residuals(self, flow, pcl1, pcl2, weights1, weights2, y):
        # this is generally better for translation (essentially in z-direction)
        # 3D geometric L2 loss
        n, _, h, w = pcl1.shape
        # se(3) to SE(3)
        pose = lie_se3_to_SE3_batch_small(y)
        # transform point cloud given the pose
        pcl2_aligned = transform(homogenous(pcl2.view(n, 3, -1)), pose).reshape(n, 4, h, w)[:, :3]
        # resample point clouds given the optical flow
        pcl2_aligned, _ = remap_from_flow(pcl2_aligned, flow)
        weights2_aligned, valid = remap_from_flow(weights2, flow)

        normals2 = []
        pad = torch.nn.ReplicationPad2d((0, 1, 0, 1))
        for i in range(n):
            nrml = normals_from_regular_grid(pcl2[i].permute(1,2,0).view((h,w,3)))
            # pad normals
            normals2.append(pad(nrml.permute(2, 0, 1)).contiguous().unsqueeze(0))
        normals2 = torch.cat(normals2)
        # define objective loss function

        residuals = batched_dot_product(normals2.view(n,3,-1),(pcl2_aligned - pcl1).view(n,3,-1))
        # reweighing residuals
        residuals *= torch.sqrt(weights2_aligned.view(n,-1)*weights1.view(n,-1))
        residuals[~valid[:, 0]] = 0.0
        return residuals

    def objective(self, *xs, y):
        flow, pcl1, pcl2, weights1, weights2 = xs
        loss3d = torch.mean(self.depth_residuals(flow, pcl1, pcl2, weights1, weights2, y=y)**2, dim=-1)
        loss2d = torch.mean(self.reprojection_residuals(flow, pcl1, pcl2, weights1, weights2, y=y)**2, dim=-1)
        return self.wvec[1].float()*loss2d + self.wvec[0].float()*loss3d

    def j_wt(self, pts):
        points3d = pts.permute(1, 0)
        # jacobian of projection and transform for se(3) (J_w*J_T)
        J = torch.zeros((len(points3d), 2, 6), dtype=points3d.dtype, device=points3d.device)
        x = points3d[:, 0]
        y = points3d[:, 1]
        zinv = 1 / points3d[:, 2]
        zinv2 = zinv ** 2
        J[:, 0, 0] = -self.intrinsics[0, 0] * x * y * zinv2
        J[:, 0, 1] = self.intrinsics[0, 0] * (1 + x ** 2 * zinv2)
        J[:, 0, 2] = -self.intrinsics[0, 0] * y * zinv
        J[:, 0, 3] = self.intrinsics[0, 0] * zinv
        J[:, 0, 5] = -self.intrinsics[0, 0] * x * zinv2
        J[:, 1, 0] = -self.intrinsics[0, 0] * (1 + y ** 2 * zinv2)
        J[:, 1, 1] = -J[:, 0, 0]
        J[:, 1, 2] = self.intrinsics[0, 0] * x * zinv
        J[:, 1, 4] = self.intrinsics[0, 0] * zinv
        J[:, 1, 5] = -self.intrinsics[0, 0] * y * zinv2
        return J

    def j_norm(self, src_pts, trg_pts, eps=1e-8):
        diff = trg_pts - src_pts
        # dot product and sqrt
        J = 2 * diff / (torch.linalg.norm(diff, ord=2, dim=0) + eps)
        return J.T.unsqueeze(1)

    def j_3d(self, pts):
        # product of J_T*J_x for se(3)
        points3d = pts.permute(1,0)
        J = torch.zeros((len(points3d), 3, 6), dtype=points3d.dtype, device=points3d.device)
        J[:, 0, 1] = points3d[:, 2]
        J[:, 0, 2] = -points3d[:, 1]
        J[:, 0, 3] = 1
        J[:, 1, 0] = -points3d[:, 2]
        J[:, 1, 2] = points3d[:, 0]
        J[:, 1, 4] = 1
        J[:, 2, 0] = points3d[:, 1]
        J[:, 2, 1] = -points3d[:, 0]
        J[:, 2, 5] = 1
        return J

    def reprojection_jacobian(self, *xs, y):
        flow, pcl1, pcl2, weights1, weights2 = xs
        n, _ ,h, w = flow.shape
        img_coordinates = create_img_coords_t(y=pcl1.shape[-2], x=pcl1.shape[-1]).to(pcl1.device)
        pose = lie_se3_to_SE3_batch_small(y)  # invert transform to be consistent with other pose estimators
        warped_pts = project(pcl2.view(n, 3, -1), pose, self.intrinsics[None, ...])
        flow_off = img_coordinates[None, :2] + flow.view(n, 2, -1)
        J = []
        for i in range(n):
            J.append(-self.j_norm(flow_off[i], warped_pts[i]) @ self.j_wt(pcl2[i].view(3,-1)))
        J = torch.stack(J, dim=0).squeeze()
        valid = (flow_off[:, 0] > 0) & (flow_off[:, 1] > 0) & (flow_off[:, 0] < w) & (flow_off[:, 1] < h)
        valid = torch.isinf(J) | torch.isnan(J) | ~valid.view(n, -1).unsqueeze(-1)
        # weight residuals by confidences
        J *= weights2.view(n, -1).unsqueeze(-1)
        J[valid] = 0.0
        J /= (h * w)  # normalize with width and height
        return J

    def depth_jacobian(self, *xs, y):
        flow, pcl1, pcl2, weights1, weights2 = xs
        n, _, h, w = flow.shape
        pose = lie_se3_to_SE3_batch_small(-y)
        pcl2_aligned = transform(homogenous(pcl2.view(n, 3, -1)), pose).reshape(n, 4, h, w)[:, :3] # are we sure we should transform pcl2?
        # resample point clouds given the optical flow
        pcl2_aligned, _ = remap_from_flow(pcl2_aligned, flow)
        weights2_aligned, valid = remap_from_flow(weights2, flow)
        pad = torch.nn.ReplicationPad2d((0, 1, 0, 1))
        jacobian = []
        for i in range(n):
            nrml = normals_from_regular_grid(pcl2[i].permute(1, 2, 0).view((h, w, 3)))
            nrml = pad(nrml.permute(2, 0, 1)).permute(1,2,0).reshape(-1, 1, 3)
            jacobian.append((nrml @ self.j_3d(pcl2_aligned[i].view(3, -1))).squeeze())
        jacobian = torch.stack(jacobian, dim=0)
        # weight jacobian by confidences
        jacobian *= torch.sqrt(weights2_aligned.view(n, -1) * weights1.view(n, -1)).unsqueeze(-1)
        jacobian[~valid[:, 0]] = 0.0
        return jacobian

    def multi_cost_fun(self, *xs, y):
        residuals_3d = self.depth_residuals(*xs, y=y.float())
        residuals_2d = self.reprojection_residuals(*xs, y=y.float())
        residuals = [residuals_3d, residuals_2d]
        loss3d = torch.mean(residuals_3d ** 2, dim=-1)
        loss2d = torch.mean(residuals_2d ** 2, dim=-1)
        self.losses.append(self.wvec[1].float() * loss2d + self.wvec[0].float() * loss3d)

        return torch.stack(residuals, dim=1).double()

    def mult_jacobian_fun(self, *xs, y):
        jacobian_3d = self.depth_jacobian(*xs, y=y.float())
        jacobian_2d = self.reprojection_jacobian(*xs, y=y.float())
        jacobians = [jacobian_3d, jacobian_2d]

        return torch.stack(jacobians, dim=1).double()

    def solve(self, *xs):
        flow, pcl1, pcl2, weights1, weights2 = xs
        # solve using method of Horn
        flow = flow.detach()
        pcl1 = pcl1.detach().clone()
        pcl2 = pcl2.detach().clone()
        weights1 = weights1.detach().clone()
        weights2 = weights2.detach().clone()

        multi_cost_fun_args = lambda y: self.multi_cost_fun(flow, pcl1, pcl2, weights1, weights2, y=y)
        multi_jaco_fun_args = lambda y: self.mult_jacobian_fun(flow, pcl1, pcl2, weights1, weights2, y=y)

        n = pcl1.shape[0]
        y = torch.zeros((n,6), device=flow.device, requires_grad=True, dtype=torch.float64)

        coeffs = lsq_lma_parallel(
            p=y,
            function=multi_cost_fun_args,
            jac_function=multi_jaco_fun_args,
            wvec=self.wvec,
            ftol=1e-9,
            ptol=1e-9,
            gtol=1e-9,
            max_iter=10,
        )

        best_idx = torch.argmin(torch.stack(self.losses)[::2], dim=0)
        y = torch.stack([coeffs[best_idx[i]][i] for i in range(n)])
        return y.float().detach(), None

    def solve_lbgfs(self, *xs):
        flow, pcl1, pcl2, weights1, weights2 = xs
        # solve using method of Horn
        flow = flow.detach()
        pcl1 = pcl1.detach().clone()
        pcl2 = pcl2.detach().clone()
        weights1 = weights1.detach().clone()
        weights2 = weights2.detach().clone()
        with torch.enable_grad():
            n = pcl1.shape[0]
            # Solve using LBFGS optimizer:
            y = torch.zeros((n,6), device=flow.device, requires_grad=True)
            optimizer = torch.optim.LBFGS([y], lr=1.0, max_iter=100, line_search_fn="strong_wolfe", )
            def fun():
                optimizer.zero_grad()
                loss = self.objective(flow, pcl1, pcl2, weights1, weights2, y=y).sum()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(y, 100)
                return loss
            optimizer.step(fun)
        return y.detach(), None