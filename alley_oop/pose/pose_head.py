import torch
import torch.nn as nn

from alley_oop.geometry.pinhole_transforms import create_img_coords_t, transform, homogenous, project
from alley_oop.geometry.absolute_pose_quarternion import align_torch
from alley_oop.ddn.ddn.pytorch.node import *
from alley_oop.network_core.raft.core.utils.flow_utils import remap_from_flow, remap_from_flow_nearest
from alley_oop.geometry.lie_3d_pseudo import pseudo_lie_se3_to_SE3_batch_small


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
    def __init__(self):
        super(DeclarativePoseHead3DNode, self).__init__(eps=1e-3)

    def reprojection_objective(self, flow, pcl1, pcl2, weights1, mask1, intrinsics, y, ret_res=False):
        # this is generally better for rotation
        n, _, h, w = flow.shape
        img_coordinates = create_img_coords_t(y=pcl1.shape[-2], x=pcl1.shape[-1]).to(pcl1.device)
        pose = pseudo_lie_se3_to_SE3_batch_small(-y)  # invert transform to be consistent with other pose estimators
        # project to image plane
        warped_pts = project(pcl1.view(n,3,-1), pose, intrinsics)
        flow_off = img_coordinates[None, :2] + flow.view(n, 2, -1)
        residuals = torch.sum((flow_off - warped_pts)**2, dim=1)
        valid = (flow_off[:, 0] > 0) & (flow_off[:, 1] > 0) & (flow_off[:, 0] < w) & (flow_off[:, 1] < h)
        valid = torch.isinf(residuals) | torch.isnan(residuals) | ~valid.view(n,-1) | ~mask1.view(n,-1)
        # weight residuals by confidences
        residuals *= weights1.view(n,-1)
        residuals[valid] = 0.0
        loss = torch.mean(residuals, dim=1) / (h*w)  # normalize with width and height
        if ret_res:
            flow = warped_pts - img_coordinates[None, :2]
            return loss, residuals, flow
        return loss

    def depth_objective(self, flow, pcl1, pcl2, weights1, weights2, mask1, mask2, y, ret_res=False):
        # this is generally better for translation (essentially in z-direction)
        # 3D geometric L2 loss
        n, _, h, w = pcl1.shape
        # se(3) to SE(3)
        pose = pseudo_lie_se3_to_SE3_batch_small(y)
        # # transform point cloud given the pose
        pcl2_aligned = transform(homogenous(pcl2.view(n, 3, -1)), pose).reshape(n, 4, h, w)[:, :3]
        # resample point clouds given the optical flow
        pcl2_aligned, _ = remap_from_flow(pcl2_aligned, flow)
        weights2_aligned, _ = remap_from_flow(weights2, flow)
        mask2_aligned, valid = remap_from_flow_nearest(mask2, flow)
        valid &= mask1 & mask2_aligned
        # define objective loss function
        residuals = torch.sum((pcl2_aligned.view(n, 3, -1) - pcl1.view(n, 3, -1)) ** 2, dim=1)
        # reweighing residuals
        residuals *= torch.sqrt(weights2_aligned.view(n,-1)*weights1.view(n,-1))
        residuals[~valid.view(n, -1)] = 0.0
        if ret_res:
            return torch.mean(residuals, dim=-1), residuals
        return torch.mean(residuals, dim=-1)

    def objective(self, *xs, y):
        flow, pcl1, pcl2, weights1, weights2, mask1, mask2, loss_weight, intrinsics= xs
        loss3d = self.depth_objective(flow, pcl1, pcl2, weights1, weights2, mask1, mask2, y)
        loss2d = self.reprojection_objective(flow, pcl1, pcl2, weights1,mask1, intrinsics, y)
        return loss_weight[:, 1]*loss2d + loss_weight[:, 0]*loss3d

    def solve(self, *xs):
        xs = [x.detach().clone() for x in xs]
        with torch.enable_grad():
            n = xs[0].shape[0]
            # Solve using LBFGS optimizer:
            y = torch.zeros((n,6), device=xs[0].device, requires_grad=True)
            torch.backends.cuda.matmul.allow_tf32 = False
            optimizer = torch.optim.LBFGS([y], lr=1.0, max_iter=100, line_search_fn="strong_wolfe", )

            def fun():
                optimizer.zero_grad()
                loss = self.objective(*xs, y=y).sum()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(y, 100)
                return loss
            optimizer.step(fun)
        torch.backends.cuda.matmul.allow_tf32 = True
        return y.detach(), None

    def gradient(self, *xs, y=None, v=None, ctx=None):
        """Computes the vector--Jacobian product, that is, the gradient of the
        loss function with respect to the problem parameters. The returned
        gradient is a tuple of batched Torch tensors. Can be overridden by the
        derived class to provide a more efficient implementation.

        Arguments:
            xs: ((b, ...), ...) tuple of Torch tensors,
                tuple of batches of input tensors

            y: (b, ...) Torch tensor or None,
                batch of minima of the objective function

            v: (b, ...) Torch tensor or None,
                batch of gradients of the loss function with respect to the
                problem output J_Y(x,y)

            ctx: dictionary of contextual information used for computing the
                 gradient

        Return Values:
            gradients: ((b, ...), ...) tuple of Torch tensors or Nones,
                batch of gradients of the loss function with respect to the
                problem parameters;
                strictly, returns the vector--Jacobian products J_Y(x,y) * y'(x)
        """
        xs, xs_split, xs_sizes, y, v, ctx = self._gradient_init(xs, y, v, ctx)

        fY, fYY, fXY = self._get_objective_derivatives(xs, y)

        if not self._check_optimality_cond(fY):
            warnings.warn(
                "Non-zero objective function gradient at y:\n{}".format(
                    fY.detach().squeeze().cpu().numpy()))
            # set gradients to zero, so we do not perform an update
            gradients = []
            for x in xs:
                if x.requires_grad:
                    gradients.append(torch.zeros_like(x, requires_grad=False))
                else:
                    gradients.append(None)
            return tuple(gradients)

        # Form H:
        H = fYY
        H = 0.5 * (H + H.transpose(1, 2))  # Ensure that H is symmetric
        if self.gamma is not None:
            H += self.gamma * torch.eye(
                self.m, dtype=H.dtype, device=H.device).unsqueeze(0)

        # Solve u = -H^-1 v:
        v = v.reshape(self.b, -1, 1)
        u = self._solve_linear_system(H, -1.0 * v)  # bxmx1
        u = u.squeeze(-1)  # bxm

        u[torch.isnan(u)] = 0.0  # check for nan values

        # Compute -b_i^T H^-1 v (== b_i^T u) for all i:
        gradients = []
        for x_split, x_size, n in zip(xs_split, xs_sizes, self.n):
            if isinstance(x_split[0], torch.Tensor) and x_split[0].requires_grad:
                gradient = []
                for Bi in fXY(x_split):
                    gradient.append(torch.einsum('bmc,bm->bc', (Bi, u)))
                gradient = torch.cat(gradient, dim=-1)  # bxn
                gradient[torch.isnan(gradient)] = 0.0  # nan values may occur due to zero-weights
                gradients.append(gradient.reshape(x_size))
            else:
                gradients.append(None)
        return tuple(gradients)
