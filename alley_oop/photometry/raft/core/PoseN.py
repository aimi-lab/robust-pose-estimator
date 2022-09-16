import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import copy

from alley_oop.geometry.pinhole_transforms import create_img_coords_t, transform, homogenous, project
from alley_oop.photometry.raft.core.corr import CorrBlock, AlternateCorrBlock
from alley_oop.photometry.raft.core.raft import RAFT
from alley_oop.photometry.raft.core.extractor import RGBDEncoder
from alley_oop.geometry.absolute_pose_quarternion import align_torch
from alley_oop.ddn.ddn.pytorch.node import AbstractDeclarativeNode, DeclarativeLayer
from alley_oop.photometry.raft.core.utils.flow_utils import remap_from_flow
from alley_oop.geometry.lie_3d import lie_se3_to_SE3_batch, lie_se3_to_SE3_batch_small


class PoseHead(nn.Module):
    def __init__(self, input_dims, apply_mask=False):
        super(PoseHead, self).__init__()
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


class HornPoseHead(PoseHead):
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
    def __init__(self, intrinsics: torch.tensor, loss_weight_3d: float=10.0):
        super(DeclarativePoseHead3DNode, self).__init__()
        self.intrinsics = intrinsics
        self.loss_weight_3d = loss_weight_3d

    def reprojection_objective(self, flow, pcl1, pcl2, weights2, y):
        n, _, h, w = flow.shape
        img_coordinates = create_img_coords_t(y=pcl1.shape[-2], x=pcl1.shape[-1]).to(pcl1.device)
        pose = lie_se3_to_SE3_batch_small(-y)  # invert transform to be consistent with other pose estimators
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
        residuals *= torch.sqrt(weights2_aligned.view(n,-1)*weights1.view(n,-1)) * residuals
        residuals[~valid[:, 0]] = 0.0
        return torch.mean(residuals, dim=-1)

    def objective(self, *xs, y):
        flow, pcl1, pcl2, weights1, weights2 = xs
        loss3d = self.depth_objective(flow, pcl1, pcl2, weights1, weights2, y)
        loss2d = self.reprojection_objective(flow, pcl1, pcl2, weights2, y)
        return loss2d + self.loss_weight_3d*loss3d

    def solve(self, *xs):
        flow, pcl1, pcl2, weights1, weights2 = xs
        # solve using method of Horn
        flow = flow.detach()
        pcl1 = pcl1.detach().clone()
        pcl2 = pcl2.detach().clone()
        with torch.enable_grad():
            n = pcl1.shape[0]
            # Solve using LBFGS optimizer:
            y = torch.zeros((n,6), device=flow.device, requires_grad=True)
            optimizer = torch.optim.LBFGS([y], lr=1.0, max_iter=30, line_search_fn="strong_wolfe", )
            def fun():
                optimizer.zero_grad()
                loss = self.objective(flow, pcl1, pcl2, weights1, weights2, y=y).sum()
                loss.backward()
                return loss
            optimizer.step(fun)
        return y.detach(), None


class PoseN(RAFT):
    def __init__(self, config, intrinsics):
        super(PoseN, self).__init__(config)
        H, W = config["image_shape"]
        if config['mode'] == 'mlp':
            self.pose_head = PoseHead((H*W) // 64)
        elif config['mode'] == 'horn':
            self.pose_head = HornPoseHead()
        elif config['mode'] == 'declarative':
            self.pose_head = DeclarativeLayer(DeclarativePoseHead3DNode(intrinsics))
        else:
            raise NotImplementedError(f'mode {config["mode"]} not supported')
        # replace by 4-channel input conv (RGB + D)
        self.pose_scale = config['pose_scale']

        img_coords = create_img_coords_t(y=H, x=W)
        self.repr = nn.Parameter(torch.linalg.inv(intrinsics.cpu())@ img_coords.view(3, -1), requires_grad=False)

    def proj(self, depth):
        n = depth.shape[0]
        opts = depth.view(n, 1, -1) * self.repr.unsqueeze(0)
        return opts.view(n,3,*depth.shape[-2:])

    def forward(self, image1, image2, depth1, depth2, conf1, conf2, iters=12, flow_init=None, pose_init=None):
        """ Estimate optical flow and rigid pose between pair of frames """
        with torch.autograd.detect_anomaly():
            pcl1 = self.proj(depth1)
            pcl2 = self.proj(depth2)
            conf1[pcl1[:,2].unsqueeze(1) < 1e-6] = 0.0
            conf2[pcl2[:,2].unsqueeze(1) < 1e-6] = 0.0

            flow_predictions = super().forward(image1, image2, iters, flow_init)
            pose_se3 = self.pose_head(flow_predictions[-1], pcl1, pcl2, conf1, conf2)
        return flow_predictions, pose_se3.float()/self.pose_scale

    def init_from_raft(self, raft_ckp):
        raft = RAFT(self.config)
        new_state_dict = OrderedDict()
        try:
            state_dict = torch.load(raft_ckp)
        except RuntimeError:
            state_dict = torch.load(raft_ckp, map_location='cpu')
        for k, v in state_dict.items():
            name = k.replace('module.','')  # remove `module.`
            new_state_dict[name] = v
        raft.load_state_dict(new_state_dict)
        self.load_state_dict(new_state_dict, strict=False)
        return self, raft

    def freeze_flow(self, freeze=True):
        for param in self.parameters():
            param.requires_grad = not freeze
        try:
            for param in self.pose_head.parameters():
                param.requires_grad = True
        except AttributeError:
            pass
        self.repr.requires_grad = False
        return self
