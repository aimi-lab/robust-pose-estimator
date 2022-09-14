import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import copy

from alley_oop.geometry.pinhole_transforms import create_img_coords_t, transform, homogenous
from alley_oop.photometry.raft.core.corr import CorrBlock, AlternateCorrBlock
from alley_oop.photometry.raft.core.raft import RAFT
from alley_oop.photometry.raft.core.extractor import RGBDEncoder
from alley_oop.geometry.absolute_pose_quarternion import align_torch
from alley_oop.ddn.ddn.pytorch.node import AbstractDeclarativeNode, DeclarativeLayer
from alley_oop.photometry.raft.core.utils.flow_utils import remap_from_flow
from alley_oop.geometry.lie_3d import lie_se3_to_SE3_batch


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
    def __init__(self):
        super(DeclarativePoseHead3DNode, self).__init__()

    def objective(self, net, flow, pcl1, pcl2, dummy, y):
        # 3D geometric L2 loss
        n,_,h,w = pcl1.shape
        # se(3) to SE(3)
        pose = lie_se3_to_SE3_batch(y)
        # transform point cloud given the pose
        pcl2_aligned = transform(homogenous(pcl2.view(n,3,-1)), pose).reshape(n, 4, h, w)
        # resample point clouds given the optical flow
        pcl_aligned, valid = remap_from_flow(pcl2_aligned, flow)
        # define objective loss function
        residuals = torch.sum((pcl_aligned - pcl1).view(n,-1)**2, dim=-1)
        residuals[~valid] = 0.0
        return torch.mean(residuals)

    def solve(self, net, flow, pcl1, pcl2, dummy):
        # solve using method of Horn
        flow = flow.detach()
        pcl1 = pcl1.detach()
        pcl2 = pcl2.detach()

        n = pcl1.shape[0]
        pcl_aligned, valid = remap_from_flow(pcl2, flow)
        # if we mask it here, each batch has a different size
        pcl_aligned.view(n, 3, -1)[~valid] = torch.nan
        pcl1.view(n, 3, -1)[~valid] = torch.nan
        y = align_torch(pcl_aligned.view(n, 3, -1), pcl1.view(n, 3, -1))[0]
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
            self.pose_head = DeclarativeLayer(DeclarativePoseHead3DNode())
        else:
            raise NotImplementedError(f'mode {config["mode"]} not supported')
        # replace by 4-channel input conv (RGB + D)
        self.rgbd = config['RGBD']
        if self.rgbd:
            self.fnet = RGBDEncoder(output_dim=256, norm_fn='instance', dropout=config['dropout'])
            self.cnet = RGBDEncoder(output_dim=self.hidden_dim + self.context_dim, norm_fn='batch',
                                    dropout=config['dropout'])
        self.pose_scale = config['pose_scale']

        img_coords = create_img_coords_t(y=H//8, x=W//8)
        self.repr = nn.Parameter(torch.linalg.inv(intrinsics.cpu())@ img_coords.view(3, -1), requires_grad=False)

    def proj(self, depth):
        n = depth.shape[0]
        opts = depth.view(n, 1, -1) * self.repr.unsqueeze(0)
        return opts.view(n,3,*depth.shape[-2:])

    def forward(self, image1, image2, depth1, depth2, conf1, conf2, iters=12, flow_init=None, pose_init=None):
        """ Estimate optical flow and rigid pose between pair of frames """
        # rescale to +/- 1.0
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0
        pcl1 = self.proj(torch.nn.functional.interpolate(depth1, scale_factor=1/8))
        pcl2 = self.proj(torch.nn.functional.interpolate(depth2, scale_factor=1/8))

        # stack images, depth and conf
        if self.rgbd:
            depth1 = 2 * depth1 - 1.0
            depth2 = 2 * depth2 - 1.0
            image1 = torch.cat((image1, depth1, conf1), dim=1)
            image2 = torch.cat((image2, depth2, conf2), dim=1)

        n = image1.shape[0]
        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with torch.cuda.amp.autocast():
            fmap1, fmap2 = self.fnet([image1, image2])

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        corr_fn = CorrBlock(fmap1, fmap2, radius=self.config['corr_radius'])

        # run the context network
        with torch.cuda.amp.autocast():
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        pose_se3 = pose_init if pose_init is not None else torch.zeros((n,6), device=image1.device)

        flow_predictions = []
        pose_se3_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)  # index correlation volume

            flow = coords1 - coords0
            with torch.cuda.amp.autocast():
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)
            pose_se3 = self.pose_head(net, flow, pcl1, pcl2, pose_se3)
            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow
            flow_up = coords1 - coords0
            flow_predictions.append(flow_up.float())

            pose_se3_predictions.append(pose_se3.float()/self.pose_scale)
        return flow_predictions, pose_se3_predictions

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
        if self.rgbd:
            self.fnet.conv1 = copy.deepcopy(raft.fnet.conv1)
            self.cnet.conv1 = copy.deepcopy(raft.cnet.conv1)
        self.load_state_dict(new_state_dict, strict=False)
        if self.rgbd:
            # replace first conv layer for RGB + D
            # compute average weights over RGB channels and use that to initialize the depth channel
            # technique from Temporal Segment Network, L. Wang 2016
            self.fnet.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3)
            self.cnet.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3)
            tmp_conv_weights = raft.fnet.conv1.weight.data.clone()
            self.fnet.conv1.weight.data[:, :3, :, :] = tmp_conv_weights.clone()
            mean_weights = torch.mean(tmp_conv_weights[:, :3, :, :], dim=1, keepdim=True)
            self.fnet.conv1.weight.data[:, 3:5, :, :] = mean_weights

            tmp_conv_weights = raft.cnet.conv1.weight.data.clone()
            self.cnet.conv1.weight.data[:, :3, :, :] = tmp_conv_weights.clone()
            mean_weights = torch.mean(tmp_conv_weights[:, :3, :, :], dim=1, keepdim=True)
            self.cnet.conv1.weight.data[:, 3:5, :, :] = mean_weights
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
