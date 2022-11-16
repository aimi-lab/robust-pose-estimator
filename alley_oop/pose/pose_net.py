import torch
import torch.nn as nn
from collections import OrderedDict
from alley_oop.geometry.pinhole_transforms import create_img_coords_t
from alley_oop.network_core.raft.core.raft import RAFT
from alley_oop.ddn.ddn.pytorch.node import DeclarativeLayer
from alley_oop.pose.pose_head import DeclarativePoseHead3DNode, remap_from_flow, remap_from_flow_nearest
from alley_oop.network_core.unet import TinyUNet


class PoseNet(nn.Module):
    def __init__(self, config):
        super(PoseNet, self).__init__()
        self.config = config
        H, W = config["image_shape"]
        self.pose_scale = config['pose_scale']
        self.register_buffer("img_coords", create_img_coords_t(y=H, x=W), persistent=False)
        self.use_weights = config["use_weights"]
        self.flow = RAFT(config)
        self.flow.freeze_bn()
        self.loss_weight = nn.Parameter(torch.tensor([1.0, 1.0]))  # 3d vs 2d loss weights
        self.pose_head = DeclarativeLayer(DeclarativePoseHead3DNode())
        self.weight_head_2d = nn.Sequential(TinyUNet(in_channels=128 + 128 + 8, output_size=(H, W)), nn.Sigmoid())
        self.weight_head_3d = nn.Sequential(TinyUNet(in_channels=128 + 128 + 8 + 8, output_size=(H, W)), nn.Sigmoid())

    def forward(self, image1l, image2l, intrinsics, baseline, image1r=None, image2r=None, depth1=None, depth2=None, mask1=None, mask2=None, flow1=None, flow2=None, iters=12, flow_init=None, pose_init=None, ret_confmap=False):
        intrinsics.requires_grad = False
        baseline.requires_grad = False
        """ estimate optical flow from stereo pair to get disparity map"""
        if depth1 is None:
            depth1, flow1, valid1 = self.flow2depth(image1l, image1r, baseline)
            mask1 = mask1 & valid1 if mask1 is not None else valid1
        if depth2 is None:
            depth2, flow2, valid2 = self.flow2depth(image2l, image2r, baseline)
            mask2 = mask2 & valid2 if mask2 is not None else valid2
        """ Estimate optical flow and rigid pose between pair of frames """

        pcl1 = self.proj(depth1, intrinsics)
        pcl2 = self.proj(depth2, intrinsics)

        flow_predictions, gru_hidden_state, context = self.flow(image1l, image2l, iters, flow_init)
        # remap from flow
        pcl2, _ = remap_from_flow(pcl2, flow_predictions[-1])
        image2l, _ = remap_from_flow(image2l, flow_predictions[-1])
        flow2, _ = remap_from_flow(flow2, flow_predictions[-1])
        mask2, valid_mapping = remap_from_flow_nearest(mask2, flow_predictions[-1])
        mask2 = valid_mapping & mask2.to(bool)
        if self.use_weights:
            inp1 = torch.nn.functional.interpolate(torch.cat((flow1, image1l, pcl1), dim=1),
                                                   scale_factor=0.125, mode='bilinear')
            inp2 = torch.nn.functional.interpolate(torch.cat((flow2, image2l, pcl2), dim=1),
                                                   scale_factor=0.125, mode='bilinear')
            conf1 = self.weight_head_2d(torch.cat((inp1, gru_hidden_state, context), dim=1))
            conf2 = self.weight_head_3d(torch.cat((inp1, inp2, gru_hidden_state, context), dim=1))
        else:
            conf1 = torch.ones_like(depth1)
            conf2 = torch.ones_like(depth2)

        # set confidence weights to zero where the mask is False
        if mask1 is not None:
            mask1.requires_grad = False
        if mask2 is not None:
            mask2.requires_grad = False

        n = image1l.shape[0]
        pose_se3 = self.pose_head(flow_predictions[-1], pcl1, pcl2, conf1, conf2, mask1, mask2, self.loss_weight.repeat(n, 1), intrinsics)
        if ret_confmap:
            return flow_predictions, pose_se3.float() / self.pose_scale, depth1, depth2, conf1, conf2
        return flow_predictions, pose_se3.float()/self.pose_scale, depth1, depth2

    def proj(self, depth, intrinsics):
        n = depth.shape[0]
        repr = torch.linalg.inv(intrinsics)@ self.img_coords.view(1, 3, -1)
        opts = depth.view(n, 1, -1) * repr
        return opts.view(n,3,*depth.shape[-2:])

    def flow2depth(self, imagel, imager, baseline):
        n, _, h, w = imagel.shape
        flow = self.flow(imagel, imager)[0][-1]
        depth = baseline[:, None, None] / -flow[:, 0]
        valid = (depth > 0) & (depth <= 1.0)
        depth[~valid] = 1.0
        return depth.unsqueeze(1), flow, valid.unsqueeze(1)

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
        self.flow.load_state_dict(new_state_dict)
        return self, raft

    def freeze_flow(self, freeze=True):
        for param in self.flow.parameters():
            param.requires_grad = not freeze
        try:
            for param in self.pose_head.parameters():
                param.requires_grad = True
            for param in self.weight_head_2d.parameters():
                param.requires_grad = True
            for param in self.weight_head_3d.parameters():
                param.requires_grad = True
            self.loss_weight.requires_grad = True
        except AttributeError:
            pass
        self.img_coords.requires_grad = False
        return self

    def train(self, mode: bool = True):
        self.weight_head_2d.train(mode)
        self.weight_head_3d.train(mode)
        self.pose_head.train(mode)
        self.flow.eval()
        return self


class DepthNet(RAFT):
    def __init__(self):
        config = {'dropout': 0.0, 'small': False}
        super().__init__(config)
        state_dict = torch.load('../alley_oop/network_core/raft/pretrained/raft-things.pth', map_location='cpu')
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.','')  # remove `module.`
            new_state_dict[name] = v
        self.load_state_dict(new_state_dict)

    def forward(self, imagel, imager, baseline, vertical_disp_thr=1.0):
        n, _, h, w = imagel.shape
        flow = super().forward(imagel, imager)[0][-1]
        # check if vertical disparity is small
        valid = torch.abs(flow[:, 1]) < vertical_disp_thr
        depth = baseline[:, None, None] / -flow[:, 0]
        valid &= (depth > 0) & (depth < 1.0)
        return depth.unsqueeze(1), valid.unsqueeze(1)
