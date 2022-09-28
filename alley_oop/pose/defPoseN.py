import torch
import torch.nn as nn
from collections import OrderedDict

from alley_oop.geometry.pinhole_transforms import create_img_coords_t
from alley_oop.network_core.raft.core.raft import RAFT
from alley_oop.ddn.ddn.pytorch.node import DeclarativeLayer
from alley_oop.pose.pose_head import MLPPoseHead, HornPoseHead, DeclarativePoseHead3DNode
from alley_oop.pose.PoseN import PoseN


class DefPoseN(PoseN):
    def __init__(self, config):
        super(DefPoseN, self).__init__(config)
        self.conf_head1 = nn.Sequential(nn.Conv2d(128+128+3+3+2+1, out_channels=32, kernel_size=(5,5), padding="same"), nn.BatchNorm2d(32),
                                       nn.Conv2d(32, out_channels=1, kernel_size=(3,3), padding="same"), nn.Sigmoid())
        self.conf_head2 = nn.Sequential(
            nn.Conv2d(128 + 128 + 3 + 3 + 2+1, out_channels=32, kernel_size=(5, 5), padding="same"), nn.BatchNorm2d(32),
            nn.Conv2d(32, out_channels=1, kernel_size=(3, 3), padding="same"), nn.Sigmoid())


    def forward(self, image1l, image2l, intrinsics, baseline, image1r=None, image2r=None, depth1=None, depth2=None, mask1=None, mask2=None, toolmask1=None, toolmask2=None, iters=12, flow_init=None, pose_init=None, ret_confmap=False):
        intrinsics.requires_grad = False
        baseline.requires_grad = False
        """ estimate optical flow from stereo pair to get disparity map"""
        if depth1 is None:
            depth1, flow1 = self.flow2depth(image1l, image1r, baseline)
        if depth2 is None:
            depth2, flow2 = self.flow2depth(image2l, image2r, baseline)

        toolmask1 = torch.zeros_like(depth1) if toolmask1 is None else toolmask1
        toolmask2 = torch.zeros_like(depth2) if toolmask2 is None else toolmask2

        """ Estimate optical flow and rigid pose between pair of frames """

        pcl1 = self.proj(depth1, intrinsics)
        pcl2 = self.proj(depth2, intrinsics)

        flow_predictions, gru_hidden_state, context = self.flow(image1l, image2l, iters, flow_init)
        if self.use_weights:
            context_up = self.up(torch.cat((gru_hidden_state, context), dim=1))
            conf1 = self.conf_head1(torch.cat((toolmask1, flow1, image1l, pcl1, context_up), dim=1))
            conf2 = self.conf_head2(torch.cat((toolmask2, flow2, image2l, pcl2, context_up), dim=1))
        else:
            conf1 = torch.ones_like(depth1)
            conf2 = torch.ones_like(depth2)

        # set confidence weights to zero where the mask is False
        if mask1 is not None:
            conf1 = conf1 * mask1
        if mask2 is not None:
            conf2 = conf2 * mask2

        n = image1l.shape[0]
        pose_se3 = self.pose_head(flow_predictions[-1], pcl1, pcl2, conf1, conf2, self.loss_weight.repeat(n, 1), intrinsics)
        if ret_confmap:
            return flow_predictions, pose_se3.float() / self.pose_scale, depth1, depth2, conf1, conf2
        return flow_predictions, pose_se3.float()/self.pose_scale, depth1, depth2

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
            for param in self.conf_head1.parameters():
                param.requires_grad = True
            for param in self.conf_head2.parameters():
                param.requires_grad = True
            self.loss_weight.requires_grad = True
        except AttributeError:
            pass
        self.img_coords.requires_grad = False
        return self

    def train(self, mode: bool = True):
        self.conf_head1.train(mode)
        self.conf_head2.train(mode)
        self.pose_head.train(mode)
        self.flow.eval()
        return self
