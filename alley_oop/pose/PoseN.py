import torch
import torch.nn as nn
from collections import OrderedDict

from alley_oop.geometry.pinhole_transforms import create_img_coords_t
from alley_oop.photometry.raft.core.raft import RAFT
from alley_oop.ddn.ddn.pytorch.node import DeclarativeLayer
from alley_oop.pose.pose_head import MLPPoseHead, HornPoseHead, DeclarativePoseHead3DNode


class PoseN(RAFT):
    def __init__(self, config, intrinsics):
        super(PoseN, self).__init__(config)
        H, W = config["image_shape"]
        if config['mode'] == 'mlp':
            self.pose_head = MLPPoseHead((H*W) // 64)
        elif config['mode'] == 'horn':
            self.pose_head = HornPoseHead()
        elif config['mode'] == 'declarative':
            self.pose_head = DeclarativeLayer(DeclarativePoseHead3DNode(intrinsics))
        else:
            raise NotImplementedError(f'mode {config["mode"]} not supported')
        # replace by 4-channel input conv (RGB + D)
        self.pose_scale = config['pose_scale']

        img_coords = create_img_coords_t(y=H, x=W)
        self.conf_head = nn.Sequential(nn.Conv2d(128+128+3, out_channels=32, kernel_size=(3,3), padding="same"), nn.BatchNorm2d(32),
                                       nn.Conv2d(32, out_channels=1, kernel_size=(3,3), padding="same"), nn.Sigmoid())
        self.up = nn.UpsamplingBilinear2d((H,W))
        self.repr = nn.Parameter(torch.linalg.inv(intrinsics.cpu())@ img_coords.view(3, -1), requires_grad=False)

    def proj(self, depth):
        n = depth.shape[0]
        opts = depth.view(n, 1, -1) * self.repr.unsqueeze(0)
        return opts.view(n,3,*depth.shape[-2:])

    def forward(self, image1, image2, depth1, depth2, iters=12, flow_init=None, pose_init=None):
        """ Estimate optical flow and rigid pose between pair of frames """
        pcl1 = self.proj(depth1)
        pcl2 = self.proj(depth2)

        flow_predictions, gru_hidden_state, context = super().forward(image1, image2, iters, flow_init)
        context_up = self.up(torch.cat((gru_hidden_state, context), dim=1))
        conf1 = self.conf_head(torch.cat((pcl1, context_up), dim=1))
        conf2 = self.conf_head(torch.cat((pcl1, context_up), dim=1))

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
        for param in super().parameters():
            param.requires_grad = not freeze
        try:
            for param in self.pose_head.parameters():
                param.requires_grad = True
            for param in self.conf_head.parameters():
                param.requires_grad = True
        except AttributeError:
            pass
        self.repr.requires_grad = False
        return self
