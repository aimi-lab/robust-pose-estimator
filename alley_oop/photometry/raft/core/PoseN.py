import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from alley_oop.photometry.raft.core.extractor import RGBDEncoder
from alley_oop.photometry.raft.core.raft import RAFT


class PoseN(RAFT):
    def __init__(self, config):
        super(PoseN, self).__init__(config)

        # replace by 4-channel input conv (RGB + D)
        self.fnet = RGBDEncoder(output_dim=256, norm_fn='instance', dropout=config['dropout'])
        self.cnet = RGBDEncoder(output_dim=self.hidden_dim + self.context_dim, norm_fn='batch', dropout=config['dropout'])

        H, W = config['image_shape']
        self.pose_regressor = nn.Sequential(nn.Linear(in_features=(H*W) // 32,out_features=64),
                                             nn.ReLU(),
                                             nn.Linear(in_features=64, out_features=6))

    def forward(self, *args, **kwargs):
        """ Estimate optical flow and pose between pair of frames """
        flow = super().forward(*args, **kwargs)
        n = flow[0].shape[0]
        lie_se3 = [self.pose_regressor(torch.cat((f[:,0], f[:,1]), dim=1).view(n, -1)) for f in flow]
        return flow, lie_se3

    def init_from_raft(self, raft_ckp):
        raft = RAFT(self.config)
        new_state_dict = OrderedDict()
        state_dict = torch.load(raft_ckp, map_location='cpu')
        for k, v in state_dict.items():
            name = k.replace('module.','')  # remove `module.`
            new_state_dict[name] = v
        raft.load_state_dict(new_state_dict)

        # replace first conv layer for RGB + D
        tmp_conv_weights = raft.fnet.conv1.weight.data.clone()
        # copy RGB weights
        self.fnet.conv1.weight.data[:, :3, :, :] = tmp_conv_weights.clone()
        # compute average weights over RGB channels and use that to initialize the depth channel
        # technique from Temporal Segment Network, L. Wang 2016
        mean_weights = torch.mean(tmp_conv_weights[:, :3, :, :], dim=1, keepdim=True)
        self.cnet.conv1.weight.data[:, 3:4, :, :] = mean_weights
        tmp_conv_weights = raft.cnet.conv1.weight.data.clone()
        self.cnet.conv1.weight.data[:, :3, :, :] = tmp_conv_weights.clone()
        mean_weights = torch.mean(tmp_conv_weights[:, :3, :, :], dim=1, keepdim=True)
        self.cnet.conv1.weight.data[:, 3:4, :, :] = mean_weights
        return self
