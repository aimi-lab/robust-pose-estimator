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

        H, W = config['image_shape']
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=(3, 3), padding='same'),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(3, 3), padding='same'))
        self.mlp = nn.Sequential(nn.Linear(in_features=(H*W) // 64,out_features=64),
                                    nn.ReLU(),
                                    nn.Linear(in_features=64, out_features=6))

    def forward(self, image1, image2, depth1, depth2, iters=12, flow_init=None, test_mode=False):
        """ Estimate optical flow and pose between pair of frames """
        flow = super().forward(image1, image2, iters, flow_init, test_mode)
        # stack the flows + depth
        lie_se3 = [self.regress_pose(f, depth1, depth2) for f in flow]
        return flow, lie_se3

    def regress_pose(self, flow, depth1, depth2):
        x = torch.cat((flow, depth1, depth2), dim=1)
        x = self.convs(x).view(x.shape[0], -1)
        return self.mlp(x)

    def init_from_raft(self, raft_ckp):
        new_state_dict = OrderedDict()
        try:
            state_dict = torch.load(raft_ckp)
        except RuntimeError:
            state_dict = torch.load(raft_ckp, map_location='cpu')
        for k, v in state_dict.items():
            name = k.replace('module.','')  # remove `module.`
            new_state_dict[name] = v
        self.load_state_dict(new_state_dict, strict=False)
        return self
