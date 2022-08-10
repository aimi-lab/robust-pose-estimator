import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import copy

from alley_oop.photometry.raft.core.corr import CorrBlock, AlternateCorrBlock
from alley_oop.photometry.raft.core.raft import RAFT
from alley_oop.photometry.raft.core.extractor import RGBDEncoder


class PoseHead(nn.Module):
    def __init__(self, input_dims):
        super(PoseHead, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=130, out_channels=32, kernel_size=(3, 3), padding='same'),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(3, 3), padding='same'))
        self.mlp = nn.Sequential(nn.Linear(in_features=input_dims,out_features=64),
                                    nn.ReLU(),
                                    nn.Linear(in_features=64, out_features=6))
    def forward(self, net, flow):

        out = self.convs(torch.cat((net, flow), dim=1)).view(net.shape[0], -1)
        return self.mlp(out)


class PoseN(RAFT):
    def __init__(self, config):
        super(PoseN, self).__init__(config)
        H, W = config["image_shape"]
        self.pose_head = PoseHead((H*W) // 64)
        # replace by 4-channel input conv (RGB + D)
        self.fnet = RGBDEncoder(output_dim=256, norm_fn='instance', dropout=config['dropout'])
        self.cnet = RGBDEncoder(output_dim=self.hidden_dim + self.context_dim, norm_fn='batch',
                                dropout=config['dropout'])

    def forward(self, image1, image2, depth1, depth2, conf1, conf2, iters=12, flow_init=None, pose_init=None):
        """ Estimate optical flow and rigid pose between pair of frames """
        # rescale to +/- 1.0
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0
        depth1 = 2 * depth1 - 1.0
        depth2 = 2 * depth2 - 1.0

        # stack images, depth and conf
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
            pose_se3 = self.pose_head(net, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow
            flow_up = coords1 - coords0
            flow_predictions.append(flow_up.float())

            #pose_se3 = pose_se3 + delta_pose
            pose_se3_predictions.append(pose_se3.float())
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
        self.fnet.conv1 = copy.deepcopy(raft.fnet.conv1)
        self.cnet.conv1 = copy.deepcopy(raft.cnet.conv1)
        self.load_state_dict(new_state_dict, strict=False)
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
        for param in self.pose_head.parameters():
            param.requires_grad = True
        return self
