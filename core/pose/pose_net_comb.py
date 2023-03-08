import torch
import torch.nn as nn
from core.pose.pose_head_comb import DPoseSE3CombinedHead
from core.optimization.declerative_node_lie import DeclarativeLayerLie
from core.interpol.flow_utils import remap_from_flow, remap_from_flow_nearest
from core.unet.unet import TinyUNet
from core.pose.pose_net import PoseNet


class PoseNetCombined(PoseNet):
    def __init__(self, config):
        super(PoseNetCombined, self).__init__()
        self.loss_weight = torch.tensor([1.0])
        H, W = config["image_shape"]
        self.pose_head = DeclarativeLayerLie(DPoseSE3CombinedHead(self.img_coords, config['lbgfs_iters'], dbg=config['dbg']))
        self.weight_head = nn.Sequential(TinyUNet(in_channels=128 + 128 + 8 + 8, output_size=(H//8, W//8)), nn.Sigmoid())

    def get_weight_maps(self, pcl1, pcl2, image1l, image2l, mask2, time_flow, stereo_flow1, stereo_flow2, gru_hidden_state, context):
        # warp reference frame using flow
        pcl2, _ = remap_from_flow(pcl2, time_flow)
        image2l, _ = remap_from_flow(image2l, 8*time_flow)  # compensate different resolutions
        stereo_flow2, _ = remap_from_flow(stereo_flow2, time_flow)
        mask2, valid_mapping = remap_from_flow_nearest(mask2, time_flow)
        mask2 = valid_mapping & mask2.to(bool)
        if self.use_weights:
            image1l = torch.nn.functional.interpolate(image1l, scale_factor=0.125, mode='bilinear')
            weights = self.weight_head(torch.cat((stereo_flow1, image1l, pcl1, stereo_flow2, image2l, pcl2,
                                                   gru_hidden_state, context), dim=1))
        else:
            weights = torch.ones_like(mask2, dtype=torch.float32)
        return weights, pcl2, mask2
