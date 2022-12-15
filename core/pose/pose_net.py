import torch
import torch.nn as nn
from lietorch import SE3
from collections import OrderedDict
from core.geometry.pinhole_transforms import create_img_coords_t
from core.RAFT.core.raft import RAFT
from core.pose.pose_head import DeclarativePoseHead3DNode, DeclarativeLayerLie
from core.interpol.flow_utils import remap_from_flow, remap_from_flow_nearest
from core.unet.unet import TinyUNet


class PoseNet(nn.Module):
    def __init__(self, config):
        super(PoseNet, self).__init__()
        self.config = config
        H, W = config["image_shape"]
        self.register_buffer("img_coords", create_img_coords_t(y=H, x=W), persistent=False)
        self.use_weights = config["use_weights"]
        self.flow = RAFT(config)
        self.flow.freeze_bn()
        self.pose_head = DeclarativeLayerLie(DeclarativePoseHead3DNode(self.img_coords, config['lbgfs_iters']))
        self.weight_head = nn.Sequential(TinyUNet(in_channels=128 + 128 + 8 + 8, output_size=(H, W)), nn.Sigmoid())

    def forward(self, image1l, image2l, intrinsics, baseline, image1r, image2r, mask1=None, mask2=None, ret_confmap=False):
        """ estimate optical flow from stereo pair to get disparity map"""
        depth1, stereo_flow1, valid1 = self.flow2depth(image1l, image1r, baseline)
        mask1 = mask1 & valid1 if mask1 is not None else valid1
        depth2, stereo_flow2, valid2 = self.flow2depth(image2l, image2r, baseline)
        mask2 = mask2 & valid2 if mask2 is not None else valid2

        # avoid computing unnecessary gradients
        mask1.requires_grad = False
        mask2.requires_grad = False
        intrinsics.requires_grad = False
        baseline.requires_grad = False

        # reproject depth to 3D
        pcl1 = self.proj(depth1, intrinsics)
        pcl2 = self.proj(depth2, intrinsics)

        """ Estimate optical flow and rigid pose between pair of frames """

        time_flow, gru_hidden_state, context = self.flow(image1l, image2l)
        time_flow = time_flow[-1]

        """ Infer weight maps """
        weights, pcl2, mask2 = self.get_weight_maps(pcl1, pcl2, image1l, image2l, mask2, time_flow,
                                                         stereo_flow1, stereo_flow2, gru_hidden_state, context)

        # estimate relative pose
        pose_se3_vec, pose_se3_tan = self.pose_head(time_flow, pcl1, pcl2, weights, mask1.bool(), mask2.bool(), intrinsics)
        if ret_confmap:
            return pose_se3_tan, depth1, depth2, weights
        return pose_se3_tan, depth1, depth2

    def infer(self, image1l, image2l, intrinsics, baseline, depth1, image2r, mask1, mask2, stereo_flow1, ret_details=False):
        with torch.inference_mode():
            """ infer depth and flow in one go using batch dimension """
            ref_imgs = torch.cat((image1l, image2l), dim=0)
            trg_imgs = torch.cat((image2l, image2r), dim=0)
            flow_predictions, gru_hidden_state, context = self.flow(ref_imgs, trg_imgs)
            time_flow = flow_predictions[-1][0].unsqueeze(0)
            stereo_flow2 = flow_predictions[-1][1].unsqueeze(0)
            gru_hidden_state = gru_hidden_state[0].unsqueeze(0)
            context = context[0].unsqueeze(0)

            # depth from flow
            n, _, h, w = image1l.shape
            depth2 = baseline[:, None, None] / -stereo_flow2[:, 0]
            valid = (depth2 > 0) & (depth2 <= 1.0)
            depth2[~valid] = 1.0
            depth2 = depth2.unsqueeze(1)
            mask2 &= valid.unsqueeze(1)
            pcl1 = self.proj(depth1, intrinsics)
            pcl2 = self.proj(depth2, intrinsics)

            """ Infer weight maps """
            weights, pcl2, mask2 = self.get_weight_maps(pcl1, pcl2, image1l, image2l, mask2, time_flow,
                                                             stereo_flow1, stereo_flow2, gru_hidden_state, context)

        pose_se3 = self.pose_head(time_flow, pcl1, pcl2, weights, mask1.bool(), mask2.bool(), intrinsics)[0]
        pose_se3 = SE3(pose_se3)
        if ret_details:
            return pose_se3, depth1, depth2, weights, time_flow, stereo_flow2
        return pose_se3

    def get_weight_maps(self, pcl1, pcl2, image1l, image2l, mask2, time_flow, stereo_flow1, stereo_flow2, gru_hidden_state, context):
        # warp reference frame using flow
        pcl2, _ = remap_from_flow(pcl2, time_flow)
        image2l, _ = remap_from_flow(image2l, time_flow)
        stereo_flow2, _ = remap_from_flow(stereo_flow2, time_flow)
        mask2, valid_mapping = remap_from_flow_nearest(mask2, time_flow)
        mask2 = valid_mapping & mask2.to(bool)
        if self.use_weights:
            inp = torch.nn.functional.interpolate(torch.cat((stereo_flow1, image1l, pcl1,stereo_flow2, image2l, pcl2), dim=1),
                                                   scale_factor=0.125, mode='bilinear')
            weights = self.weight_head(torch.cat((inp, gru_hidden_state, context), dim=1))
        else:
            weights = torch.ones_like(mask2, dtype=torch.float32)
        return weights, pcl2, mask2

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
        new_state_dict = OrderedDict()
        try:
            state_dict = torch.load(raft_ckp)
        except RuntimeError:
            state_dict = torch.load(raft_ckp, map_location='cpu')
        for k, v in state_dict.items():
            name = k.replace('module.','')  # remove `module.`
            new_state_dict[name] = v
        self.flow.load_state_dict(new_state_dict)
        return self

    def freeze_flow(self, freeze=True):
        for param in self.flow.parameters():
            param.requires_grad = not freeze
        try:
            for param in self.pose_head.parameters():
                param.requires_grad = True
            for param in self.weight_head_2d.parameters():
                param.requires_grad = True
            for param in self.weight_head.parameters():
                param.requires_grad = True
            self.loss_weight.requires_grad = True
        except AttributeError:
            pass
        self.img_coords.requires_grad = False
        return self

    def train(self, mode: bool = True):
        self.weight_head_2d.train(mode)
        self.weight_head.train(mode)
        self.pose_head.train(mode)
        self.flow.eval()
        return self

