import torch
import torch.nn as nn
from lietorch import SE3
from collections import OrderedDict
from core.geometry.pinhole_transforms import create_img_coords_t
from core.RAFT.core.raft import RAFT
from core.pose.pose_head import DPoseSE3Head
from core.optimization.declerative_node_lie import DeclarativeLayerLie
from core.interpol.flow_utils import remap_from_flow, remap_from_flow_nearest
from core.unet.unet import TinyUNet


class PoseNet(nn.Module):
    def __init__(self, config):
        super(PoseNet, self).__init__()
        self.config = config
        self.loss_weight = nn.Parameter(torch.tensor([1.0, 1.0]))
        H, W = config["image_shape"]
        self.register_buffer("img_coords", create_img_coords_t(y=H//8, x=W//8), persistent=False)
        self.use_weights = config["use_weights"]
        self.flow = RAFT(config)
        self.flow.freeze_bn()
        self.pose_head = DeclarativeLayerLie(DPoseSE3Head(self.img_coords, config['lbgfs_iters']))
        self.weight_head_2d = nn.Sequential(TinyUNet(in_channels=128 + 128 + 8, output_size=(H // 8, W // 8)),
                                            nn.Sigmoid())
        self.weight_head_3d = nn.Sequential(TinyUNet(in_channels=128 + 128 + 8 + 8, output_size=(H // 8, W // 8)),
                                            nn.Sigmoid())

    def forward(self, image1l, image2l, intrinsics, baseline, image1r, image2r, mask1=None, mask2=None, ret_confmap=False):
        """ estimate optical flow from stereo pair to get disparity map"""
        depth1, stereo_flow1, valid1 = self.flow2depth(image1l, image1r, baseline)
        mask1 = mask1[:,:,3::8, 3::8] & valid1 if mask1 is not None else valid1
        depth2, stereo_flow2, valid2 = self.flow2depth(image2l, image2r, baseline)
        mask2 = mask2[:,:,3::8, 3::8] & valid2 if mask2 is not None else valid2

        # avoid computing unnecessary gradients
        mask1.requires_grad = False
        mask2.requires_grad = False
        intrinsics.requires_grad = False
        baseline.requires_grad = False
        intrinsics = intrinsics.clone()
        intrinsics[:, :2,:] /= 8.0
        # reproject depth to 3D
        pcl1 = self.proj(depth1, intrinsics)
        pcl2 = self.proj(depth2, intrinsics)

        """ Estimate optical flow and rigid pose between pair of frames """

        time_flow, gru_hidden_state, context = self.flow(image1l, image2l, upsample=False)
        time_flow = time_flow[-1]

        """ Infer weight maps """
        *maps, pcl2, mask2 = self.get_weight_maps(pcl1, pcl2, image1l, image2l, mask2, time_flow,
                                                         stereo_flow1, stereo_flow2, gru_hidden_state, context)

        # estimate relative pose
        pose_head_ret = self.pose_head(time_flow, pcl1, pcl2, *maps, mask1.bool(), mask2.bool(), intrinsics,
                                       self.loss_weight.repeat((time_flow.shape[0], 1)))
        # perform supervision on down-sampled warp field to avoid backprop through up-sampling layer
        return self.forward_out(pose_head_ret, depth1, depth2, maps, ret_confmap)

    def infer(self, image1l, image2l, intrinsics, baseline, depth1, image2r, mask1, mask2, stereo_flow1, ret_details=False):
        with torch.inference_mode():
            """ infer depth and flow in one go using batch dimension """
            ref_imgs = torch.cat((image1l, image2l), dim=0)
            trg_imgs = torch.cat((image2l, image2r), dim=0)
            flow_predictions, gru_hidden_state, context = self.flow(ref_imgs, trg_imgs, upsample=False)
            time_flow = flow_predictions[-1][0].unsqueeze(0)
            stereo_flow2 = flow_predictions[-1][1].unsqueeze(0)
            gru_hidden_state = gru_hidden_state[0].unsqueeze(0)
            context = context[0].unsqueeze(0)

            # depth from flow
            n, _, h, w = image1l.shape
            depth2 = baseline[:, None, None] / -stereo_flow2[:, 0] / 8.0
            valid = (depth2 > 0) & (depth2 <= 1.0)
            depth2[~valid] = 1.0
            depth2 = depth2.unsqueeze(1)
            mask2 = mask2[:, :, 3::8, 3::8] & valid.unsqueeze(1)
            mask1 = mask1[:, :, 3::8, 3::8]
            intrinsics = intrinsics.clone()
            intrinsics[:, :2, :2] /= 8.0
            pcl1 = self.proj(depth1, intrinsics)
            pcl2 = self.proj(depth2, intrinsics)

            """ Infer weight maps """
            *maps, pcl2, mask2 = self.get_weight_maps(pcl1, pcl2, image1l, image2l, mask2, time_flow,
                                                             stereo_flow1, stereo_flow2, gru_hidden_state, context)
        pose_head_ret= self.pose_head(time_flow, pcl1, pcl2, *maps, mask1.bool(), mask2.bool(), intrinsics, self.loss_weight[None,:])
        return self.infer_out(pose_head_ret, depth1, depth2, maps, time_flow, stereo_flow2, mask2, ret_details)

    def forward_out(self, pose_head_ret, depth1, depth2, maps, ret_confmap=False):
        *_, pose_tan = pose_head_ret
        if ret_confmap:
            return pose_tan, depth1, depth2, maps
        return pose_tan, depth1, depth2

    def infer_out(self, pose_head_ret,depth1, depth2, maps, time_flow, stereo_flow2, mask2, ret_details=False):
        pose_se3, *_ = pose_head_ret
        pose_se3 = SE3(pose_se3)

        if ret_details:
            return pose_se3[0], depth1, depth2, maps, time_flow, stereo_flow2
        return pose_se3[0]

    def get_weight_maps(self, pcl1, pcl2, image1l, image2l, mask2, time_flow, stereo_flow1, stereo_flow2, gru_hidden_state, context):
        # warp reference frame using flow
        pcl2, _ = remap_from_flow(pcl2, time_flow)
        image2l, _ = remap_from_flow(image2l, 8*time_flow)  # compensate different resolutions
        stereo_flow2, _ = remap_from_flow(stereo_flow2, time_flow)
        mask2, valid_mapping = remap_from_flow_nearest(mask2, time_flow)
        mask2 = valid_mapping & mask2.to(bool)
        if self.use_weights:
            inp1 = torch.cat((stereo_flow1, image1l[:,:,3::8, 3::8], pcl1), dim=1)
            inp2 = torch.cat((stereo_flow2, image2l, pcl2), dim=1)
            conf1 = self.weight_head_2d(torch.cat((inp1, gru_hidden_state, context), dim=1))
            conf2 = self.weight_head_3d(torch.cat((inp1, inp2, gru_hidden_state, context), dim=1))
        else:
            conf1 = torch.ones_like(mask2, dtype=torch.float32)
            conf2 = torch.ones_like(mask2, dtype=torch.float32)
        return conf1, conf2, pcl2, mask2

    def proj(self, depth, intrinsics):
        n = depth.shape[0]
        repr = torch.linalg.inv(intrinsics)@ self.img_coords.view(1, 3, -1)
        opts = depth.view(n, 1, -1) * repr
        return opts.view(n,3,*depth.shape[-2:])

    def flow2depth(self, imagel, imager, baseline, upsample=False):
        n, _, h, w = imagel.shape
        flow = self.flow(imagel, imager, upsample=upsample)[0][-1]
        depth = baseline[:, None, None] / -flow[:, 0]
        if not upsample:
            depth/= 8.0  # factor 8 of upsampling
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
        for param in self.parameters():
            param.requires_grad = True
        for param in self.flow.parameters():
            param.requires_grad = not freeze
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        self.flow.eval()
        return self

    @property
    def loss_seq(self):
        return torch.tensor(self.pose_head.problem.losses)

