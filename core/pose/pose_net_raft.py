import torch
import torch.nn as nn
from lietorch import SE3
from collections import OrderedDict
from core.geometry.pinhole_transforms import create_img_coords_t
from core.RAFT3D.raft3d.raft3d import *
from core.interpol.flow_utils import remap_from_flow, remap_from_flow_nearest


class PoseNetRAFT(RAFT3D):
    def __init__(self, args):
        super(PoseNetRAFT, self).__init__(args)

    def initializer(self, image1):
        batch_size, ch, ht, wd = image1.shape
        device = image1.device
        Ts, coords0 = super().initializer(image1)
        Ps = SE3.Identity(batch_size, 1, device=device)
        return Ps, Ts, coords0

    def forward(self, image1, image2, depth1, depth2, intrinsics, iters=12, train_mode=False):
        """ Estimate optical flow between pair of frames """

        Ps, Ts, coords0 = self.initializer(image1)
        corr_fn, net, inp = self.features_and_correlation(image1, image2)

        # intrinsics and depth at 1/8 resolution
        intrinsics_r8 = intrinsics / 8.0 #ToDo check intrinsics!
        depth1_r8 = depth1[:, 3::8, 3::8]
        depth2_r8 = depth2[:, 3::8, 3::8]

        flow_est_list = []
        flow_rev_list = []

        for itr in range(iters):
            Ts = Ts.detach()
            Ps = Ps.detach()

            coords1_xyz, _ = pops.projective_transform(Ts, depth1_r8, intrinsics_r8)

            coords1, zinv_proj = coords1_xyz.split([2, 1], dim=-1)
            zinv, _ = depth_sampler(1.0 / depth2_r8, coords1)

            corr = corr_fn(coords1.permute(0, 3, 1, 2).contiguous())
            flow = coords1 - coords0

            dz = zinv.unsqueeze(-1) - zinv_proj
            twist = Ts.log()

            net, mask, ae, delta, weight = \
                self.update_block(net, inp, corr, flow, dz, twist)

            target = coords1_xyz.permute(0, 3, 1, 2) + delta
            target = target.contiguous()

            # Gauss-Newton step
            # Ts = se3_field.step(Ts, ae, target, weight, depth1_r8, intrinsics_r8)
            Ts = se3_field.step_inplace(Ts, ae, target, weight, depth1_r8, intrinsics_r8)

            if train_mode:
                flow2d_rev = target.permute(0, 2, 3, 1)[..., :2] - coords0
                flow2d_rev = se3_field.cvx_upsample(8 * flow2d_rev, mask)

                Ts_up = se3_field.upsample_se3(Ts, mask)
                flow2d_est, flow3d_est, valid = pops.induced_flow(Ts_up, depth1, intrinsics)

                flow_est_list.append(flow2d_est)
                flow_rev_list.append(flow2d_rev)

        if train_mode:
            return flow_est_list, flow_rev_list

        Ts_up = se3_field.upsample_se3(Ts, mask)
        return Ts_up