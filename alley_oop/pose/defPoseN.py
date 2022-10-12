from alley_oop.pose.PoseN import *


class DefPoseN(PoseN):
    def __init__(self, config):
        super(DefPoseN, self).__init__(config)
        H, W = config["image_shape"]
        self.conf_head1 = nn.Sequential(TinyUNet(in_channels=128 + 128 + 3 + 3 + 2+1, output_size=(H, W)), nn.ReLU())
        self.conf_head2 = nn.Sequential(TinyUNet(in_channels=128 + 128 + 3 + 3 + 2+1, output_size=(H, W)), nn.ReLU())


    def forward(self, image1l, image2l, intrinsics, baseline, image1r, image2r, toolmask1=None, toolmask2=None, iters=12, flow_init=None, pose_init=None, ret_confmap=False):
        intrinsics.requires_grad = False
        baseline.requires_grad = False
        """ estimate optical flow from stereo pair to get disparity map"""
        depth1, flow1, valid1 = self.flow2depth(image1l, image1r, baseline)
        depth2, flow2, valid2 = self.flow2depth(image2l, image2r, baseline)
        """ Estimate optical flow and rigid pose between pair of frames """

        pcl1 = self.proj(depth1, intrinsics)
        pcl2 = self.proj(depth2, intrinsics)

        flow_predictions, gru_hidden_state, context = self.flow(image1l, image2l, iters, flow_init)
        if self.use_weights:
            inp1 = torch.nn.functional.interpolate(torch.cat((toolmask1, flow1, image1l, pcl1), dim=1), scale_factor=0.125, mode='bilinear')
            inp2 = torch.nn.functional.interpolate(torch.cat((toolmask2, flow2, image2l, pcl2), dim=1), scale_factor=0.125, mode='bilinear')
            conf1 = self.conf_head1(torch.cat((inp1, gru_hidden_state, context), dim=1))
            conf2 = self.conf_head2(torch.cat((inp2, gru_hidden_state, context), dim=1))
        else:
            conf1 = torch.ones_like(depth1)
            conf2 = torch.ones_like(depth2)

        n = image1l.shape[0]
        pose_se3 = self.pose_head(flow_predictions[-1], pcl1, pcl2, conf1, conf2, valid1, valid2, self.loss_weight.repeat(n, 1), intrinsics)
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
