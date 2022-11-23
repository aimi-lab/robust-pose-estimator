from core.pose.pose_net import PoseNet
from core.geometry.pinhole_transforms import inv_transform
import torch
from core.fusion.surfel_map import SurfelMap, Frame
from core.geometry.lie_3d_pseudo import pseudo_lie_se3_to_SE3
from collections import OrderedDict
import warnings
from typing import Tuple


class PoseEstimator(torch.nn.Module):
    def __init__(self, config, intrinsics: torch.Tensor, baseline: float, checkpoint: str, img_shape: Tuple,
                 init_pose: torch.tensor=torch.eye(4)):
        """

        """
        super(PoseEstimator, self).__init__()

        # load model from checkpoint
        checkp = torch.load(checkpoint)
        checkp['config']['model']['image_shape'] = (img_shape[1], img_shape[0])
        checkp['config']['model']['lbgfs_iters'] = config['lbgfs_iters']
        model = PoseNet(checkp['config']['model'])
        new_state_dict = OrderedDict()
        state_dict = checkp['state_dict']
        for k, v in state_dict.items():
            name = k.replace('module.', '')  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        model.eval()

        self.model = model
        self.intrinsics = intrinsics.unsqueeze(0).float()
        self.scale = 1 / config['depth_clipping'][1]
        self.register_buffer("last_pose",init_pose.double(), persistent=False)
        self.register_buffer("baseline", torch.tensor(baseline).unsqueeze(0).float(), persistent=False)
        self.last_frame = None
        self.frame2frame = config['frame2frame']
        self.scene = None
        self.frame = None
        self.config = config

    def forward(self, limg, rimg, mask):
        # update frame
        self.last_frame = self.frame
        self.frame = Frame(limg, rimg, mask=mask)

        if self.frame2frame:
            rel_pose_se3, ret_frame, flow = self.get_pose_f2f()
        else:
            if self.scene is None:
                # initialize scene with first frame
                self.scene = SurfelMap(frame=self.frame, kmat=self.intrinsics.squeeze(), upscale=1,
                                       d_thresh=self.config['dist_thr'],
                                       pmat=self.last_pose.data.float(), average_pts=self.config['average_pts']).to(self.device)
            rel_pose_se3, ret_frame, flow = self.get_pose_f2m()

        # check if pose is valid
        if (torch.isnan(rel_pose_se3).any()) | (torch.abs(rel_pose_se3) > 1.0e-1).any():
            # pose estimation failed, keep last image as reference and skip this one
            warnings.warn('pose estimation not converged, skip.', RuntimeWarning)
            rel_pose = torch.eye(4, dtype=torch.float64, device=self.last_pose.device)
            success = False
        else:
            rel_pose = pseudo_lie_se3_to_SE3(rel_pose_se3.double())
            success = True
        self.last_frame = ret_frame
        # chain relative pose with last pose estimation to obtain absolute pose
        rel_pose[:3, 3] /= self.scale # de-normalize depth scaling
        self.last_pose.data = self.last_pose.data @ rel_pose

        # update scene model
        if success & (flow is not None) & (self.scene is not None):
            self.scene.fuse(self.frame, self.last_pose.data)

        return self.last_pose.data, self.scene, flow

    def get_pose_f2f(self):
        """
            estimate relative pose between current and last frame
        """
        flow = None
        if self.last_frame is None:
            # if this is the first frame, we don't need to compute the relative pose
            rel_pose_se3 = torch.zeros(6, dtype=torch.float64, device=self.last_pose.device)
            depth, stereo_flow, valid = self.model.flow2depth(self.frame.img, self.frame.rimg,self.baseline*self.scale)
            self.frame.depth = depth/self.scale
            self.frame.mask &= valid
            self.frame.flow = stereo_flow
            ret_frame = None
        else:
            # get pose
            rel_pose_se3, depth1, depth2, conf_1, conf_2, flow, stereo_flow = self.model.infer(self.last_frame.img, self.frame.img,
                                                                 self.intrinsics, self.baseline*self.scale,
                                                                 depth1=self.last_frame.depth*self.scale,
                                                                 image2r=self.frame.rimg,
                                                                 mask1=self.last_frame.mask, mask2=self.frame.mask,
                                                                 stereo_flow1=self.last_frame.flow,
                                                                 ret_details=True)
            # assign values for visualization purpose
            self.frame.confidence = conf_2
            self.frame.depth = depth2/self.scale
            self.frame.flow = stereo_flow
            self.last_frame.confidence = conf_1
            ret_frame = self.last_frame

        return rel_pose_se3.double(), ret_frame, flow

    def get_pose_f2m(self):
        """
            estimate relative pose between current and canocial scene model
        """
        # transform scene to last camera pose coordinates
        scene_tlast = self.scene.transform_cpy(inv_transform(self.last_pose.float()))
        # render frame from scene
        model_frame = scene_tlast.render(self.intrinsics.squeeze())[0]
        # get pose
        flow, rel_pose_se3, depth1, depth2, conf_1, conf_2 = self.model.infer(model_frame.img, self.frame.img,
                                                                              self.intrinsics,
                                                                              self.baseline * self.scale,
                                                                              depth1=model_frame.depth*self.scale,
                                                                              image2r=self.frame.rimg,
                                                                              mask1=model_frame.mask,
                                                                              mask2=self.frame.mask,
                                                                              stereo_flow1=model_frame.flow,
                                                                              ret_details=True)
        # assign values for visualization purpose
        self.frame.confidence = conf_2
        self.frame.depth = depth2/self.scale
        model_frame.confidence = conf_1
        return rel_pose_se3.double(), model_frame, flow

    @property
    def device(self):
        return self.last_pose.device

    def get_last_frame(self):
        return self.last_frame

    def get_frame(self):
        return self.frame
