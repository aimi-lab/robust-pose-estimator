from alley_oop.photometry.poseNet import PoseNet
from alley_oop.geometry.pinhole_transforms import inv_transform
import torch
from typing import Tuple
from alley_oop.fusion.surfel_map import SurfelMap, FrameClass


class AttentionPoseEstimator(torch.nn.Module):
    def __init__(self, intrinsics: torch.Tensor, config: dict):
        """

        """
        super(AttentionPoseEstimator, self).__init__()
        self.model = PoseNet()
        self.intrinsics = intrinsics
        self.config = config
        self.last_pose= torch.nn.Parameter(torch.eye(4), requires_grad=False)
        self.cost = [0]
        self.last_frame = None
        self.frame2frame = config['frame2frame']

    def estimate(self, frame: FrameClass, scene: SurfelMap):
        if self.frame2frame:
            if self.last_frame is not None:
                rel_pose= self.model(self.last_frame.img, frame.img)
                ret_frame = self.last_frame
            else:
                rel_pose = self.last_pose.data.clone()
                ret_frame = None
            self.last_frame = frame
        else:
            # transform scene to last camera pose coordinates
            scene_tlast = scene.transform_cpy(inv_transform(self.last_pose))
            model_frame = scene_tlast.render(self.intrinsics)
            rel_pose = self.model(model_frame.img, frame.img)
            ret_frame = model_frame
        self.last_pose.data = self.last_pose.data @ torch.linalg.inv(rel_pose)
        return self.last_pose, ret_frame

    @property
    def device(self):
        return self.last_pose.device

