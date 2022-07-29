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
        ret_frame = None
        if self.frame2frame:
            if self.last_frame is not None:
                self.last_pose.data = self.model(self.last_frame.img, frame.img)
                ret_frame = self.last_frame
            self.last_frame = frame
        else:
            # transform scene to last camera pose coordinates
            scene_tlast = scene.transform_cpy(inv_transform(self.last_pose))
            model_frame = scene_tlast.render(self.intrinsics)
            self.last_pose.data = self.model(model_frame.img, frame.img)
            ret_frame = model_frame
        return self.last_pose, ret_frame

    @property
    def device(self):
        return self.last_pose.device

