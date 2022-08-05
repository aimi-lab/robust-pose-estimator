from alley_oop.photometry.raft.core.PoseN import PoseN
from alley_oop.geometry.pinhole_transforms import inv_transform
import torch
from typing import Tuple
from alley_oop.fusion.surfel_map import SurfelMap, FrameClass
from alley_oop.geometry.lie_3d import lie_se3_to_SE3


class RAFTPoseEstimator(torch.nn.Module):
    def __init__(self, intrinsics: torch.Tensor, checkpoint: str, frame2frame: bool=False):
        """

        """
        super(RAFTPoseEstimator, self).__init__()
        checkp = torch.load(checkpoint)
        model = PoseN(checkp['config']['model'])
        model.load_state_dict(checkp['state_dict'])
        model.train()
        self.model = model
        self.intrinsics = intrinsics
        self.last_pose= torch.nn.Parameter(torch.eye(4), requires_grad=False)
        self.cost = [0]
        self.last_frame = None
        self.frame2frame = frame2frame

    def estimate(self, frame: FrameClass, scene: SurfelMap):
        if self.frame2frame:
            if self.last_frame is not None:
                rel_pose_se3= self.model(self.last_frame.img, frame.img, self.last_frame.depth, frame.depth, self.last_frame.confidence, frame.confidence)
                rel_pose = lie_se3_to_SE3(rel_pose_se3)
                ret_frame = self.last_frame
            else:
                rel_pose = self.last_pose.data.clone()
                ret_frame = None
            self.last_frame = frame
        else:
            # transform scene to last camera pose coordinates
            scene_tlast = scene.transform_cpy(inv_transform(self.last_pose))
            model_frame = scene_tlast.render(self.intrinsics)
            rel_pose_se3 = self.model(model_frame.img, frame.img, model_frame.depth, frame.depth, model_frame.confidence, frame.confidence)
            rel_pose = lie_se3_to_SE3(rel_pose_se3)
            ret_frame = model_frame
        self.last_pose.data = self.last_pose.data @ torch.linalg.inv(rel_pose)
        return self.last_pose, ret_frame

    @property
    def device(self):
        return self.last_pose.device

