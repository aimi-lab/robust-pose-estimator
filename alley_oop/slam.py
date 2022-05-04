import torch
from alley_oop.fusion.surfel_map import SurfelMap
from alley_oop.fusion.dummy_map import DummyMap
from alley_oop.pose.pyramid_pose_estimator import PyramidPoseEstimator
from alley_oop.pose.frame_class import FrameClass
from typing import Union
from numpy import ndarray
from torch import tensor
import cv2


class SLAM(object):
    def __init__(self, intrinsics, config):
        super().__init__()
        self.scene = None
        self.pose_estimator = PyramidPoseEstimator(intrinsics=intrinsics, config=config)
        self.device = intrinsics.device
        self.dtype = torch.float64
        self.intrinsics = intrinsics.to(self.dtype)
        self.cnt = 0
        self.rendered_frame = None
        self.frame = None

    def processFrame(self, img: Union[ndarray, tensor], depth:Union[ndarray, tensor], mask:Union[ndarray, tensor]=None):
        with torch.no_grad():
            if not torch.is_tensor(img):
                img, depth, mask = self._pre_process(img, depth, mask)
            self.frame = FrameClass(img, depth, intrinsics=self.intrinsics) #ToDo support mask
            if self.scene is None:
                # initialize scene with first frame
                self.scene = SurfelMap(dept=self.frame.depth, kmat=self.intrinsics, normals=self.frame.normals.view(3,-1),
                                       gray=self.frame.img_gray.view(1, -1), img_shape=self.frame.shape)
            pose, self.rendered_frame = self.pose_estimator.estimate(self.frame, self.scene)
            if self.cnt > 0:
                self.scene.fuse(self.frame.depth, self.frame.img_gray, self.frame.normals, pose)
            self.cnt += 1
            return pose, self.scene

    def _pre_process(self, img:ndarray, depth:ndarray, mask:ndarray=None):
        img = (torch.tensor(img).permute(2,0,1).unsqueeze(0)/255.0).to(self.dtype).to(self.device)
        depth = cv2.bilateralFilter(depth, None, sigmaColor=10, sigmaSpace=10)
        depth = (torch.tensor(depth).unsqueeze(0).unsqueeze(0)).to(self.dtype).to(self.device)
        if mask is not None:
            mask = (torch.tensor(mask).unsqueeze(0).unsqueeze(0)).to(self.dtype).to(self.device)
        return img, depth, mask

    def to(self, device: torch.device):
        self.device = device
        self.scene = self.scene.to(device)
        self.pose_estimator = self.pose_estimator.to(device)
        self.intrinsics = self.intrinsics.to(device)
        return self

    def getPointCloud(self):
        return self.scene.opts.T, self.scene.gray.T

    def get_rendered_frame(self):
        return self.rendered_frame

    def get_frame(self):
        return self.frame
