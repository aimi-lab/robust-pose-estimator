import torch
from alley_oop.fusion.surfel_map import SurfelMap
from alley_oop.pose.raft_pose_estimator import RAFTPoseEstimator
from alley_oop.pose.frame_class import FrameClass
from typing import Union, Tuple
from torch import tensor
from alley_oop.preprocessing.preprocess import PreProcess
from alley_oop.utils.logging import OptimizationRecordings


class SLAM(object):
    def __init__(self, intrinsics:torch.tensor, config:dict, baseline: float, checkpoint: str, img_shape: Tuple, init_pose: torch.tensor=torch.eye(4)):
        """
        Alley-OOP SLAM (imitation of ElasticFusion)
        :param intrinsics: camera intrinsics tensor
        :param config: configuration dictionary
        """
        super().__init__()
        self.scene = None
        self.device = intrinsics.device
        self.dtype = torch.float32
        self.intrinsics = intrinsics.to(self.dtype)
        self.cnt = 0
        self.rendered_frame = None
        self.frame = None
        self.depth_scale = 1/config['depth_clipping'][1]
        depth_min = config['depth_clipping'][0]*self.depth_scale  # internal normalization of depth
        self.dbg_opt = config['debug']
        self.recorder = OptimizationRecordings()
        self.optim_res = None
        self.config = config
        self.pre_process = PreProcess(self.depth_scale, depth_min, self.intrinsics,
                                      self.dtype, mask_specularities=config['mask_specularities'],
                                      compensate_illumination=config['compensate_illumination'])
        init_pose[:3, 3] *= self.depth_scale
        self.init_pose = init_pose.to(self.device)
        self.pose_estimator = RAFTPoseEstimator(self.intrinsics, baseline, checkpoint, (img_shape[1], img_shape[0]), config['frame2frame'],
                                                self.init_pose)
        self.pose_estimator.model.use_weights = config['conf_weighing']

    def processFrame(self, img: tensor, depth:tensor, mask:tensor=None):
        """
        track frame and fuse points to SurfelMap
        :param img: RGB input image
        :param depth: input depth map
        :param mask: input mask (to mask out tools)
        :param confidence: depth confidence value between 0 and 1
        """
        with torch.no_grad():
            self.frame = FrameClass(img, depth, intrinsics=self.intrinsics, mask=mask)
            if self.scene is None:
                # initialize scene with first frame
                self.scene = SurfelMap(frame=self.frame, kmat=self.intrinsics.squeeze(), upscale=1,
                                       d_thresh=self.config['dist_thr'], depth_scale=self.depth_scale,
                                       pmat=self.init_pose, average_pts=self.config['average_pts']).to(self.device)
            pose, self.rendered_frame = self.pose_estimator.estimate(self.frame, self.scene)
            pose_scaled = pose.clone()
            pose_scaled[:3, 3] /= self.depth_scale  # de-normalize depth scaling
            if self.cnt > 0:
                self.scene.fuse(self.frame, pose)
                if self.dbg_opt:
                    print(f"number of surfels: {self.scene.opts.shape[1]}, stable: {(self.scene.conf >= 1.0).sum().item()}")
                self.recorder(self.scene, pose_scaled)
            self.cnt += 1

            return pose_scaled, self.scene, pose

    def to(self, device: torch.device):
        self.device = device
        if self.scene is not None:
            self.scene = self.scene.to(device)
        self.pose_estimator = self.pose_estimator.to(device)
        self.intrinsics = self.intrinsics.to(device)
        return self

    def get_rendered_frame(self):
        return self.rendered_frame

    def get_frame(self):
        return self.frame

