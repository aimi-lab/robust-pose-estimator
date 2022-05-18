import torch
from alley_oop.fusion.surfel_map import SurfelMap
from alley_oop.pose.pyramid_pose_estimator import PyramidPoseEstimator
from alley_oop.pose.frame_class import FrameClass
from typing import Union, Tuple
from numpy import ndarray
from torch import tensor
import numpy as np
import cv2


class SLAM(object):
    def __init__(self, intrinsics:torch.tensor, config:dict, img_shape: Tuple):
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
        self.pose_estimator = PyramidPoseEstimator(intrinsics=self.intrinsics, config=config, img_shape=img_shape)
        self.cnt = 0
        self.rendered_frame = None
        self.frame = None
        self.depth_scale = 1/config['depth_clipping'][1]
        depth_min = config['depth_clipping'][0]*self.depth_scale  # internal normalization of depth
        self.dbg_opt = config['debug']
        self.recorder = OptimizationRecordings(config['pyramid_levels'])
        self.optim_res = None
        self.config = config
        self.pre_process = PreProcess(self.depth_scale, depth_min, self.dtype)

    def processFrame(self, img: tensor, depth:tensor, mask:tensor=None):
        """
        track frame and fuse points to SurfelMap
        :param img: RGB input image
        :param depth: input depth map
        :param mask: input mask (to mask out tools)
        """
        with torch.inference_mode():
            self.frame = FrameClass(img, depth, intrinsics=self.intrinsics, mask=mask)
            if self.scene is None:
                # initialize scene with first frame
                self.scene = SurfelMap(frame=self.frame, kmat=self.intrinsics, upscale=1,
                                       d_thresh=self.config['dist_thr'], depth_scale=self.depth_scale).to(self.device)
            pose, self.rendered_frame = self.pose_estimator.estimate(self.frame, self.scene)
            if self.dbg_opt:
                print(f"optimization costs: {self.pose_estimator.cost}")

            if self.cnt > 0:
                self.scene.fuse(self.frame, pose)
                if self.dbg_opt:
                    print(f"number of surfels: {self.scene.opts.shape[1]}, stable: {(self.scene.conf > self.scene.conf_thr).sum().item()}")
                self.recorder(self.scene, self.pose_estimator)
                self.optim_res = self.pose_estimator.optim_res
            self.cnt += 1
            pose_scaled = pose.clone()
            pose_scaled[:3,3] /= self.depth_scale  # de-normalize depth scaling
            return pose_scaled, self.scene

    def to(self, device: torch.device):
        self.device = device
        if self.scene is not None:
            self.scene = self.scene.to(device)
        self.pose_estimator = self.pose_estimator.to(device)
        self.intrinsics = self.intrinsics.to(device)
        return self

    def getPointCloud(self):
        return self.scene.opts.T/self.depth_scale, self.scene.gray.T/self.depth_scale

    def get_rendered_frame(self):
        return self.rendered_frame

    def get_frame(self):
        return self.frame

    def get_optimization_res(self):
        return self.optim_res

    def plot_recordings(self, show=False):
        return self.recorder.plot(show)


class OptimizationRecordings():
    def __init__(self, pyramid_levels):
        self.costs_combined = [[] for i in range(pyramid_levels)]
        self.costs_rgb = [[] for i in range(pyramid_levels)]
        self.costs_icp = [[] for i in range(pyramid_levels)]
        self.surfels_total = []
        self.surfels_stable = []
        self.pyramid_levels = pyramid_levels

    def __call__(self, scene, estimator):
        self.surfels_total.append(scene.opts.shape[1])
        self.surfels_stable.append((scene.conf > scene.conf_thr).sum().item())
        for i in range(self.pyramid_levels):
            self.costs_combined[i].append(estimator.cost[i][0])
            self.costs_icp[i].append(estimator.cost[i][1])
            self.costs_rgb[i].append(estimator.cost[i][2])

    def plot(self, show=False):
        if not show:
            import matplotlib as mpl
            mpl.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(self.pyramid_levels+1,1)

        for i in range(self.pyramid_levels):
            ax[i].set_title(f'Optimization Cost at Pyramid Lv {i}')
            ax[i].plot(self.costs_combined[i])
            ax[i].plot(self.costs_icp[i])
            ax[i].plot(self.costs_rgb[i])
            ax[i].grid()
            ax[0].legend(['combined', 'icp', 'rgb'])
        ax[self.pyramid_levels].plot(self.surfels_stable)
        ax[self.pyramid_levels].plot(self.surfels_total)
        ax[self.pyramid_levels].legend(['stable', 'unstable'])
        ax[self.pyramid_levels].set_title(f'Number of Surfels')
        ax[self.pyramid_levels].grid()
        ax[self.pyramid_levels].set_xlabel('time [frames]')
        plt.tight_layout()
        if show:
            plt.show()
        return fig, ax


class PreProcess(object):
    def __init__(self, scale, depth_min, dtype=torch.float32):
        self.depth_scale = scale
        self.depth_min = depth_min
        self.dtype = dtype

    def __call__(self, img:ndarray, depth:ndarray, mask:ndarray=None, dummy_label=None):
        img = (torch.tensor(img).permute(2, 0, 1) / 255.0).to(self.dtype)
        depth = depth * self.depth_scale  # normalize depth for numerical stability
        depth = cv2.bilateralFilter(depth, None, sigmaColor=0.01, sigmaSpace=10)
        mask = np.ones_like(depth).astype(bool) if mask is None else mask
        # depth clipping
        mask &= (depth > self.depth_min) & (depth < 1.0)
        # border points are usually unstable
        mask = cv2.erode(mask.astype(np.uint8), kernel=np.ones((7, 7)))
        depth = (torch.tensor(depth).unsqueeze(0)).to(self.dtype)
        mask = (torch.tensor(mask).unsqueeze(0)).to(torch.bool)

        return img, depth, mask

