from alley_oop.interpol.gauss_pyramid import FrameGaussPyramid
from alley_oop.pose.frame_class import FrameClass
from alley_oop.pose.icp_rgb_pose_estimation import RGBICPPoseEstimator
from alley_oop.pose.rotation_estimation import RotationEstimator
from alley_oop.geometry.pinhole_transforms import inv_transform
import torch
from typing import Tuple
import numpy as np
from alley_oop.geometry.lie_3d import lie_se3_to_SE3, lie_SE3_to_se3
from alley_oop.fusion.surfel_map import SurfelMap


class PyramidPoseEstimator(torch.nn.Module):
    """ This is an implementation of Elastic Fusion pose estimation
    It estimates the camera rotation and translation in three stages and pyramid levels
"""

    def __init__(self, intrinsics: torch.Tensor, config: dict, img_shape: Tuple):
        """

        """
        super(PyramidPoseEstimator, self).__init__()
        self.pyramid = FrameGaussPyramid(level_num=config['pyramid_levels']-1, intrinsics=intrinsics)
        self.config = config
        self.last_frame_pyr = None
        self.last_pose_lie = torch.nn.Parameter(torch.zeros(6, dtype=intrinsics.dtype))
        self.cost = config['pyramid_levels'] * [0]
        self.optim_res = config['pyramid_levels'] * [0]

        # initialize optimizers
        intrinsics_pyr = self.pyramid.get_intrinsics()
        shape_pyr = self.pyramid.get_pyr_shapes((img_shape[1], img_shape[0]))
        self.rot_estimator = RotationEstimator(shape_pyr[-1], intrinsics_pyr[-1],
                                               self.config['rot']['n_iter'], self.config['rot']['Ftol'])
        self.pose_estimator = []
        for i in range(config['pyramid_levels']):
            self.pose_estimator.append(RGBICPPoseEstimator(shape_pyr[i], intrinsics_pyr[i],
                                             self.config['icp_weight'],
                                             self.config['n_iter'][i],
                                             self.config['Ftol'][i],
                                             dist_thr=self.config['dist_thr'],
                                             association_mode=self.config['mode'][i], dbg_opt=config['debug']))
        self.pose_estimator = torch.nn.ModuleList(self.pose_estimator)

    def estimate(self, frame: FrameClass, scene: SurfelMap, div_thr=0.1):
        # transform scene to last camera pose coordinates
        scene_tlast = scene.transform_cpy(inv_transform(self.last_pose))
        # apply gaussian pyramid to current and rendered images
        frame_pyr, intrinsics_pyr = self.pyramid(frame)
        model_frame = None
        if self.last_frame_pyr is not None:
            # render view of scene from last camera pose
            model_frame = scene_tlast.render(self.pyramid._top_instrinsics)
            model_frame_pyr, _ = self.pyramid(model_frame)
            # compute SO(3) pre-alignment from previous image to current image

            pose_rot_lie, *_ = self.rot_estimator.estimate(frame_pyr[-1], self.last_frame_pyr[-1])

            pose_cur2last_lie = torch.zeros(6, dtype=pose_rot_lie.dtype, device=pose_rot_lie.device)
            pose_cur2last_lie[:3] = -pose_rot_lie  # initial guess is rotation only
            # combined icp + rgb pose estimation
            for pyr_level in range(len(frame_pyr)-1, -1, -1):
                pose_cur2last_lie_old = pose_cur2last_lie.clone()
                pose_cur2last_lie, _, optim_res = self.pose_estimator[pyr_level].estimate_gn(frame_pyr[pyr_level],
                                                                                      model_frame_pyr[pyr_level],
                                                                                      scene_tlast, init_x=pose_cur2last_lie)
                # catch divergent optimization
                if (torch.linalg.norm(pose_cur2last_lie) > div_thr) | torch.isnan(pose_cur2last_lie).any():
                    pose_cur2last_lie = pose_cur2last_lie_old
                self.cost[pyr_level] = self.pose_estimator[pyr_level].best_cost
                self.optim_res[pyr_level] = optim_res
            pose = self.last_pose_lie - pose_cur2last_lie
            self.last_pose_lie.data = pose
        self.last_frame_pyr = frame_pyr
        return self.last_pose, model_frame

    @property
    def device(self):
        return self.last_pose.device

    @property
    def last_pose(self):
        return lie_se3_to_SE3(self.last_pose_lie).float()
