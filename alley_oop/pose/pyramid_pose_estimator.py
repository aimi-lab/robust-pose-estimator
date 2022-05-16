from alley_oop.interpol.gauss_pyramid import FrameGaussPyramid
from alley_oop.pose.frame_class import FrameClass
from alley_oop.pose.icp_rgb_pose_estimation import RGBICPPoseEstimator
from alley_oop.pose.rotation_estimation import RotationEstimator
import torch
from typing import Tuple
import numpy as np
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
        self.last_pose = torch.nn.Parameter(torch.eye(4, dtype=intrinsics.dtype))
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
                                             association_mode=self.config['mode'][i]))
        self.pose_estimator = torch.nn.ModuleList(self.pose_estimator)

    def estimate(self, frame: FrameClass, scene: SurfelMap):
        # transform scene to last camera pose coordinates
        scene_tlast = scene.transform_cpy(torch.linalg.inv(self.last_pose))
        # apply gaussian pyramid to current and rendered images
        frame_pyr, intrinsics_pyr = self.pyramid(frame)
        model_frame = None
        if self.last_frame_pyr is not None:
            # render view of scene from last camera pose
            model_frame = scene_tlast.render(self.pyramid._top_instrinsics)
            model_frame_pyr, _ = self.pyramid(model_frame)
            # compute SO(3) pre-alignment from previous image to current image

            R_cur2last, *_ = self.rot_estimator.estimate(frame_pyr[-1], self.last_frame_pyr[-1])

            T_cur2last = torch.eye(4, dtype=R_cur2last.dtype, device=R_cur2last.device)
            T_cur2last[:3,:3] = R_cur2last  # initial guess is rotation only
            # combined icp + rgb pose estimation
            for pyr_level in range(len(frame_pyr)-1, -1, -1):
                T_cur2last, _, optim_res = self.pose_estimator[pyr_level].estimate_gn(frame_pyr[pyr_level],
                                                                                      model_frame_pyr[pyr_level],
                                                                                      scene_tlast, init_pose=T_cur2last)
                self.cost[pyr_level] = self.pose_estimator[pyr_level].best_cost
                self.optim_res[pyr_level] = optim_res
            pose = self.last_pose @ T_cur2last
            self.last_pose.data = pose
        self.last_frame_pyr = frame_pyr
        return self.last_pose, model_frame

    @property
    def device(self):
        return self.last_pose.device

    def plot(self, T, ref_frame, model, intrinsics):
        ref_pcl = SurfelMap(frame=ref_frame, kmat=intrinsics)

        ref_pcl.transform(torch.linalg.inv(T))

        from viewer.view_render import Render
        viewer = Render(model.pcl2open3d(), blocking=True)
        viewer.render(np.eye(4),add_pcd=ref_pcl.pcl2open3d() )
