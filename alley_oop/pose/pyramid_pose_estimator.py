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

    def __init__(self, intrinsics: torch.Tensor, config: dict):
        """

        """
        super(PyramidPoseEstimator, self).__init__()
        self.pyramid = FrameGaussPyramid(level_num=config['pyramid_levels']-1, intrinsics=intrinsics)
        self.config = config
        self.last_frame_pyr = None
        self.last_pose = torch.nn.Parameter(torch.eye(4, dtype=intrinsics.dtype))

    def estimate(self, frame: FrameClass, model: SurfelMap):
        # transform model to last camera pose coordinates
        model = model.transform_cpy(self.last_pose)
        # apply gaussian pyramid to current and rendered images
        frame_pyr, intrinsics_pyr = self.pyramid(frame)
        model_frame = None
        if self.last_frame_pyr is not None:
            # render view of model from last camera pose
            model_frame = model.render(self.pyramid._top_instrinsics)
            model_frame_pyr, _ = self.pyramid(model_frame)
            # compute SO(3) pre-alignment from previous image to current image
            rot_estimator = RotationEstimator(frame_pyr[-1].shape, intrinsics_pyr[-1],
                                               self.config['rot']['n_iter'], self.config['rot']['Ftol']).to(self.device)
            R_last2cur, *_ = rot_estimator.estimate(frame_pyr[-1], self.last_frame_pyr[-1])

            T_last2cur = torch.eye(4, dtype=R_last2cur.dtype, device=R_last2cur.device)
            T_last2cur[:3,:3] = R_last2cur  # initial guess is rotation only
            # combined icp + rgb pose estimation
            for pyr_level in range(len(frame_pyr)-1, -1, -1):
                pose_estimator = RGBICPPoseEstimator(frame_pyr[pyr_level].shape, intrinsics_pyr[pyr_level],
                                                     self.config['icp_weight'],
                                                     self.config['n_iter'][pyr_level],
                                                     self.config['Ftol'][pyr_level]).to(self.device)
                T_last2cur, *_ = pose_estimator.estimate_gn(frame_pyr[pyr_level], model_frame_pyr[pyr_level], model, init_pose=T_last2cur)
                #self.plot(T_last2cur, frame, model, self.pyramid._top_instrinsics)
            pose = self.last_pose @ T_last2cur
            self.last_pose.data = pose
        self.last_frame_pyr = frame_pyr
        return self.last_pose, model_frame

    @property
    def device(self):
        return self.last_pose.device

    def plot(self, T, ref_frame, model, intrinsics):
        ref_pcl = SurfelMap(dept=ref_frame.depth, kmat=intrinsics, normals=ref_frame.normals.view(3, -1),
                            gray=ref_frame.img_gray.view(-1),
                            img_shape=ref_frame.shape)

        ref_pcl.transform(torch.linalg.inv(T))

        from viewer.view_render import Render
        viewer = Render(model.pcl2open3d(), blocking=True)
        viewer.render(np.eye(4),add_pcd=ref_pcl.pcl2open3d() )
