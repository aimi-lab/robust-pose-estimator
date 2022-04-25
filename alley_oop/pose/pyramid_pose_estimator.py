from alley_oop.interpol.gauss_pyramid import GaussPyramid
from alley_oop.pose.icp_rgb_pose_estimation import RGBICPPoseEstimator
from alley_oop.pose.rotation_estimation import RotationEstimator
import torch
from typing import Tuple


class PyramidPoseEstimator(torch.nn.Module):
    """ This is an implementation of Elastic Fusion pose estimation
    It estimates the camera rotation and translation in three stages and pyramid levels
"""

    def __init__(self, intrinsics: torch.Tensor, config: dict):
        """

        """
        super(PyramidPoseEstimator, self).__init__()
        self.pyramid = GaussPyramid(level_num=config['pyramid_levels'], intrinsics=intrinsics)
        self.config = config
        self.last_img_pyr = None
        self.last_pose = torch.nn.Parameter(torch.eye(4, dtype=intrinsics.dtype))

    def estimate(self, img, depth, model):
        # transform model to last camera pose coordinates
        model.transform(torch.linalg.inv(self.last_pose))
        # render view of model from last camera pose
        model_img, model_depth = model.render(self.pyramid._top_instrinsics)
        # apply gaussian pyramid to current and rendered images
        img_pyr, depth_pyr, intrinsics_pyr = self.pyramid(img, depth)
        model_img_pyr, model_depth_pyr, _ = self.pyramid(model_img, model_depth)
        if self.last_img_pyr is not None:
            # compute SO(3) pre-alignment from previous image to current image
            rot_estimator = RotationEstimator(img_pyr[2].shape, intrinsics_pyr[2],
                                               self.config['rot']['n_iter'], self.config['rot']['Ftol'])
            R_last2cur = rot_estimator.estimate(img_pyr[2], self.last_img_pyr[2])

            T_last2cur = torch.eye(4, dtype=R_last2cur.dtype, device=R_last2cur.device)
            T_last2cur[:3,:3] = R_last2cur  # initial guess is rotation only
            # combined icp + rgb pose estimation
            for pyr_level in range(len(img_pyr)):
                pose_estimator = RGBICPPoseEstimator(img_pyr[pyr_level].shape, intrinsics_pyr[pyr_level], ...)
                T_last2cur, _ = pose_estimator.estimate_gn(img_pyr[pyr_level], depth_pyr[pyr_level], model_img_pyr[pyr_level], model, init_pose=T_last2cur)

            pose = self.last_pose @ T_last2cur
            self.last_pose = pose
        self.last_img_pyr = img_pyr
        return self.last_pose
