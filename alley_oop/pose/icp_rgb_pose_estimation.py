import torch
from alley_oop.geometry.lie_3d import lie_se3_to_SE3, lie_SE3_to_se3
from alley_oop.geometry.point_cloud import PointCloud
from alley_oop.pose.frame_class import FrameClass
from typing import Tuple
import warnings
from alley_oop.pose.rgb_pose_estimation import RGBPoseEstimator
from alley_oop.pose.icp_estimation import ICPEstimator


class RGBICPPoseEstimator(torch.nn.Module):
    """ this is an implementation of the combined geometric and photometric alignment in Elastic Fusion
        references: Newcombe et al, Kinect Fusion Chapter 3.5 (https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6162880)
                    Henry et al, Patch Volumes: Segmentation-based Consistent Mapping with RGB-D Cameras (https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6599102)
    It estimates the camera rotation and translation
"""

    def __init__(self, img_shape: Tuple, intrinsics: torch.Tensor, icp_weight: float=10.0, n_iter: int=20, Ftol: float=0.001, xtol: float=1e-8,
                 dist_thr: float=200.0/15, normal_thr: float=0.94, association_mode='projective'):
        """

        :param img_shape: height and width of images to process
        :param intrinsics: camera intrinsics
        :param icp_weight: weight for ICP term in linear combination of losses (0 -> RGB only)
        :param n_iter: max number of iterations of the optimizer
        :param Ftol: cost function threshold
        :param xtol: variable change threshold
        :param dist_thr: euclidean distance threshold to accept/reject point correspondences
        :param normal_thr: angular difference threshold of normals to accept/reject point correspondences
        :param association_mode: projective or euclidean correspondence in ICP
        """
        super(RGBICPPoseEstimator, self).__init__()
        self.rgb_estimator = RGBPoseEstimator(img_shape, intrinsics)
        self.icp_estimator = ICPEstimator(img_shape, intrinsics, dist_thr=dist_thr, normal_thr=normal_thr,
                                          association_mode=association_mode)
        assert icp_weight >= 0.0
        self.icp_weight = icp_weight
        self.n_iter = n_iter
        self.Ftol = Ftol
        self.xtol = xtol

    def estimate_gn(self, ref_frame: FrameClass, target_frame: FrameClass, target_pcl:PointCloud,
                    ref_mask: torch.tensor=None, target_mask: torch.tensor=None, init_pose: torch.Tensor=None):
        """ Minimize combined energy using Gauss-Newton and solving the normal equations."""
        ref_pcl = PointCloud()
        ref_pcl.from_depth(ref_frame.depth, self.icp_estimator.intrinsics, normals=ref_frame.normals)
        x = torch.zeros(6, dtype=ref_frame.depth.dtype, device=ref_frame.depth.device)
        if init_pose is not None:
            x = lie_SE3_to_se3(init_pose)
        cost = None
        converged = False
        for i in range(self.n_iter):
            # geometric
            icp_residuals = self.icp_estimator.residual_fun(x, ref_pcl, target_pcl, ref_mask)
            icp_jacobian = self.icp_estimator.jacobian(x, ref_pcl, target_pcl, ref_mask)
            # photometric
            rgb_residuals = self.rgb_estimator.residual_fun(x, ref_frame.img_gray, ref_pcl, target_frame.img_gray, target_mask)
            rgb_jacobian = self.rgb_estimator.jacobian(x, ref_frame.img_gray, ref_pcl, target_frame.img_gray, target_mask)

            # normal equations to be solved
            A = self.icp_weight*icp_jacobian.T @ icp_jacobian + rgb_jacobian.T @ rgb_jacobian
            b = self.icp_weight*icp_jacobian.T @ icp_residuals + rgb_jacobian.T @ rgb_residuals

            # Todo try several optimizer methods, this may be synchronized with CPU (cholesky, QR etc)
            x0 = torch.linalg.lstsq(A,b).solution

            #self.rgb_estimator.plot(x, ref_img, ref_depth, target_img)
            #self.icp_estimator.plot_correspondence(x, ref_img, ref_depth, target_img)
            #self.icp_estimator.plot(x, ref_pcl, target_pcl)

            cost = self.icp_weight*self.icp_estimator.cost_fun(icp_residuals) + self.rgb_estimator.cost_fun(rgb_residuals)
            if cost < self.Ftol:
                converged = True
                break
            if torch.linalg.norm(x0,ord=2) < self.xtol*(self.xtol + torch.linalg.norm(x,ord=2)):
                converged = True
                break
            x -= x0
        if not converged:
            warnings.warn(f"not converged after {self.n_iter}", RuntimeWarning)

        return lie_se3_to_SE3(x), cost
