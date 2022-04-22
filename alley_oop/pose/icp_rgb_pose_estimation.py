import torch
from alley_oop.geometry.lie_3d import lie_se3_to_SE3
from alley_oop.geometry.point_cloud import PointCloud
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

    def __init__(self, img_shape: Tuple, intrinsics: torch.Tensor, icp_weight: float=10.0, n_iter: int=20, res_thr: float=0.001,
                 dist_thr: float=200.0/15, normal_thr: float=0.94, association_mode='projective'):
        """

        :param img_shape: height and width of images to process
        :param intrinsics: camera intrinsics
        :param icp_weight: weight for ICP term in linear combination of losses (0 -> RGB only)
        :param n_iter: max number of iterations of the optimizer
        :param res_thr: cost function threshold
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
        self.res_thr = res_thr

    def estimate_gn(self, ref_img: torch.Tensor, ref_depth: torch.Tensor, target_img: torch.Tensor, target_pcl:PointCloud,
                    ref_mask: torch.tensor=None,
                    target_mask: torch.tensor=None, xtol=1e-4):
        """ Minimize combined energy using Gauss-Newton and solving the normal equations."""
        ref_pcl = PointCloud()
        ref_pcl.from_depth(ref_depth, self.icp_estimator.intrinsics)
        x = torch.zeros(6, dtype=ref_img.dtype, device=ref_img.device)
        cost = None
        converged = False
        for i in range(self.n_iter):
            # geometric
            icp_residuals = self.icp_estimator.residual_fun(x, ref_pcl, target_pcl, ref_mask)
            icp_jacobian = self.icp_estimator.jacobian(x, ref_pcl, target_pcl, ref_mask)
            # photometric
            rgb_residuals = self.rgb_estimator.residual_fun(x, ref_img, ref_pcl, target_img, target_mask)
            rgb_jacobian = self.rgb_estimator.jacobian(x, ref_img, ref_pcl, target_img, target_mask)

            # normal equations to be solved
            A = self.icp_weight*icp_jacobian.T @ icp_jacobian + rgb_jacobian.T @ rgb_jacobian
            b = self.icp_weight*icp_jacobian.T @ icp_residuals + rgb_jacobian.T @ rgb_residuals

            # Todo try several optimizer methods, this may be synchronized with CPU (cholesky, QR etc)
            x0 = torch.linalg.lstsq(A,b).solution
            x -= x0

            #self.rgb_estimator.plot(x, ref_img, ref_depth, target_img)
            #self.icp_estimator.plot_correspondence(x, ref_img, ref_depth, target_img)
            #self.icp_estimator.plot(x, ref_pcl, target_pcl)

            cost = self.icp_weight*self.icp_estimator.cost_fun(icp_residuals) + self.rgb_estimator.cost_fun(rgb_residuals)
            if cost < self.res_thr:
                converged = True
                break
            if torch.linalg.norm(x0,ord=2) < xtol*(xtol + torch.linalg.norm(x,ord=2)):
                converged = True
                break

        if not converged:
            warnings.warn(f"not converged after {self.n_iter}", RuntimeWarning)

        return lie_se3_to_SE3(x[:3], x[3:], homogenous=True), cost
