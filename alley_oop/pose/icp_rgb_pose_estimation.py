import torch
from alley_oop.fusion.surfel_map import SurfelMap
from alley_oop.pose.frame_class import FrameClass
from typing import Tuple
import warnings
from alley_oop.pose.rgb_pose_estimation import RGBPoseEstimator
from alley_oop.pose.icp_estimation import ICPEstimator
from torch.profiler import profile, record_function
from torchimize.functions import lsq_gna_parallel_plain


class RGBICPPoseEstimator(torch.nn.Module):
    """ this is an implementation of the combined geometric and photometric alignment in Elastic Fusion
        references: Newcombe et al, Kinect Fusion Chapter 3.5 (https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6162880)
                    Henry et al, Patch Volumes: Segmentation-based Consistent Mapping with RGB-D Cameras (https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6599102)
    It estimates the camera rotation and translation
"""

    def __init__(self, img_shape: Tuple, intrinsics: torch.Tensor, icp_weight: float=10.0, n_iter: int=20, Ftol: float=0.001, xtol: float=1e-8,
                 dist_thr: float=200.0/15, normal_thr: float=20, association_mode='projective', dbg_opt=False):
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
        self.best_cost = (0,0,0)
        self.dbg_opt = dbg_opt

    def multi_cost_fun(self, xfloat, ref_pcl, target_pcl, ref_frame, target_frame):

        icp_residuals = self.icp_estimator.residual_fun(xfloat[0, :].float(), ref_pcl, target_pcl, ref_frame.mask)
        rgb_residuals = self.rgb_estimator.residual_fun(xfloat[0, :].float(), ref_frame.img_gray, ref_pcl, target_frame.img_gray, target_frame.mask, ref_frame.mask)

        residuals = [icp_residuals, rgb_residuals]

        dims = torch.tensor([icp_residuals.size(), rgb_residuals.size()])
        size = torch.max(dims)
        tidx = torch.argmin(dims)
        residuals[tidx] = torch.cat((residuals[tidx], torch.zeros(size-residuals[tidx].shape[0])))
    
        return torch.stack(residuals, dim=0)[None, ...].double()

    def multi_jaco_fun(self, xfloat, ref_pcl, target_pcl, ref_frame, target_frame):

        icp_jacobian = self.icp_estimator.jacobian(xfloat[0, :].float(), ref_pcl, target_pcl, ref_frame.mask)
        rgb_jacobian = self.rgb_estimator.jacobian(xfloat[0, :].float(), ref_frame.img_gray, ref_pcl, target_frame.img_gray, target_frame.mask, ref_frame.mask)

        jacobians = [icp_jacobian, rgb_jacobian]

        dims = torch.tensor([icp_jacobian.size(), rgb_jacobian.size()])
        size = torch.max(dims[:, 0])
        tidx = torch.argmin(dims[:, 0])
        jacobians[tidx] = torch.cat((jacobians[tidx], torch.zeros(size-jacobians[tidx].shape[0], jacobians[tidx].shape[-1])))

        return torch.stack(jacobians, dim=0)[None, ...].double()

    def estimate_gn(self, ref_frame: FrameClass, target_frame: FrameClass, target_pcl:SurfelMap, init_x: torch.Tensor=None):

        ref_pcl = SurfelMap(frame=ref_frame, kmat=self.icp_estimator.intrinsics, ignore_mask=True)

        multi_cost_fun_args = lambda p: self.multi_cost_fun(p, ref_pcl, target_pcl, ref_frame, target_frame)
        multi_jaco_fun_args = lambda p: self.multi_jaco_fun(p, ref_pcl, target_pcl, ref_frame, target_frame)

        coeffs = lsq_gna_parallel_plain(
                            p = init_x[None, ...].double(),
                            function = multi_cost_fun_args,
                            jac_function = multi_jaco_fun_args,
                            wvec = torch.ones(2, device=init_x.device, dtype=init_x.dtype),
                            l = .1,
                            max_iter = self.n_iter,
                        )

        return coeffs, coeffs, None

    def _estimate_gn(self, ref_frame: FrameClass, target_frame: FrameClass, target_pcl:SurfelMap, init_x: torch.Tensor=None):
        """ Minimize combined energy using Gauss-Newton and solving the normal equations."""
        ref_pcl = SurfelMap(frame=ref_frame, kmat=self.icp_estimator.intrinsics, ignore_mask=True)
        x = torch.zeros(6, dtype=torch.float64, device=ref_frame.depth.device)
        if init_x is not None:
            x = init_x.double()
        optim_results = {'combined': [],'icp':[], 'rgb':[], 'icp_pts': [], 'rgb_pts': [], 'best_iter': 0, 'dx': [], 'cond': []}
        converged = False
        best_sol = [x.clone(), torch.inf*torch.ones(1, device=x.device).squeeze()]
        for i in range(self.n_iter):
            # geometric
            xfloat = x.float()
            icp_residuals = self.icp_estimator.residual_fun(xfloat, ref_pcl, target_pcl, ref_frame.mask)
            icp_jacobian = self.icp_estimator.jacobian(xfloat, ref_pcl, target_pcl, ref_frame.mask)
            # photometric
            rgb_residuals = self.rgb_estimator.residual_fun(xfloat, ref_frame.img_gray, ref_pcl, target_frame.img_gray, target_frame.mask, ref_frame.mask)
            rgb_jacobian = self.rgb_estimator.jacobian(xfloat, ref_frame.img_gray, ref_pcl, target_frame.img_gray, target_frame.mask, ref_frame.mask)

            # normal equations to be solved
            if (icp_residuals.ndim == 0) | (rgb_residuals.ndim == 0) | (icp_jacobian.ndim == 0)| (rgb_jacobian.ndim == 0):
                break
            A = self.icp_weight*icp_jacobian.T @ icp_jacobian + rgb_jacobian.T @ rgb_jacobian #
            b = self.icp_weight*icp_jacobian.T @ icp_residuals + rgb_jacobian.T @ rgb_residuals #

            x0 = torch.linalg.lstsq(A.double(),b.double(), driver='gels').solution
            cost_icp, cost_rgb = self.icp_weight * self.icp_estimator.cost_fun(icp_residuals), self.rgb_estimator.cost_fun(rgb_residuals)
            cost = (cost_icp + cost_rgb) / len(icp_residuals)
            if self.dbg_opt:
                optim_results['icp'].append(cost_icp)
                optim_results['rgb'].append(cost_rgb)
                optim_results['icp_pts'].append(len(icp_residuals))
                optim_results['rgb_pts'].append(len(rgb_residuals))
                optim_results['combined'].append(cost)
                optim_results['dx'].append(torch.linalg.norm(x0,ord=2))
                optim_results['cond'].append(torch.linalg.cond(A))

            costs = torch.stack([cost, best_sol[1]])  # if cost < best_sol[1]: best_sol = (x.clone(), cost)
            xs = torch.stack([x, best_sol[0]])
            best_ids = torch.argmin(costs)
            best_sol = [xs[best_ids], costs[best_ids]]
            # if cost < self.Ftol:
            #     converged = True
            #     break
            # if torch.linalg.norm(x0,ord=2) < self.xtol*(self.xtol + torch.linalg.norm(x,ord=2)):
            #     converged = True
            #     break
            x -= x0
        return best_sol[0], best_sol[1], optim_results
