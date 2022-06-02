from ast import Pass
import torch
from alley_oop.fusion.surfel_map import SurfelMap
from alley_oop.pose.frame_class import FrameClass
from typing import Dict, Tuple
import warnings
from alley_oop.pose.rgb_pose_estimation import RGBPoseEstimator
from alley_oop.pose.icp_estimation import ICPEstimator
from torch.profiler import profile, record_function
from torchimize.functions import lsq_gna_parallel_plain, lsq_gna_parallel, lsq_lma_parallel


class RGBICPPoseEstimator(torch.nn.Module):
    """ this is an implementation of the combined geometric and photometric alignment in Elastic Fusion
        references: Newcombe et al, Kinect Fusion Chapter 3.5 (https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6162880)
                    Henry et al, Patch Volumes: Segmentation-based Consistent Mapping with RGB-D Cameras (https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6599102)
    It estimates the camera rotation and translation
"""

    def __init__(self, img_shape: Tuple, intrinsics: torch.Tensor, icp_weight: float=10.0, n_iter: int=20, Ftol: float=0.001, xtol: float=1e-8,
                 dist_thr: float=200.0/15, normal_thr: float=20, association_mode='projective', dbg_opt=False, conf_weighing: bool=False):
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
        :param conf_weighing: weight residuals (and jacobian) by src and target confidence
        """
        super(RGBICPPoseEstimator, self).__init__()
        self.rgb_estimator = RGBPoseEstimator(img_shape, intrinsics, conf_weighing=conf_weighing)
        self.icp_estimator = ICPEstimator(img_shape, intrinsics, dist_thr=dist_thr, normal_thr=normal_thr,
                                          association_mode=association_mode, conf_weighing=conf_weighing)
        assert icp_weight >= 0.0
        self.icp_weight = icp_weight
        self.wvec = torch.nn.Parameter(torch.tensor([self.icp_weight, 1]))
        self.n_iter = n_iter
        self.Ftol = Ftol
        self.xtol = xtol
        self.best_cost = (0,0,0)
        self.dbg_opt = dbg_opt

        self.optim_results = {'combined': [],'icp':[], 'rgb':[], 'icp_pts': [], 'rgb_pts': [], 'best_iter': 0, 'dx': [], 'cond': []}

    def multi_cost_fun(self, xfloat, ref_pcl, target_pcl, ref_frame, target_frame):

        icp_residuals = self.icp_estimator.residual_fun(xfloat.reshape(-1).float(), ref_pcl, target_pcl, ref_frame.mask)
        rgb_residuals = self.rgb_estimator.residual_fun(xfloat.reshape(-1).float(), ref_frame, ref_pcl, target_frame)

        if (icp_residuals.ndim == 0) | (rgb_residuals.ndim == 0):
            raise AttributeError('At least one residual vector contains zero dimensions.')

        # track optimization
        cost_icp, cost_rgb = self.icp_weight * self.icp_estimator.cost_fun(icp_residuals), self.rgb_estimator.cost_fun(rgb_residuals)
        cost = (cost_icp + cost_rgb) / len(icp_residuals)
        self.optim_results['combined'].append(cost) #if self.i % 2 == 1 else None   # only append every other cost as cost fun gets called twice per LM iteration
        self.i += 1

        self.optim_results['icp'].append(cost_icp)
        self.optim_results['rgb'].append(cost_rgb)
        self.optim_results['icp_pts'].append(len(icp_residuals))
        self.optim_results['rgb_pts'].append(len(rgb_residuals))
        self.optim_results['dx'].append(torch.linalg.norm(xfloat,ord=2))

        residuals = [icp_residuals, rgb_residuals]

        # enable tensor cost stacking by padding tensor with less values
        dims = torch.tensor([icp_residuals.size(), rgb_residuals.size()])
        size = torch.max(dims)
        tidx = torch.argmin(dims)
        residuals[tidx] = torch.cat((residuals[tidx], torch.zeros(size-residuals[tidx].shape[0], device=xfloat.device)))
    
        return torch.stack(residuals, dim=0)[None, ...]

    def multi_jaco_fun(self, xfloat, ref_pcl, target_pcl, ref_frame, target_frame):

        icp_jacobian = self.icp_estimator.jacobian(xfloat.reshape(-1).float(), ref_pcl, target_pcl, ref_frame.mask)
        rgb_jacobian = self.rgb_estimator.jacobian(xfloat.reshape(-1).float(), ref_frame, ref_pcl, target_frame)

        if (icp_jacobian.ndim == 0) | (rgb_jacobian.ndim == 0):
            raise AttributeError('At least one Jacobian contains zero dimensions.')

        jacobians = [icp_jacobian, rgb_jacobian]

        # enable tensor cost stacking by padding tensor with less values
        dims = torch.tensor([icp_jacobian.size(), rgb_jacobian.size()])
        size = torch.max(dims[:, 0])
        tidx = torch.argmin(dims[:, 0])
        jacobians[tidx] = torch.cat((jacobians[tidx], torch.zeros((size-jacobians[tidx].shape[0], jacobians[tidx].shape[-1]), device=xfloat.device)))

        return torch.stack(jacobians, dim=0)[None, ...]

    def estimate_gn(self, ref_frame: FrameClass, target_frame: FrameClass, target_pcl:SurfelMap, init_x: torch.Tensor=None):

        self.optim_results = {'combined': [],'icp':[], 'rgb':[], 'icp_pts': [], 'rgb_pts': [], 'best_iter': 0, 'dx': [], 'cond': []}
        self.i = 0

        ref_pcl = SurfelMap(frame=ref_frame, kmat=self.icp_estimator.intrinsics, ignore_mask=True)

        multi_cost_fun_args = lambda p: self.multi_cost_fun(p, ref_pcl, target_pcl, ref_frame, target_frame)
        multi_jaco_fun_args = lambda p: self.multi_jaco_fun(p, ref_pcl, target_pcl, ref_frame, target_frame)

        init_x = torch.zeros(1,6).to(ref_frame.img.device) if init_x is None else init_x
        init_x = init_x[None, ...] if len(init_x.shape) == 1 else init_x

        coeffs = lsq_gna_parallel(
                            p = init_x.double(),
                            function = multi_cost_fun_args,
                            jac_function = multi_jaco_fun_args,
                            wvec = self.wvec,
                            ftol = 1e-9,
                            ptol = 1e-9,
                            gtol = 1e-9,
                            l = 1.,                            
                            max_iter = self.n_iter,
                        )

        best_idx = torch.argmin(torch.tensor(self.optim_results['combined']))
        best_sol = coeffs[best_idx], self.optim_results['combined'][best_idx]
        self.best_cost = (self.optim_results['combined'][best_idx],
                          self.optim_results['icp'][best_idx],
                          self.optim_results['rgb'][best_idx])
        assert len(coeffs) == len(self.optim_results['combined'])

        return best_sol[0], best_sol[1], self.optim_results
