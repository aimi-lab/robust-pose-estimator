import torch
import numpy as np
import warnings
from alley_oop.geometry.lie_3d import lie_se3_to_SE3
from alley_oop.geometry.pinhole_transforms import forward_project
from alley_oop.geometry.point_cloud import PointCloud
from typing import Tuple
import matplotlib.pyplot as plt
from torchimize.functions import lsq_lma
from torch.nn.functional import conv2d, pad
from alley_oop.interpol.synth_view import synth_view


class RGBPoseEstimator(torch.nn.Module):
    """ this is an implementation of the geometric alignment in Elastic Fusion
        references: Newcombe et al, Kinect Fusion Chapter 3.5 (https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6162880)
                    Henry et al, Patch Volumes: Segmentation-based Consistent Mapping with RGB-D Cameras (https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6599102)
    It estimates the camera rotation and translation between a scene and a current depth map
"""

    def __init__(self, img_shape: Tuple, intrinsics: torch.Tensor, n_iter: int=10000, res_thr: float=0.00001):
        """

        :param img_shape: height and width of images to process
        :param intrinsics: camera intrinsics
        :param n_iter: max number of iterations of the optimizer
        :param res_thr: cost function threshold
        """
        super(RGBPoseEstimator, self).__init__()
        assert len(img_shape) == 2
        assert intrinsics.shape == (3,3)
        self.img_shape = img_shape
        self.n_iter = n_iter
        self.res_thr = res_thr
        self.intrinsics = torch.nn.Parameter(intrinsics)
        self.d = torch.nn.Parameter(torch.empty(0))  # dummy device store
        self.trg_ids = None
        self.src_grid_ids = None

    @staticmethod
    def cost_fun(residuals):
        return (residuals**2).mean()

    def residual_fun(self, x, ref_img, ref_pcl, trg_img, mask=None):
        x = torch.tensor(x, dtype=ref_img.dtype, device=ref_img.device) if not torch.is_tensor(x) else x
        T_est = lie_se3_to_SE3(x[:3], x[3:], homogenous=True)
        self.warped_img, self.valid = self._warp_img(ref_img, ref_pcl, T_est)
        residuals = self.warped_img - trg_img.view(-1)[self.valid]
        if mask is not None:
            residuals = residuals[mask.view(-1)[self.valid]]
        return residuals

    def jacobian(self, x, ref_img, ref_pcl, trg_img, mask=None):
        J_img = self._image_jacobian(trg_img.unsqueeze(0).unsqueeze(0)).squeeze()
        J = J_img.unsqueeze(1) @ self.j_wt(ref_pcl.pts)
        J = J[self.valid]
        if mask is not None:
            J = J[mask.view(-1)[self.valid]]
        return J.squeeze()

    def estimate_lm(self, ref_img: torch.Tensor, ref_depth: torch.Tensor, target_img: torch.Tensor, mask: torch.Tensor=None):
        """ Levenberg-Marquard estimation."""
        ref_pcl = PointCloud()
        ref_pcl.from_depth(ref_depth, self.intrinsics)
        x_list, eps = lsq_lma(torch.zeros(6).to(ref_depth.device).to(ref_depth.dtype), self.residual_fun, self.jacobian,
                              args=(ref_img, ref_pcl, target_img, mask,), max_iter=self.n_iter, tol=self.res_thr)

        x = x_list[-1]
        cost = self.cost_fun(self.residual_fun(x, ref_img, ref_pcl, target_img, mask))
        return lie_se3_to_SE3(x[:3], x[3:], homogenous=True), cost

    def plot(self, x, ref_img, ref_depth, target_img):
        R, t = lie_se3_to_SE3(x[:3].cpu(), x[3:].cpu())
        warped_img = synth_view(ref_img.unsqueeze(0).unsqueeze(0).cpu().float(), ref_depth.unsqueeze(0).float().cpu(), R.float(),
                                t.unsqueeze(1).float(), self.intrinsics.float().cpu()).squeeze()

        fig, ax = plt.subplots(1,3)
        ax[0].imshow(ref_img.cpu())
        ax[0].set_title('reference')
        ax[1].imshow(target_img.cpu())
        ax[1].set_title('target')
        ax[2].imshow(warped_img.cpu())
        ax[2].set_title('estimated')
        for a in ax:
            a.axis('off')
        plt.show()

    @staticmethod
    def _image_jacobian(img: torch.Tensor):
        sobel = [[-0.125, -0.25, -0.125], [0, 0, 0], [0.125, 0.25, 0.125]]
        batch, channels, h, w = img.shape
        sobel_kernely = torch.tensor(sobel, dtype=torch.float32).unsqueeze(0).expand(1, channels, 3, 3).to(img.device)
        sobel_kernelx = torch.tensor(sobel, dtype=torch.float32).unsqueeze(0).expand(1, channels, 3, 3).transpose(2,
                                                                                                                  3).to(
            img.device)
        x_grad = pad(conv2d(img, sobel_kernelx.to(img.dtype), stride=1, padding='valid', groups=channels)[..., 1:-1, 1:-1],
                     (2, 2, 2, 2)).reshape(batch, channels, -1)
        y_grad = pad(conv2d(img, sobel_kernely.to(img.dtype), stride=1, padding='valid', groups=channels)[..., 1:-1, 1:-1],
                     (2, 2, 2, 2)).reshape(batch, channels, -1)
        jacobian = torch.stack((x_grad, y_grad), dim=-1)
        return jacobian

    def j_wt(self, points3d):
        # jacobian of projection and transform for se(3) (J_w*J_T)
        J = torch.zeros((len(points3d), 2, 6), dtype=points3d.dtype).to(points3d.device)
        x = points3d[:, 0]
        y = points3d[:, 1]
        zinv = 1/points3d[:, 2]
        zinv2 = zinv**2
        J[:, 0, 0] = -self.intrinsics[0,0]*x*y*zinv2
        J[:, 0, 1] = self.intrinsics[0,0]*(1+x**2*zinv2)
        J[:, 0, 2] = -self.intrinsics[0,0]*y*zinv
        J[:, 0, 3] = self.intrinsics[0,0]*zinv
        J[:, 0, 5] = -self.intrinsics[0,0]*x*zinv2
        J[:, 1, 0] = -self.intrinsics[0,0]*(1+y**2*zinv2)
        J[:, 1, 1] = -J[:, 0, 0]
        J[:, 1, 2] = self.intrinsics[0,0]*x*zinv
        J[:, 1, 4] = self.intrinsics[0,0]*zinv
        J[:, 1, 5] = -self.intrinsics[0,0]*y*zinv2

        return J

    def _warp_img(self, img:torch.Tensor, pcl:PointCloud, T:torch.Tensor):
        assert T.shape == (4,4)
        # transform and project
        # Note that, we implicitly perform nearest neighbour interpolation which is not a continuous function. This
        # requires careful choice of optimization parameters and (and step parameter for automatic estimation of the Jacobian)
        pts = torch.vstack([pcl.pts.T, torch.ones(pcl.pts.shape[0], device=pcl.pts.device, dtype=pcl.pts.dtype)])
        img_pts = forward_project(pts, self.intrinsics, T[:3,:3], T[:3,3,None]).long().T
        # filter points that are not in the image
        valid = (img_pts[:, 1] < img.shape[0]) & (img_pts[:, 0] < img.shape[1]) & (
                    img_pts[:, 1] > 0) & (img_pts[:, 0] > 0)
        ids = img_pts.long()[valid][:, 1], img_pts.long()[valid][:, 0]
        return img[ids], valid
