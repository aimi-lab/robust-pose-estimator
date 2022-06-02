import torch
import numpy as np
import warnings
from alley_oop.geometry.lie_3d import lie_se3_to_SE3
from alley_oop.geometry.pinhole_transforms import forward_project
from alley_oop.fusion.surfel_map import SurfelMap
from typing import Tuple
import matplotlib.pyplot as plt
from torch.nn.functional import conv2d, pad
from alley_oop.interpol.synth_view import synth_view
from alley_oop.pose.frame_class import FrameClass
from torchimize.functions import lsq_lma


class RGBPoseEstimator(torch.nn.Module):
    """ this is an implementation of the geometric alignment in Elastic Fusion
        references: Newcombe et al, Kinect Fusion Chapter 3.5 (https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6162880)
                    Henry et al, Patch Volumes: Segmentation-based Consistent Mapping with RGB-D Cameras (https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6599102)
    It estimates the camera rotation and translation between a scene and a current depth map
"""

    def __init__(self, img_shape: Tuple, intrinsics: torch.Tensor, n_iter: int=10000, Ftol: float=0.00001,
                 xtol: float=1e-8, conf_weighing: bool=False):
        """

        :param img_shape: height and width of images to process
        :param intrinsics: camera intrinsics
        :param n_iter: max number of iterations of the optimizer
        :param Ftol: cost function threshold
        :param xtol: variable change threshold
        :param conf_weighing: weight residuals (and jacobian) by src and target confidence
        """
        super(RGBPoseEstimator, self).__init__()
        assert len(img_shape) == 2
        assert intrinsics.shape == (3,3)
        self.img_shape = img_shape
        self.n_iter = n_iter
        self.Ftol = Ftol
        self.xtol = xtol
        self.intrinsics = torch.nn.Parameter(intrinsics)
        self.d = torch.nn.Parameter(torch.empty(0))  # dummy device store
        self.trg_ids = None
        self.src_grid_ids = None
        self.conf_weighing = conf_weighing

    @staticmethod
    def cost_fun(residuals):
        return (residuals**2).mean()

    def residual_fun(self, x, ref_frame:FrameClass, ref_pcl:SurfelMap, trg_frame:FrameClass):
        T_est = lie_se3_to_SE3(x)
        self.src_ids, self.trg_ids = self._warp_frame(ref_frame, ref_pcl, T_est)
        residuals = ref_frame.img_gray.view(-1)[self.src_ids] - trg_frame.img_gray.view(-1)[self.trg_ids]
        mask = ref_frame.mask.view(-1)[self.src_ids] & trg_frame.mask.view(-1)[self.trg_ids]
        # weight residuals by confidences
        if self.conf_weighing:
            residuals = torch.sqrt(ref_frame.confidence.view(-1)[self.src_ids]* trg_frame.confidence.view(-1)[self.trg_ids]) * residuals
        residuals = residuals[mask]
        return residuals

    def jacobian(self, x, ref_frame:FrameClass, ref_pcl:SurfelMap, trg_frame:FrameClass):
        J_img = self._image_jacobian(ref_frame.img_gray).squeeze()
        J = J_img.unsqueeze(1) @ self.j_wt(ref_pcl.opts.T)
        J = J[self.src_ids].squeeze()
        mask = ref_frame.mask.view(-1)[self.src_ids] & trg_frame.mask.view(-1)[self.trg_ids]
        # weight residuals by confidences
        if self.conf_weighing:
            J = torch.sqrt(ref_frame.confidence.view(-1)[self.src_ids] * trg_frame.confidence.view(-1)[self.trg_ids])[:, None] * J
        J = J[mask]
        return J

    def plot(self, x, ref_img, ref_depth, target_img, cost):
        T = lie_se3_to_SE3(x.cpu())
        warped_img = synth_view(ref_img.cpu().float(), ref_depth.float().cpu(), T[:3,:3].float(),
                                T[:3,3].unsqueeze(1).float(), self.intrinsics.float().cpu()).squeeze()

        fig, ax = plt.subplots(1,3)
        ax[0].imshow(ref_img.squeeze().cpu(), vmin=0, vmax=1)
        ax[0].set_title(f'reference: cost={cost}')
        ax[1].imshow(target_img.squeeze().cpu(), vmin=0, vmax=1)
        ax[1].set_title('target')
        ax[2].imshow(warped_img.squeeze().cpu(), vmin=0, vmax=1)
        ax[2].set_title('estimated')
        for a in ax:
            a.axis('off')
        plt.show()

    @staticmethod
    def _image_jacobian(img: torch.Tensor):
        sobel = [[-0.125, -0.25, -0.125], [0, 0, 0], [0.125, 0.25, 0.125]]
        batch, channels, h, w = img.shape
        sobel_kernely = torch.tensor(sobel, dtype=img.dtype, device=img.device).unsqueeze(0).expand(1, channels, 3, 3)
        sobel_kernelx = torch.tensor(sobel, dtype=img.dtype, device=img.device).unsqueeze(0).expand(1, channels, 3, 3).transpose(2,
                                                                                                                  3)
        x_grad = pad(conv2d(img, sobel_kernelx, stride=1, padding='valid', groups=channels)[..., 1:-1, 1:-1],
                     (2, 2, 2, 2)).reshape(batch, channels, -1)
        y_grad = pad(conv2d(img, sobel_kernely, stride=1, padding='valid', groups=channels)[..., 1:-1, 1:-1],
                     (2, 2, 2, 2)).reshape(batch, channels, -1)
        jacobian = torch.stack((x_grad, y_grad), dim=-1)
        return jacobian

    def j_wt(self, points3d):
        # jacobian of projection and transform for se(3) (J_w*J_T)
        J = torch.zeros((len(points3d), 2, 6), dtype=points3d.dtype, device=points3d.device)
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

    def _warp_frame(self, frame:FrameClass, pcl:SurfelMap, T:torch.Tensor):
        assert T.shape == (4,4)
        # transform and project
        # Note that, we implicitly perform nearest neighbour interpolation which is not a continuous function. This
        # requires careful choice of optimization parameters and (and step parameter for automatic estimation of the Jacobian)
        pts = torch.vstack([pcl.opts, torch.ones(pcl.opts.shape[1], device=pcl.opts.device, dtype=pcl.opts.dtype)])
        img_pts = forward_project(pts, self.intrinsics, T[:3,:3], T[:3,3,None]).long().T
        # filter points that are not in the image
        valid = (img_pts[:, 1] < frame.img_gray.shape[-2]) & (img_pts[:, 0] < frame.img_gray.shape[-1]) & (
                    img_pts[:, 1] > 0) & (img_pts[:, 0] > 0)
        ids = img_pts.long()[valid][:, 1]*frame.shape[1]+ img_pts.long()[valid][:, 0]
        return ids, valid
