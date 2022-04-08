import torch
import numpy as np
from torch.nn.functional import conv2d, pad
from alley_oop.geometry.lie_3d import lie_so3_to_SO3, lie_hatmap
import warnings
from alley_oop.geometry.warping import HomographyWarper
from typing import Tuple


class RotationEstimator(torch.nn.Module):
    """ this is an implementation of
    https://www.robots.ox.ac.uk/~cmei/articles/omni_track_mei.pdf
    https://www.robots.ox.ac.uk/~cmei/articles/single_view_track_ITRO.pdf

    It estimates the camera rotation between two images (assuming rotation only) using efficient second-order minimization
"""

    def __init__(self, img_shape: Tuple, intrinsics: torch.Tensor, n_iter: int=10000, res_thr: float=0.0001):
        super(RotationEstimator, self).__init__()
        assert len(img_shape) == 2
        assert intrinsics.shape == (3,3)
        self.n_iter = n_iter
        self.res_thr = res_thr
        self.intrinsics = torch.nn.Parameter(intrinsics)
        self.warper = HomographyWarper(img_shape[0], img_shape[1], normalized_coordinates=False)
        self.batch_proj_jac = torch.nn.Parameter((self._batch_jw(img_shape, intrinsics) @ self._j_rot())[:, :2]) # remove third line (zeros because we don't use w coordinates)
        self.d = torch.nn.Parameter(torch.empty(0))  # dummy device store

    def estimate(self, ref_img: torch.Tensor, target_img:torch.Tensor, mask: torch.Tensor=None):
        """ estimate rotation using efficient second-order optimization"""
        R_lr = torch.eye(3).to(self.d.device)
        residuals = None
        warped_img = None
        converged = False
        for i in range(self.n_iter):
            # compute residuals f(x)
            warped_img = self._warp_img(ref_img, R_lr, self.intrinsics)
            J = self._ems_jacobian(warped_img, target_img)
            J_pinv = torch.linalg.pinv(J)
            residuals = ((warped_img - target_img)).reshape(-1, 1)
            if mask is not None:
                residuals = mask.reshape(-1,1)*residuals
            # compute update parameter x0
            x0 = J_pinv @ residuals
            # update rotation estimate
            R_lr = R_lr @ lie_so3_to_SO3(x0.squeeze())
            # print("residuals: ", (residuals ** 2).mean().item())
            # print("x0: ", x0.numpy().squeeze())
            # import matplotlib.pyplot as plt
            # fig, ax = plt.subplots(2, 3)
            # ax[0,0].imshow(target_img)
            # ax[0,1].imshow(warped_img)
            # ax[0,2].imshow(residuals.reshape((128,160)), vmin=-1, vmax=1)
            #
            # ax[1, 0].imshow(J_pinv[0,...].reshape((128, 160)))
            # ax[1, 1].imshow(J_pinv[1,...].reshape((128, 160)))
            # ax[1, 2].imshow(J_pinv[2,...].reshape((128, 160)))
            # plt.show()
            if (residuals**2).mean() < self.res_thr:
                converged = True
                break
        if not converged:
            warnings.warn(f"EMS not converged after {self.n_iter}", RuntimeWarning)
        return R_lr, residuals, warped_img

    @staticmethod
    def _image_jacobian(img:torch.Tensor):
        sobel = [[-0.125, -0.25, -0.125], [0, 0, 0], [0.125, 0.25, 0.125]]
        if img.ndim < 4:
            img = img.unsqueeze(1)
        batch, channels, h, w= img.shape
        sobel_kernely = torch.tensor(sobel, dtype=torch.float32).unsqueeze(0).expand(1, channels, 3, 3).to(img.device)
        sobel_kernelx = torch.tensor(sobel, dtype=torch.float32).unsqueeze(0).expand(1, channels, 3, 3).transpose(2,3).to(img.device)
        x_grad = pad(conv2d(img, sobel_kernelx, stride=1, padding='valid', groups=channels)[...,1:-1,1:-1], (2,2,2,2)).reshape(batch, channels, -1)
        y_grad = pad(conv2d(img, sobel_kernely, stride=1, padding='valid', groups=channels)[...,1:-1,1:-1], (2,2,2,2)).reshape(batch, channels, -1)
        jacobian = torch.stack((x_grad, y_grad), dim=-1)
        return jacobian

    @staticmethod
    def _batch_jw(img_shape:Tuple, K:torch.Tensor):
        """ jacobian of action w with respect to R(0)
            w<KR(x)K'><(u v 1)T>
            adapted from https://www.robots.ox.ac.uk/~cmei/articles/single_view_track_ITRO.pdf where H(x) = K R(x) K'
            """
        assert K.shape == (3,3)
        u, v = torch.meshgrid(torch.arange(img_shape[-1]), torch.arange(img_shape[-2]))
        u = u.T.reshape(-1)
        v = v.T.reshape(-1)

        # intrinsics
        cu = K[0,2]
        cv = K[1,2]
        f = K[0,0]

        # fast
        J2 = torch.zeros((img_shape[-1] * img_shape[-2], 3, 9))
        J2[:, 0, 0] = u -cu
        J2[:, 0, 1] = v -cv
        J2[:, 0, 2] = f

        J2[:, 0, 6] = -1/f*(u-cu)**2
        J2[:, 0, 7] = -1/f*(u-cu)*(v-cv)
        J2[:, 0, 8] = -(u -cu)

        J2[:, 1, 3] = u -cu
        J2[:, 1, 4] = v -cv
        J2[:, 1, 5] = f

        J2[:, 1, 6] = -1/f*(u-cu)*(v-cv)
        J2[:, 1, 7] = -1/f*(v-cv)**2
        J2[:, 1, 8] = -(v-cv)

        return J2

    @staticmethod
    def _j_rot():
        """ jacobian of R(x) wrt. x -> generators of so(3) in vector form
            adapted from https://www.robots.ox.ac.uk/~cmei/articles/single_view_track_ITRO.pdf
            """
        J = torch.empty((9,3))
        J[:, 0] = torch.tensor(lie_hatmap(np.array([1,0,0]))).reshape(-1)
        J[:, 1] = torch.tensor(lie_hatmap(np.array([0,1,0]))).reshape(-1)
        J[:, 2] = torch.tensor(lie_hatmap(np.array([0,0,1]))).reshape(-1)
        return J

    def _ems_jacobian(self, img1:torch.Tensor, img2:torch.Tensor):
        """ Jacobian for efficient least squares (Jimg1 + Jimg2)/2 * Jproj in R^(h*w)x2"""
        assert img1.ndim == 2
        assert img2.ndim == 2
        h,w = img1.shape
        assert h == img2.shape[0]
        assert w == img2.shape[1]

        J_img = self._image_jacobian(torch.stack((img1, img2)))

        J_img = (J_img[0] + J_img[1])/2
        J_img = J_img.reshape(h*w,1,2).squeeze(0)
        J = J_img @ self.batch_proj_jac
        return J.squeeze(1)

    def _warp_img(self, img:torch.Tensor, R:torch.Tensor, K:torch.Tensor):
        assert R.shape == (3,3)
        assert K.shape == (3,3)
        homography = (K @ R@ torch.linalg.inv(K))
        if homography.ndim == 2:
            homography = homography.unsqueeze(0)
        if img.ndim == 2:
            img = img.unsqueeze(0).unsqueeze(0)
        return self.warper(img, torch.linalg.inv(homography)).squeeze() #ToDo why do we have to invert the homography to be consistent with opencv?



