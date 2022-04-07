import torch
from torch.nn.functional import conv2d
import numpy as np
from alley_oop.geometry.lie_3d import lie_so3_to_SO3, lie_hatmap
import warnings


class RotationEstimator(object):
    """ this is an implementation of
    https://www.robots.ox.ac.uk/~cmei/articles/omni_track_mei.pdf
    https://www.robots.ox.ac.uk/~cmei/articles/single_view_track_ITRO.pdf

    It estimates the camera rotation between two images (assuming rotation only) using efficient second-order minimization
"""

    def __init__(self, img_shape, intrinsics, n_iter=20, res_thr=0.00001):
        self.n_iter = n_iter
        self.res_thr = res_thr
        self.intrinsics = intrinsics
        self.batch_proj_jac = (self._batch_jw(img_shape, intrinsics) @ self._j_rot())[:, :2] # remove third line (zeros because we don't use w coordinates)

    def estimate(self, ref_img, target_img):
        R_lr = torch.eye(3)
        residuals = None
        warped_img = None
        converged = False
        for i in range(self.n_iter):
            # compute residuals f(x)
            warped_img = self._warp_img(ref_img, R_lr, self.intrinsics)
            J = self._ems_jacobian(warped_img, target_img)
            J_pinv = torch.linalg.pinv(J)
            residuals = ((warped_img - target_img)).reshape(-1, 1)
            # compute update parameter x0
            x0 = -J_pinv @ residuals
            # update rotation estimate
            R_lr = R_lr @ self._so3(x0)

            if residuals.mean() < self.res_thr:
                converged = True
                break
        if not converged:
            warnings.warn(f"EMS not converged after {self.n_iter}", RuntimeWarning)
        return R_lr, residuals, warped_img

    @staticmethod
    def _image_jacobian(img):
        sobel = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
        if img.ndim < 4:
            img = img.unsqueeze(1)
        batch, channels, h, w= img.shape
        sobel_kernelx = torch.tensor(sobel, dtype=torch.float32).unsqueeze(0).expand(1, channels, 3, 3)
        sobel_kernely = torch.tensor(sobel, dtype=torch.float32).unsqueeze(0).expand(1, channels, 3, 3).transpose(2,3)
        x_grad = conv2d(img, sobel_kernelx, stride=1, padding=1, groups=channels).reshape(batch, channels, -1)
        y_grad = conv2d(img, sobel_kernely, stride=1, padding=1, groups=channels).reshape(batch, channels, -1)
        jacobian = torch.stack((x_grad, y_grad), dim=-1)
        return jacobian

    @staticmethod
    def _batch_jw(img_shape, K):
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

    def _ems_jacobian(self, img1, img2):
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

    @staticmethod
    def _so3(x):
        return torch.tensor(lie_so3_to_SO3(x.numpy().squeeze())).float()

    @staticmethod
    def _warp_img(img, R, K):
        import cv2
        img2cv = cv2.warpPerspective(img.numpy(), (K @ R@ torch.linalg.inv(K)).numpy().squeeze(), (img.shape[1], img.shape[0]))
        return torch.tensor(img2cv)











