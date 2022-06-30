import torch
from numpy import ndarray
import numpy as np
import cv2
from alley_oop.metrics.projected_photo_metrics import disparity_photo_loss
from alley_oop.geometry.normals import normals_from_regular_grid
from alley_oop.utils.pytorch import batched_dot_product
from alley_oop.utils.rgb2gray import rgb2gray_t
from alley_oop.geometry.pinhole_transforms import create_img_coords_t, reverse_project


class PreProcess(object):
    def __init__(self, scale, depth_min, intrinsics, dtype=torch.float32, mask_specularities:bool=True, compensate_illumination:bool=False):
        self.depth_scale = scale
        self.depth_min = depth_min
        self.intrinsics = intrinsics
        self.dtype = dtype
        self.mask_specularities = mask_specularities
        self.cmp_illumination = compensate_illumination

    def __call__(self, img:ndarray, depth:ndarray, mask:ndarray=None, img_r:ndarray=None, disp:ndarray=None):
        is_numpy = not torch.is_tensor(img)
        if is_numpy:
            # normalize img
            img = img.astype(np.float32) / 255.0
        else:
            depth = depth.numpy().squeeze()
            mask = mask.numpy().squeeze()
        # normalize depth for numerical stability
        depth = depth * self.depth_scale

        # filter depth to smooth out noisy points
        depth = cv2.bilateralFilter(depth, d=-1, sigmaColor=0.01, sigmaSpace=10)
        mask = np.ones_like(depth).astype(bool) if mask is None else mask
        if self.mask_specularities:
            mask &= self.specularity_mask(img)
        # depth clipping
        mask &= (depth > self.depth_min) & (depth < 1.0)
        # border points are usually unstable, mask them out
        mask = cv2.erode(mask.astype(np.uint8), kernel=np.ones((7, 7)))

        depth = (torch.tensor(depth).unsqueeze(0)).to(self.dtype)
        mask = (torch.tensor(mask).unsqueeze(0)).to(torch.bool)
        img = (torch.tensor(img).permute(2, 0, 1)).to(self.dtype) if is_numpy else img

        if img_r is not None:
            assert disp is not None
            if is_numpy:
                img_r = img_r.astype(np.float32) / 255.0
                img_r = (torch.tensor(img_r).permute(2, 0, 1)).to(self.dtype)
                disp = torch.tensor(disp)
            confidence = disparity_photo_loss(img.unsqueeze(0), img_r.unsqueeze(0), disp.unsqueeze(0), alpha=5.437).squeeze(0)
        else:
            # use generic depth based uncertainty model
            confidence = torch.exp(-.5 * depth ** 2 / .6 ** 2)
        # compensate illumination and transform to gray-scale image
        if self.cmp_illumination:
            img = self.compensate_illumination(rgb2gray_t(img, ax0=0), depth, self.intrinsics)

        return img, depth, mask, confidence

    def specularity_mask(self, img, spec_thr=0.96):
        """ specularities can cause issues in the photometric pose estimation.
            We can easily mask them by looking for maximum intensity values in all color channels """
        if torch.is_tensor(img):
            mask = (img.sum(dim=0) < (3*spec_thr)).numpy()
        else:
            mask = img.sum(axis=-1) < (3*spec_thr)
        return mask

    @staticmethod
    def compensate_illumination(img, depth, intrinsics):
        """
        use lambertian model and inverse-square law to estimate the albedo

        p = depth^2 * I / cos(alpha)
        where cos(alpha) = n_surface * n_illumination
        """
        v, u = torch.meshgrid(torch.arange(img.shape[1]), torch.arange(img.shape[2]))

        normals_light_src = torch.stack(
            (u - intrinsics[0, 2], v - intrinsics[1, 2], intrinsics[0, 0] * torch.ones(img.shape[-2:])))
        normals_light_src /= torch.linalg.norm(normals_light_src, ord=2, dim=0, keepdims=True)

        img_pts = create_img_coords_t(depth.shape[-2], depth.shape[-1])
        pts = reverse_project(img_pts, intrinsics, dpth=depth).T
        surface_normals = normals_from_regular_grid(pts.view((*depth.shape[-2:], 3)))
        # pad normals
        pad = torch.nn.ReplicationPad2d((0, 1, 0, 1))
        surface_normals = pad(surface_normals.permute(2, 0, 1))

        cos_alpha = batched_dot_product(normals_light_src.reshape(3, -1).T, surface_normals.reshape(3, -1).T).reshape(1,
                                                                                                                      *img.shape[
                                                                                                                       -2:])
        cos_alpha = torch.clamp(cos_alpha, 0.3, 1)
        albedo = torch.clamp(5.0*depth ** 2 * img / cos_alpha, 0.0, 1.0)  # scale factor should be between 2 and 20.0 so we select 10.0,it doesn't affect the optimization
        return albedo
