import torch
from numpy import ndarray
import numpy as np
import cv2
from alley_oop.geometry.normals import normals_from_regular_grid
from alley_oop.utils.pytorch import batched_dot_product
from alley_oop.utils.rgb2gray import rgb2gray_t
from alley_oop.geometry.pinhole_transforms import create_img_coords_t, reverse_project


class PreProcess(object):
    """
        Pre-Process rgbd (from stereo) data for alley-oop slam

            :param scale: depth scaling for normalization
            :param depth_min: minimum depth, if depth is smaller it is masked out
            :param intrinsics: camera intrinsics
            :param dtype: data-type
            :param mask_specularities: mask-out specularities
            :param compensate_illumination: transform to albedo space (do not use with noisy depth)
            :param conf_thr: minimum depth confidence, if depth-confidence is smaller it is masked out
            """
    def __init__(self, scale:float, depth_min:float, intrinsics:torch.tensor, dtype=torch.float32, mask_specularities:bool=True,
                 compensate_illumination:bool=False, conf_thr:float=0.1):

        self.depth_scale = scale
        self.depth_min = depth_min
        self.intrinsics = intrinsics
        self.dtype = dtype
        self.mask_specularities = mask_specularities
        self.cmp_illumination = compensate_illumination
        self.conf_thr = conf_thr

    def __call__(self, img:torch.tensor, depth:torch.tensor, mask:torch.tensor=None, semantics:torch.tensor=None):
        assert torch.is_tensor(img)

        # need to go back to numpy to use opencv functions
        depth = depth.cpu().numpy().squeeze()
        mask = mask.cpu().numpy().squeeze()

        # filter depth to smooth out noisy points
        #depth = cv2.bilateralFilter(depth, d=-1, sigmaColor=0.01, sigmaSpace=10)
        mask = np.ones_like(depth).astype(bool) if mask is None else mask
        if self.mask_specularities:
            mask &= self.specularity_mask(img)
        # depth clipping
        mask &= (depth > self.depth_min) & (depth < 1.0)
        # border points are usually unstable, mask them out
        mask = cv2.erode(mask.astype(np.uint8), kernel=np.ones((11, 11)))

        depth = (torch.tensor(depth)[None,None,...]).to(self.dtype)
        mask = (torch.tensor(mask)[None,None,...]).to(torch.bool)

        # compensate illumination and transform to gray-scale image
        if self.cmp_illumination:
            img = self.compensate_illumination(rgb2gray_t(img.cpu(), ax0=0), depth, self.intrinsics)

        return img, depth, mask

    def specularity_mask(self, img, spec_thr=0.96):
        """ specularities can cause issues in the photometric pose estimation.
            We can easily mask them by looking for maximum intensity values in all color channels """
        mask = (img.sum(dim=1) < (3*spec_thr)).squeeze().numpy()
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
