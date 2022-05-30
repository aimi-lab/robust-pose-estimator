import torch
from numpy import ndarray
import numpy as np
import cv2
from alley_oop.metrics.projected_photo_metrics import disparity_photo_loss


class PreProcess(object):
    def __init__(self, scale, depth_min, dtype=torch.float32, mask_specularities:bool=True):
        self.depth_scale = scale
        self.depth_min = depth_min
        self.dtype = dtype
        self.mask_specularities = mask_specularities

    def __call__(self, img:ndarray, depth:ndarray, mask:ndarray=None, img_r:ndarray=None, disp:ndarray=None):
        # normalize img
        img = img.astype(np.float32) / 255.0
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
        img = (torch.tensor(img).permute(2, 0, 1)).to(self.dtype)

        if img_r is not None:
            assert disp is not None
            img_r = img_r.astype(np.float32) / 255.0
            img_r = (torch.tensor(img_r).permute(2, 0, 1)).to(self.dtype)
            disp = torch.tensor(disp)
            confidence = disparity_photo_loss(img.unsqueeze(0), img_r.unsqueeze(0), disp.unsqueeze(0), alpha=5.437).squeeze(0)
        else:
            # use generic depth based uncertainty model
            confidence = torch.exp(-.5 * depth ** 2 / .6 ** 2)

        return img, depth, mask, confidence

    def specularity_mask(self, img, spec_thr=0.96):
        """ specularities can cause issues in the photometric pose estimation.
            We can easily mask them by looking for maximum intensity values in all color channels """
        mask = img.sum(axis=-1) < (3*spec_thr)
        return mask

    def get_confidence(self, img, disparity):
        return disparity_photo_loss()