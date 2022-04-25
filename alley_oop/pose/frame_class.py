import torch
from alley_oop.geometry.normals import normals_from_regular_grid
from typing import Union
from alley_oop.utils.rgb2gray import rgb2gray_t
from alley_oop.geometry.pinhole_transforms import create_img_coords_t, reverse_project


class FrameClass:
    """
        Class containing image, depth and normals
    """
    def __init__(self, img: torch.Tensor, depth: torch.Tensor, normals: torch.Tensor=None, intrinsics: torch.Tensor=None):
        """

        :param img: RGB image in range (0, 1) with shape Nx3xHxW
        :param depth: depth map in mm with shape Nx1xHxW
        :param normals: surface normals with shape Nx3xHxW (optional)
        :param intrinsics: camera intrinsics for normal computation (optional if normals provided)
        """

        assert img.ndim == 4
        assert depth.ndim == 4

        self.img = img
        self.img_gray = rgb2gray_t(img, ax0=1)
        self.depth = depth

        if normals is not None:
            assert normals.ndim == 4
            self.normals = normals
        else:
            assert intrinsics is not None
            rmat = torch.eye(3).to(depth.dtype).to(depth.device)
            tvec = torch.zeros((3,1)).to(depth.dtype).to(depth.device)
            img_pts = create_img_coords_t(depth.shape[-2], depth.shape[-1]).to(depth.dtype).to(depth.device)
            pts = reverse_project(img_pts, intrinsics, rmat, tvec, dpth=depth.squeeze()).T
            normals = normals_from_regular_grid(pts.view((*self.depth.shape[-2:], 3)))
            # pad normals
            pad = torch.nn.ReplicationPad2d((0, 1, 0, 1))
            self.normals = pad(normals.permute(2, 1, 0))

    def to(self, dev_or_type: Union[torch.device, torch.dtype]):
        self.img = self.img.to(dev_or_type)
        self.img_gray = self.img_gray.to(dev_or_type)
        self.depth = self.depth.to(dev_or_type)
        self.normals = self.normals.to(dev_or_type)
        return self

    @property
    def shape(self):
        return self.img.shape[-2:]
