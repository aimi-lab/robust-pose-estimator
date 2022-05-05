import torch
from alley_oop.geometry.normals import normals_from_regular_grid
from typing import Union
from alley_oop.utils.rgb2gray import rgb2gray_t
from alley_oop.geometry.pinhole_transforms import create_img_coords_t, reverse_project


class FrameClass:
    """
        Class containing image, depth and normals
    """
    def __init__(self, img: torch.Tensor, depth: torch.Tensor, normals: torch.Tensor=None,
                 intrinsics: torch.Tensor=None, mask: torch.Tensor=None):
        """

        :param img: RGB image in range (0, 1) with shape Nx3xHxW or gray-scale Nx1xHxW
        :param depth: depth map in mm with shape Nx1xHxW
        :param normals: surface normals with shape Nx3xHxW (optional)
        :param intrinsics: camera intrinsics for normal computation (optional if normals provided)
        """
        assert img.ndim == 4
        assert depth.ndim == 4
        self.img = img.contiguous()

        if img.shape[1] == 3:
            self.img_gray = rgb2gray_t(self.img, ax0=1).contiguous()
        else:
            self.img_gray = self.img

        if mask is None:
            mask = torch.ones_like(self.img_gray, dtype=torch.bool)
        self.mask = mask
        self.depth = depth.contiguous()

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
            self.normals = pad(normals.permute(2, 0, 1)).contiguous().unsqueeze(0)

        assert self.img.shape[-2:] == self.img_gray.shape[-2:]
        assert self.img.shape[-2:] == self.depth.shape[-2:]
        assert self.img.shape[-2:] == self.mask.shape[-2:]
        assert self.img.shape[-2:] == self.normals.shape[-2:]

    def to(self, dev_or_type: Union[torch.device, torch.dtype]):
        self.img = self.img.to(dev_or_type)
        self.img_gray = self.img_gray.to(dev_or_type)
        self.depth = self.depth.to(dev_or_type)
        self.normals = self.normals.to(dev_or_type)
        self.mask = self.mask.to(dev_or_type)
        return self

    @property
    def shape(self):
        return self.img.shape[-2:]

    def plot(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1,4)
        img, img_gray, depth, mask = self.to_numpy()
        ax[0].imshow(img)
        ax[1].imshow(img_gray)
        ax[2].imshow(depth)
        ax[3].imshow(mask, vmin=0, vmax=1)
        for a in ax:
            a.axis('off')
        plt.show()

    def to_numpy(self):
        img = self.img.detach().cpu().permute(0,2,3,1).squeeze().numpy()
        img_gray = self.img_gray.detach().cpu().squeeze().numpy()
        depth = self.depth.detach().cpu().squeeze().numpy()
        mask = self.mask.detach().cpu().squeeze().numpy()
        return img, img_gray, depth, mask
