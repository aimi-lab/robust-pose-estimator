import torch
from torch.nn.functional import conv2d, pad, max_pool2d
from typing import List
from alley_oop.pose.frame_class import FrameClass


class GaussPyramid(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.dtype = torch.float32

        self._level_num = kwargs['level_num'] if 'level_num' in kwargs else 2
        self._kernel_size = kwargs['kernel_size'] if 'kernel_size' in kwargs else 5
        self._kernel_std = kwargs['kernel_std'] if 'kernel_std' in kwargs else 1.08
        self._top_level = torch.nn.Parameter(torch.as_tensor(kwargs['img'], dtype=self.dtype) if 'img' in kwargs else None)
        self._ds_step = kwargs['ds_step'] if 'ds_step' in kwargs else 2
        self._top_instrinsics = torch.nn.Parameter(kwargs['intrinsics'] if 'intrinsics' in kwargs else torch.eye(3))

        self.gauss_kernel = torch.nn.Parameter(self.gauss_2d(size=self._kernel_size, std=self._kernel_std))
        self.levels = []
        self.intrinsics_levels = []

    def forward(self, img: torch.Tensor = None, intrinsics: torch.Tensor = None) -> List[torch.Tensor]:
        """ create Gaussian image pyramid """

        # re-initialization
        self._top_level = torch.nn.Parameter(self._top_level if img is None else torch.as_tensor(img, dtype=self.dtype))
        self._top_instrinsics = torch.nn.Parameter(self._top_instrinsics if intrinsics is None else torch.as_tensor(intrinsics, dtype=self.dtype))
        self.levels = [self._top_level]
        self.intrinsics_levels = [self._top_instrinsics]

        # iterate through pyramid levels
        for _ in range(self._level_num):
            self.levels.append(self.create_next_level(self.levels[-1]))
            self.intrinsics_levels.append(self.create_next_intrinsics(self.intrinsics_levels[-1]))

        return self.levels, self.intrinsics_levels

    def create_next_level(self, x: torch.Tensor, border_mode: str = 'replicate', border_value: float = 0) -> torch.Tensor:

        channels = x.shape[1]
        padnum = self._kernel_size // 2
        padded = pad(x, (padnum, padnum, padnum, padnum), mode=border_mode, value=border_value)
        gsconv = conv2d(padded, self.gauss_kernel.repeat((channels,1,1,1)), stride=1, padding='same', groups=channels)[..., padnum:-padnum, padnum:-padnum]
        downsp = gsconv[..., 0::self._ds_step, 0::self._ds_step]

        return downsp

    def create_next_intrinsics(self, x: torch.Tensor) -> torch.Tensor:

        x = x.clone()
        x[0, 0] /= self._ds_step
        x[1, 1] /= self._ds_step
        x[:2, -1] /= self._ds_step

        return x

    def gauss_1d(self, size: int = 3, std: float = 1.0) -> torch.Tensor:
        """ returns a 1-D Gaussian """

        x = torch.arange(0, size, dtype=self.dtype) - (size - 1.0) / 2.0
        w = torch.exp(-x ** 2 / (2 * std**2))

        return w

    def gauss_2d(self, size: int = 3, std: float = 1.0) -> torch.Tensor:
        """ returns 2-D Gaussian kernel of dimensions: 1 x 1 x size x size """

        gkern1d = self.gauss_1d(size=size, std=std)
        gkern2d = torch.outer(gkern1d, gkern1d)[None, None, ...]

        return gkern2d / gkern2d.sum()

    @property
    def top_level(self):
        return self._top_level

    @property
    def top_instrinsics(self):
        return self._top_instrinsics


class FrameGaussPyramid(GaussPyramid):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dtype = torch.float32
        self._top_level_frame = kwargs['frame'] if 'frame' in kwargs else None
        self.level_frame = []

    def forward(self, frame:FrameClass=None, intrinsics:torch.tensor=None):
        # re-initialization
        self._top_level_frame = frame if frame is not None else self._top_level_frame
        assert self._top_level_frame is not None

        self.level_frame = [self._top_level_frame]
        img_pyr, _ = super().forward(self._top_level_frame.img)
        depth_pyr, _ = super().forward(self._top_level_frame.depth)
        mask_pyr = self.discrete_pyramid(self._top_level_frame.mask)
        for img, depth, intrinsics, mask in zip(img_pyr[1:], depth_pyr[1:], self.intrinsics_levels[1:], mask_pyr[1:]):
            self.level_frame.append(FrameClass(img, depth, intrinsics=intrinsics, mask=mask))
        return self.level_frame, self.intrinsics_levels

    def discrete_pyramid(self, mask: torch.Tensor) -> List[torch.Tensor]:
        """ create discrete image pyramid """

        # initialization
        levels = [mask]

        # iterate through pyramid levels
        for _ in range(self._level_num):
            levels.append(self.create_next_level_mask(levels[-1]))

        return levels

    def create_next_level_mask(self, x: torch.Tensor) -> torch.Tensor:
        return (-max_pool2d(-x.float(), kernel_size=self._ds_step, stride=self._ds_step)).to(torch.bool)
