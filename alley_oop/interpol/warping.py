import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional


def create_meshgrid(
        height: int,
        width: int,
        normalized_coordinates: Optional[bool] = True) -> torch.Tensor:
    """Generates a coordinate grid for an image.

    When the flag `normalized_coordinates` is set to True, the grid is
    normalized to be in the range [-1,1] to be consistent with the pytorch
    function grid_sample.
    http://pytorch.org/docs/master/nn.html#torch.nn.functional.grid_sample

    Args:
        height (int): the image height (rows).
        width (int): the image width (cols).
        normalized_coordinates (Optional[bool]): whether to normalize
          coordinates in the range [-1, 1] in order to be consistent with the
          PyTorch function grid_sample.

    Return:
        torch.Tensor: returns a grid tensor with shape :math:`(1, H, W, 2)`.
    """
    # generate coordinates
    xs: Optional[torch.Tensor] = None
    ys: Optional[torch.Tensor] = None
    if normalized_coordinates:
        xs = torch.linspace(-1, 1, width)
        ys = torch.linspace(-1, 1, height)
    else:
        xs = torch.linspace(0, width - 1, width)
        ys = torch.linspace(0, height - 1, height)
    # generate grid by stacking coordinates
    base_grid: torch.Tensor = torch.stack(
        torch.meshgrid([xs, ys])).transpose(1, 2)  # 2xHxW
    return torch.unsqueeze(base_grid, dim=0).permute(0, 2, 3, 1)  # 1xHxWx2


def convert_points_to_homogeneous(points: torch.Tensor) -> torch.Tensor:
    r"""Function that converts points from Euclidean to homogeneous space.
    """
    if not isinstance(points, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(points)))
    if len(points.shape) < 2:
        raise ValueError("Input must be at least a 2D tensor. Got {}".format(
            points.shape))

    return torch.nn.functional.pad(points, [0, 1], "constant", 1.0)


def convert_points_from_homogeneous(
        points: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    r"""Function that converts points from homogeneous to Euclidean space.
    """
    if not isinstance(points, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(points)))
    if len(points.shape) < 2:
        raise ValueError("Input must be at least a 2D tensor. Got {}".format(
            points.shape))

    # we check for points at infinity
    z_vec: torch.Tensor = points[..., -1:]

    # set the results of division by zeror/near-zero to 1.0
    # follow the convention of opencv:
    # https://github.com/opencv/opencv/pull/14411/files
    scale: torch.Tensor = torch.where(
        torch.abs(z_vec) > eps,
        torch.tensor(1.) / z_vec,
        torch.ones_like(z_vec))

    return scale * points[..., :-1]


def transform_points(trans_01: torch.Tensor,
                     points_1: torch.Tensor) -> torch.Tensor:
    r"""Function that applies transformations to a set of points.

    Args:
        trans_01 (torch.Tensor): tensor for transformations of shape
          :math:`(B, D+1, D+1)`.
        points_1 (torch.Tensor): tensor of points of shape :math:`(B, N, D)`.
    Returns:
        torch.Tensor: tensor of N-dimensional points.

    Shape:
        - Output: :math:`(B, N, D)`
    """
    if not torch.is_tensor(trans_01) or not torch.is_tensor(points_1):
        raise TypeError("Input type is not a torch.Tensor")
    if not trans_01.device == points_1.device:
        raise TypeError("Tensor must be in the same device")
    if not trans_01.shape[0] == points_1.shape[0]:
        raise ValueError("Input batch size must be the same for both tensors")
    if not trans_01.shape[-1] == (points_1.shape[-1] + 1):
        raise ValueError("Last input dimensions must differe by one unit")
    # to homogeneous
    points_1_h = convert_points_to_homogeneous(points_1)  # BxNxD+1
    # transform coordinates
    points_0_h = torch.matmul(
        trans_01.unsqueeze(1), points_1_h.unsqueeze(-1))
    points_0_h = torch.squeeze(points_0_h, dim=-1)
    # to euclidean
    points_0 = convert_points_from_homogeneous(points_0_h)  # BxNxD
    return points_0


class HomographyWarper(nn.Module):
    r"""Warps image patches or tensors by homographies.

    .. math::

        X_{dst} = H_{src}^{\{dst\}} * X_{src}

    Args:
        height (int): The height of the image to warp.
        width (int): The width of the image to warp.
        mode (str): interpolation mode to calculate output values
          'bilinear' | 'nearest'. Default: 'bilinear'.
        padding_mode (str): padding mode for outside grid values
          'zeros' | 'border' | 'reflection'. Default: 'zeros'.
        normalized_coordinates (bool): wether to use a grid with
          normalized coordinates.
    """

    def __init__(
            self,
            height: int,
            width: int,
            mode: str = 'bilinear',
            padding_mode: str = 'zeros',
            normalized_coordinates: bool = True) -> None:
        super(HomographyWarper, self).__init__()
        self.width: int = width
        self.height: int = height
        self.mode: str = mode
        self.padding_mode: str = padding_mode
        self.normalized_coordinates: bool = normalized_coordinates

        # create base grid to compute the flow
        self.grid: torch.Tensor = nn.Parameter(create_meshgrid(
            height, width, normalized_coordinates=normalized_coordinates), requires_grad=False)

    def warp_grid(self, dst_homo_src: torch.Tensor) -> torch.Tensor:
        r"""Computes the grid to warp the coordinates grid by an homography.

        Args:
            dst_homo_src (torch.Tensor): Homography or homographies (stacked) to
                              transform all points in the grid. Shape of the
                              homography has to be :math:`(N, 3, 3)`.

        Returns:
            torch.Tensor: the transformed grid of shape :math:`(N, H, W, 2)`.
        """
        batch_size: int = dst_homo_src.shape[0]
        # expand grid to match the input batch size
        grid: torch.Tensor = self.grid.expand(batch_size, -1, -1, -1)  # NxHxWx2
        if len(dst_homo_src.shape) == 3:  # local homography case
            dst_homo_src = dst_homo_src.view(batch_size, 1, 3, 3)  # NxHxWx3x3
        # perform the actual grid transformation,
        # the grid is copied to input device and casted to the same type
        flow: torch.Tensor = transform_points(
            dst_homo_src, grid)  # NxHxWx2
        return flow.view(batch_size, self.height, self.width, 2)  # NxHxWx2

    def forward(  # type: ignore
            self,
            patch_src: torch.Tensor,
            dst_homo_src: torch.Tensor) -> torch.Tensor:
        r"""Warps an image or tensor from source into reference frame.

        Args:
            patch_src (torch.Tensor): The image or tensor to warp.
                                      Should be from source.
            dst_homo_src (torch.Tensor): The homography or stack of homographies
             from source to destination. The homography assumes normalized
             coordinates [-1, 1].

        Return:
            torch.Tensor: Patch sampled at locations from source to destination.

        Shape:
            - Input: :math:`(N, C, H, W)` and :math:`(N, 3, 3)`
            - Output: :math:`(N, C, H, W)`

        Example:
            >>> input = torch.rand(1, 3, 32, 32)
            >>> homography = torch.eye(3).view(1, 3, 3)
            >>> warper = HomographyWarper(32, 32)
            >>> output = warper(input, homography)  # NxCxHxW
        """
        if not dst_homo_src.device == patch_src.device:
            raise TypeError("Patch and homography must be on the same device. \
                            Got patch.device: {} dst_H_src.device: {}."
                            .format(patch_src.device, dst_homo_src.device))
        warping_grid = self.warp_grid(dst_homo_src)
        if not self.normalized_coordinates:
            warping_grid[..., 0] = 2*warping_grid[..., 0] / patch_src.shape[-1] -1
            warping_grid[..., 1] = 2 * warping_grid[..., 1] / patch_src.shape[-2] - 1
        return F.grid_sample(patch_src,warping_grid ,
                             mode=self.mode, padding_mode=self.padding_mode)
