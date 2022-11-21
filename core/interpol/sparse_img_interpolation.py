import torch
from torch.nn.functional import conv2d, pad
from core.utils.pytorch import MedianPool2d


class SparseImgInterpolator(torch.nn.Module):
    """ Interpolate 2D tensor with sparse missing values"""
    def __init__(self, kernel_size: int, sigma: float, prior_val: float=0.0):
        """
        :param kernel_size: size of interpolation kernel
        :param sigma: sigma of Gauss interpolation kernel
        :param prior_val: prior value to fill missing values prior to interpolation
        """
        super().__init__()
        self.prior_val = prior_val
        self.kernel = torch.nn.Parameter(self.gauss_2d(kernel_size, sigma))
        self._kernel_size = kernel_size

    def forward(self, x:torch.tensor):
        """

        :param x: 2D tensor shape NxCxHxW with nan as missing values
        """
        mask = torch.isnan(x)
        x[mask] = self.prior_val
        channels = x.shape[1]
        padnum = self._kernel_size // 2
        padded = pad(x, (padnum, padnum, padnum, padnum), mode='reflect')
        gsconv = conv2d(padded, self.kernel.repeat((channels, 1, 1, 1)).to(x.dtype), stride=1, padding='same',
                        groups=channels)[..., padnum:-padnum, padnum:-padnum]
        x[mask] = gsconv[mask]
        return x

    def gauss_1d(self, size: int = 3, std: float = 1.0) -> torch.Tensor:
        """ returns a 1-D Gaussian """

        x = torch.arange(0, size) - (size - 1.0) / 2.0
        w = torch.exp(-x ** 2 / (2 * std ** 2))

        return w

    def gauss_2d(self, size: int = 3, std: float = 1.0) -> torch.Tensor:
        """ returns 2-D Gaussian kernel of dimensions: 1 x 1 x size x size """

        gkern1d = self.gauss_1d(size=size, std=std)
        gkern2d = torch.outer(gkern1d, gkern1d)[None, None, ...]
        gkern2d[:, :,size//2, size//2] = 0
        gkern2d /= gkern2d.sum()
        return gkern2d


class SparseMedianInterpolator(torch.nn.Module):
    """ Interpolate 2D tensor with sparse missing values"""
    def __init__(self, kernel_size: int, prior_val: float=0.0):
        """
        :param kernel_size: size of interpolation kernel
        :param sigma: sigma of Gauss interpolation kernel
        :param prior_val: prior value to fill missing values prior to interpolation
        """
        super().__init__()
        self.prior_val = prior_val
        self.median = MedianPool2d(kernel_size=kernel_size, same=True)

    def forward(self, x:torch.tensor):
        """

        :param x: 2D tensor shape NxCxHxW with nan as missing values
        """
        mask = torch.isnan(x)
        x[mask] = self.prior_val
        gsconv = self.median(x)
        x[mask] = gsconv[mask]
        return x