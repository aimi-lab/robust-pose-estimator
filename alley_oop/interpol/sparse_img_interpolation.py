import torch
from torch.nn.functional import conv2d, pad


class SparseImgInterpolator(torch.nn.Module):
    def __init__(self, kernel_size, sigma, prior_val):
        super().__init__()
        self.prior_val = prior_val
        self.kernel = torch.nn.Parameter(self.gauss_2d(kernel_size, sigma))
        self._kernel_size = kernel_size

    def forward(self, x):
        mask = torch.isnan(x).to(x.device)
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

        return gkern2d / gkern2d.sum()
