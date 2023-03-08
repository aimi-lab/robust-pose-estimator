import torch
from typing import Iterable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple


def batched_dot_product(t1:torch.tensor, t2:torch.tensor):
    if t1.ndim == 2:
        # 3xn
        return (t1.unsqueeze(1) @ t2.unsqueeze(-1)).squeeze(0).squeeze(0)
    else:
        # nx3xm
        n,c,m = t1.shape
        tb1 = t1.permute(0, 2, 1).reshape(-1, c)
        tb2 = t2.permute(0, 2, 1).reshape(-1, c)
        b = torch.bmm(tb1.unsqueeze(1), tb2.unsqueeze(-1)).squeeze(0)
        return b.view(n, m)


def beye(shape:Iterable, device=None, dtype=None):
    assert len(shape) == 2
    return torch.eye(shape[1], device=device, dtype=dtype).repeat(shape[0],1,1)


def beye_like(t:torch.tensor):
    assert t.ndim == 3
    return beye(t.shape[:2], t.device, t.dtype)


def grid_sample(image, optical):
    # this is slower than the official implementation but twice differentiable
    N, C, IH, IW = image.shape
    _, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]

    ix = ((ix + 1) / 2) * (IW - 1)
    iy = ((iy + 1) / 2) * (IH - 1)
    with torch.no_grad():
        ix_nw = torch.floor(ix)
        iy_nw = torch.floor(iy)
        ix_ne = ix_nw + 1
        iy_ne = iy_nw
        ix_sw = ix_nw
        iy_sw = iy_nw + 1
        ix_se = ix_nw + 1
        iy_se = iy_nw + 1

    nw = (ix_se - ix) * (iy_se - iy)
    ne = (ix - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix) * (iy - iy_ne)
    se = (ix - ix_nw) * (iy - iy_nw)

    with torch.no_grad():
        torch.clamp(ix_nw, 0, IW - 1, out=ix_nw)
        torch.clamp(iy_nw, 0, IH - 1, out=iy_nw)

        torch.clamp(ix_ne, 0, IW - 1, out=ix_ne)
        torch.clamp(iy_ne, 0, IH - 1, out=iy_ne)

        torch.clamp(ix_sw, 0, IW - 1, out=ix_sw)
        torch.clamp(iy_sw, 0, IH - 1, out=iy_sw)

        torch.clamp(ix_se, 0, IW - 1, out=ix_se)
        torch.clamp(iy_se, 0, IH - 1, out=iy_se)

    image = image.view(N, C, IH * IW)

    nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1))
    ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1))
    sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1))
    se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1))

    out_val = (nw_val.view(N, C, H, W) * nw.view(N, 1, H, W) +
               ne_val.view(N, C, H, W) * ne.view(N, 1, H, W) +
               sw_val.view(N, C, H, W) * sw.view(N, 1, H, W) +
               se_val.view(N, C, H, W) * se.view(N, 1, H, W))

    return out_val


class MedianPool2d(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.

    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """

    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super(MedianPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding

    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd,
        # would likely be more efficient to implement from scratch at C/Cuda level
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x

def image_gradient(img: torch.Tensor):
    sobel = [[-0.125, -0.25, -0.125], [0, 0, 0], [0.125, 0.25, 0.125]]
    batch, channels, h, w = img.shape
    sobel_kernely = torch.tensor(sobel, dtype=img.dtype, device=img.device).unsqueeze(0).expand(1, channels, 3, 3)
    sobel_kernelx = torch.tensor(sobel, dtype=img.dtype, device=img.device).unsqueeze(0).expand(1, channels, 3,
                                                                                                3).transpose(2,
                                                                                                             3)
    x_grad = pad(conv2d(img, sobel_kernelx, stride=1, padding='valid', groups=channels)[..., 1:-1, 1:-1],
                 (2, 2, 2, 2)).reshape(batch, channels, -1)
    y_grad = pad(conv2d(img, sobel_kernely, stride=1, padding='valid', groups=channels)[..., 1:-1, 1:-1],
                 (2, 2, 2, 2)).reshape(batch, channels, -1)
    gradient = torch.stack((x_grad, y_grad), dim=-1)
    return gradient


def skewmat(vec: torch.Tensor = None) -> torch.Tensor:
    """
    create hat-map in so(3) from Euler vector in R^3
    :param wvec: Euler vector in R^3
    :return: hat-map in so(3)
    """

    if vec.ndim == 1:
        vec = vec.unsqueeze(0)
    assert vec.shape[1] == 3, 'argument must be a 3-vector'

    W_row0 = torch.tensor([0, 0, 0, 0, 0, 1, 0, -1, 0.0], device=vec.device, dtype=vec.dtype).view(3, 3)
    W_row1 = torch.tensor([0, 0, -1, 0, 0, 0, 1, 0, 0.0], device=vec.device, dtype=vec.dtype).view(3, 3)
    W_row2 = torch.tensor([0, 1, 0, -1, 0, 0, 0, 0, 0.0], device=vec.device, dtype=vec.dtype).view(3, 3)

    wmat = torch.stack(
        [torch.matmul(vec, W_row0.T), torch.matmul(vec, W_row1.T), torch.matmul(vec, W_row2.T)], dim=-1)

    return wmat