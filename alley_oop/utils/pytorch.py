import torch
from typing import Iterable
import math
import torch.nn as nn
import torch.nn.functional as F


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


class Dilation2d(nn.Module):
    '''
    Base class for morpholigical operators
    For now, only supports stride=1, dilation=1, kernel_size H==W, and padding='same'.
    '''

    def __init__(self, channels, kernel_size=5):
        '''
        in_channels: scalar
        out_channels: scalar, the number of the morphological neure.
        kernel_size: scalar, the spatial size of the morphological neure.
        soft_max: bool, using the soft max rather the torch.max(), ref: Dense Morphological Networks: An Universal Function Approximator (Mondal et al. (2019)).
        beta: scalar, used by soft_max.
        type: str, dilation2d or erosion2d.
        '''
        super(Dilation2d, self).__init__()
        self.kernel = nn.Parameter(torch.ones(1,channels,kernel_size, kernel_size), requires_grad=False)

    def forward(self, x):
        '''
        x: tensor of shape (B,C,H,W)
        '''
        x = torch.clamp(torch.nn.functional.conv2d(x, self.kernel, padding='same'), 0, 1)

        return x


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
