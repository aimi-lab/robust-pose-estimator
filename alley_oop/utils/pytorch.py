import torch
from typing import Iterable
import math
import torch.nn as nn
import torch.nn.functional as F


def batched_dot_product(t1:torch.tensor, t2:torch.tensor):
    return (t1.unsqueeze(1) @ t2.unsqueeze(-1)).squeeze()


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


