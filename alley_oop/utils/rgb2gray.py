import torch
from typing import Union, Tuple

def rgb2gray_t(
    rgb: torch.Tensor,
    ax0: int = None,
    vec: Union[torch.Tensor, Tuple[float, float, float]] = (0.114, 0.299, 0.587),
    ) -> torch.Tensor:
    """
    Convert torch tensor to gray scale according to paper by Whelan et al.
    http://thomaswhelan.ie/Whelan16ijrr.pdf

    :param rgb: rgb image
    :param ax0: channel dimension axis
    :param vec: 
    
    """

    vec = torch.Tensor(vec).to(rgb.dtype).to(rgb.device)

    if ax0 is None and rgb.shape[-1] == 3:
        gry = (rgb @ vec)[..., None]
    elif ax0 is not None:
        gry = torch.swapaxes((torch.swapaxes(rgb, ax0, -1) @ vec)[..., None], -1, ax0)
    else:
        raise AttributeError("undefined argument 'ax0'")

    return gry


def rgb2r_t(
        rgb: torch.Tensor,
) -> torch.Tensor:
    """
    Convert RGB image to relative color components.
    r = R/(R+G+B)

    :param rgb: rgb image

    """
    assert rgb.ndim == 4
    assert rgb.shape[1] == 3
    rimg = rgb[:,0].unsqueeze(1) / rgb.sum(dim=1, keepdim=True)

    return rimg