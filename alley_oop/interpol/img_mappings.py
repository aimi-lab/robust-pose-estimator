import numpy as np
import torch
from scipy.interpolate import LinearNDInterpolator, CloughTocher2DInterpolator, NearestNDInterpolator


def img_map_scipy(
        img: np.ndarray = None, 
        ipts: np.ndarray = None, 
        npts: np.ndarray = None, 
        mode: str = 'bilinear', 
        fill_val: float = 0
    ) -> np.ndarray:

    # interpolator settings
    if mode == 'bilinear':
        InterpolClass = LinearNDInterpolator
    elif mode == 'clough':
        InterpolClass = CloughTocher2DInterpolator
    else:
        InterpolClass = NearestNDInterpolator

    nimg = np.zeros_like(img)
    for i in range(img.shape[-1]):
        interpolator = InterpolClass(ipts[:2].T, values=img[..., i].flatten(), rescale=False, fill_value=fill_val)
        nimg[..., i] = interpolator(npts[:2].T).reshape(img.shape[:2])

    return nimg


def img_map_torch(
        img: np.ndarray = None, 
        npts: np.ndarray = None, 
        mode: str = 'bilinear'
    ) -> torch.Tensor:

    # create new sample grid
    y, x = img.shape[-2:]
    grid = torch.swapaxes(npts[:2].reshape([2, *img.shape[-2:]]).transpose(1, 2), 0, -1) / torch.Tensor([x, y]).to(img.device) * 2 - 1

    # interpolate new image
    nimg = torch.nn.functional.grid_sample(img, grid[None, :], mode=mode, padding_mode='zeros', align_corners=False)

    return nimg
