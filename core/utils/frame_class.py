import torch
from typing import Union


class Frame:
    """
        Class containing image, depth and normals
    """
    def __init__(self, img: torch.Tensor, rimg: torch.Tensor=None, depth: torch.Tensor=None,
                 mask: torch.Tensor=None, confidence: torch.Tensor=None, flow: torch.Tensor=None):
        """

        :param img: RGB image in range (0, 255) with shape Nx3xHxW
        :param rimg: right RGB image of stereo rig in range (0, 255) with shape Nx3xHxW (optional)
        :param depth: depth map in mm with shape Nx1xHxW (optional)
        :param mask: binary mask to include or exclude points with shape Nx1xHxW (optional)
        :param confidence: depth confidence map (0 to 1) with shape Nx1xHxW (optional)
        """
        assert img.ndim == 4
        self.img = img.contiguous()
        if rimg is None:
            self.rimg = img.contiguous()
        else:
            self.rimg = rimg.contiguous()

        if mask is None:
            mask = torch.ones((1,1,*self.shape), dtype=torch.bool, device=self.device)
        self.mask = mask.bool()

        if depth is None:
            self.depth = torch.ones((1,1,*self.shape), device=self.device)
        else:
            self.depth = depth.contiguous()

        self.confidence = confidence.contiguous() if confidence is not None else torch.ones((1,1,*self.shape), device=self.device)
        self.flow = flow.contiguous() if flow is not None else torch.zeros((1,2,*self.shape), device=self.device)

        assert self.rimg.shape == self.img.shape
        assert self.img.shape[-2:] == self.depth.shape[-2:]
        assert self.img.shape[-2:] == self.mask.shape[-2:]
        assert self.img.shape[-2:] == self.confidence.shape[-2:]
        assert self.img.shape[-2:] == self.flow.shape[-2:]

    def to(self, dev_or_type: Union[torch.device, torch.dtype]):
        self.img = self.img.to(dev_or_type)
        self.rimg = self.rimg.to(dev_or_type)
        self.depth = self.depth.to(dev_or_type)
        self.mask = self.mask.to(dev_or_type)
        self.confidence = self.confidence.to(dev_or_type)
        return self

    @property
    def shape(self):
        return self.img.shape[-2:]

    @property
    def device(self):
        return self.img.device

    def plot(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2,3)
        img,rimg, depth, mask, confidence = self.to_numpy()
        ax[0, 0].imshow(img)
        ax[0, 0].set_title('img left')
        ax[0, 1].imshow(rimg)
        ax[0, 1].set_title('img right')
        ax[0, 2].imshow(depth)
        ax[0, 2].set_title('depth')
        ax[1, 0].imshow(mask, vmin=0, vmax=1, interpolation=None)
        ax[1, 0].set_title('mask')
        ax[1, 1].imshow(confidence, vmin=0, vmax=1, interpolation=None)
        ax[1, 1].set_title('confidence')
        for a in ax.flatten():
            a.axis('off')
        plt.show()

    def to_numpy(self):
        img = self.img.detach().cpu().permute(0,2,3,1).squeeze().numpy()/255.0
        rimg = self.rimg.detach().cpu().permute(0, 2, 3, 1).squeeze().numpy()/255.0
        depth = self.depth.detach().cpu().squeeze().numpy()
        mask = self.mask.detach().cpu().squeeze().numpy()
        confidence = self.confidence.detach().cpu().squeeze().numpy()
        return img, rimg, depth, mask, confidence
