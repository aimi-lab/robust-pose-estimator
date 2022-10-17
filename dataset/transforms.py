from torchvision.transforms.functional import resize, center_crop
from torchvision.transforms import InterpolationMode


class Compose(object):
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, *args):
        for tr in self.transforms:
            args = tr(*args)
        return args


class StereoTransform(object):
    def __call__(self, left, right, mask, semantics):
        return left, right, mask, semantics


class ResizeStereo(StereoTransform):
    def __init__(self, size):
        self.size = [int(size[1]), int(size[0])]

    def __call__(self, left, right, mask=None, semantics=None):
        # resize with cropping to conserve aspect ratio
        h, w = left.shape[-2:]

        scale = max(self.size[0] / h, self.size[1] / w)
        size = [int(scale*h), int(scale*w)]
        left = self._resize_with_crop(left, size)
        right = self._resize_with_crop(right, size)
        mask = self._resize_with_crop(mask, size, InterpolationMode.NEAREST)
        semantics = self._resize_with_crop(semantics, size, InterpolationMode.NEAREST)
        return left, right, mask, semantics

    def _resize_with_crop(self, img, size, mode=InterpolationMode.BILINEAR):
        if img is not None:
            img = resize(img, size=size, interpolation=mode)
            img = center_crop(img, self.size)
        return img


class RGBDTransform(object):
    def __call__(self, img, depth, mask, semantics):
        return img, depth, mask, semantics


class ResizeRGBD(RGBDTransform):
    def __init__(self, size):
        self.size = [int(size[1]), int(size[0])]

    def __call__(self, img, depth, mask=None, semantics=None):
        # resize with cropping to conserve aspect ratio
        h, w = img.shape[-2:]

        scale = max(self.size[0] / h, self.size[1] / w)
        size = [int(scale * h), int(scale * w)]
        img = self._resize_with_crop(img, size)
        depth = self._resize_with_crop(depth, size)
        mask = self._resize_with_crop(mask, size, InterpolationMode.NEAREST)
        semantics = self._resize_with_crop(semantics, size, InterpolationMode.NEAREST)
        return img, depth, mask, semantics

    def _resize_with_crop(self, img, size, mode=InterpolationMode.BILINEAR):
        if img is not None:
            img = resize(img, size=size, interpolation=mode)
            img = center_crop(img, self.size)
        return img
