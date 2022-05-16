import cv2


class RGBDTransform(object):
    def __call__(self, left, depth, mask=None):
        return left, depth, mask


class ResizeRGBD(RGBDTransform):
    def __init__(self, size, disparity=True):
        self.size = int(size[0]), int(size[1])
        self.disparity = disparity

    def __call__(self, img, disp_or_depth=None, mask=None):
        # resize with cropping to conserve aspect ratio
        h, w = img.shape[:2]
        scale = max(self.size[0] / w, self.size[1] / h)
        crop = w * scale - self.size[0], h * scale - self.size[1]
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
        img = img[int(crop[1] / 2):int(h * scale - crop[1] / 2), int(crop[0] / 2):int(w * scale - crop[0] / 2)]
        if disp_or_depth is not None:
            disp_or_depth = cv2.resize(disp_or_depth, (int(w * scale), int(h * scale)), cv2.INTER_NEAREST)
            disp_or_depth = disp_or_depth[int(crop[1] / 2):int(h*scale-crop[1] / 2), int(crop[0] / 2):int(w*scale-crop[0] / 2)]
            # we need to scale the disparity when resizing
            if self.disparity:
                disp_or_depth *= scale
        if mask is not None:
            mask = cv2.resize(mask, (int(w * scale), int(h * scale)), cv2.INTER_NEAREST)
            mask = mask[int(crop[1] / 2):int(h * scale - crop[1] / 2), int(crop[0] / 2):int(w * scale - crop[0] / 2)]
        return img, disp_or_depth, mask


class Compose(object):
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, img, disp_or_depth=None, mask=None):
        args = (img, disp_or_depth, mask)
        for tr in self.transforms:
            args = tr(*args)
        return args



class StereoTransform(object):
    def __call__(self, left, depth, mask=None):
        return left, depth, mask


class ResizeStereo(StereoTransform):
    def __init__(self, size):
        self.size = int(size[0]), int(size[1])

    def __call__(self, left_img, right_img, mask=None):
        # resize with cropping to conserve aspect ratio
        h, w = left_img.shape[:2]
        scale = max(self.size[0] / w, self.size[1] / h)
        crop = w * scale - self.size[0], h * scale - self.size[1]
        left_img = cv2.resize(left_img, (int(w * scale), int(h * scale)))
        left_img = left_img[int(crop[1] / 2):int(h * scale - crop[1] / 2), int(crop[0] / 2):int(w * scale - crop[0] / 2)]
        right_img = cv2.resize(right_img, (int(w * scale), int(h * scale)))
        right_img = right_img[int(crop[1] / 2):int(h * scale - crop[1] / 2),
                   int(crop[0] / 2):int(w * scale - crop[0] / 2)]
        if mask is not None:
            mask = cv2.resize(mask, (int(w * scale), int(h * scale)), cv2.INTER_NEAREST)
            mask = mask[int(crop[1] / 2):int(h * scale - crop[1] / 2), int(crop[0] / 2):int(w * scale - crop[0] / 2)]
        return left_img, right_img, mask
