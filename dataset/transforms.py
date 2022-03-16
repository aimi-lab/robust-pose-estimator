import cv2


class RGBDTransform(object):
    def __call__(self, left, depth, mask=None):
        return left, depth, mask


class ResizeRGBD(RGBDTransform):
    def __init__(self, size):
        self.size = int(size[0]), int(size[1])

    def __call__(self, img, disparity=None, mask=None, label=None):
        # resize with cropping to conserve aspect ratio
        h, w = img.shape[:2]
        scale = max(self.size[0] / w, self.size[1] / h)
        crop = w * scale - self.size[0], h * scale - self.size[1]
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
        img = img[int(crop[1] / 2):int(h * scale - crop[1] / 2), int(crop[0] / 2):int(w * scale - crop[0] / 2)]
        if disparity is not None:
            disparity = cv2.resize(disparity, (int(w * scale), int(h * scale)), cv2.INTER_NEAREST)
            disparity = disparity[int(crop[1] / 2):int(h*scale-crop[1] / 2), int(crop[0] / 2):int(w*scale-crop[0] / 2)]
            # we need to scale the disparity when resizing
            disparity *= scale
        if mask is not None:
            mask = cv2.resize(mask, (int(w * scale), int(h * scale)), cv2.INTER_NEAREST)
            mask = mask[int(crop[1] / 2):int(h * scale - crop[1] / 2), int(crop[0] / 2):int(w * scale - crop[0] / 2)]
        if label is not None:
            label = cv2.resize(label, (int(w * scale), int(h * scale)), cv2.INTER_NEAREST)
            label = label[int(crop[1] / 2):int(h * scale - crop[1] / 2), int(crop[0] / 2):int(w * scale - crop[0] / 2)]
        return img, disparity, mask, label


class StereoTransform(object):
    def __call__(self, left, depth, mask=None):
        return left, depth, mask


class ResizeStereo(StereoTransform):
    def __init__(self, size):
        self.size = int(size[0]), int(size[1])

    def __call__(self, left_img, right_img, label=None):
        # resize with cropping to conserve aspect ratio
        h, w = left_img.shape[:2]
        scale = max(self.size[0] / w, self.size[1] / h)
        crop = w * scale - self.size[0], h * scale - self.size[1]
        left_img = cv2.resize(left_img, (int(w * scale), int(h * scale)))
        left_img = left_img[int(crop[1] / 2):int(h * scale - crop[1] / 2), int(crop[0] / 2):int(w * scale - crop[0] / 2)]
        right_img = cv2.resize(right_img, (int(w * scale), int(h * scale)))
        right_img = right_img[int(crop[1] / 2):int(h * scale - crop[1] / 2),
                   int(crop[0] / 2):int(w * scale - crop[0] / 2)]
        if label is not None:
            label = cv2.resize(label, (int(w * scale), int(h * scale)), cv2.INTER_NEAREST)
            label = label[int(crop[1] / 2):int(h * scale - crop[1] / 2), int(crop[0] / 2):int(w * scale - crop[0] / 2)]
            return left_img, right_img, label
        else:
            return left_img, right_img
