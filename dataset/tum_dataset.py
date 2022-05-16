import cv2
import os
import numpy as np
import glob
import torch
import warnings
from torch.utils.data import Dataset
from dataset.transforms import ResizeRGBD
from typing import Tuple


class TUMDataset(Dataset):
    def __init__(self, input_folder:str, img_size: Tuple):
        super().__init__()
        self.imgs = sorted(glob.glob(os.path.join(input_folder, 'rgb', '*.png')))
        self.depth = sorted(glob.glob(os.path.join(input_folder, 'depth', '*.png')))
        assert len(self.imgs) == len(self.depth)
        assert len(self.imgs) > 0

        self.transform = ResizeRGBD(img_size, disparity=False)
        img = cv2.cvtColor(cv2.imread(self.imgs[0]), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        self.scale = max(img_size[0] / w, img_size[1] / h)

    def __getitem__(self, item):
        img = cv2.cvtColor(cv2.imread(self.imgs[item]), cv2.COLOR_BGR2RGB)
        img_number = os.path.basename(self.imgs[item]).split('.png')[0]
        # find depth map according to file-look up
        depth = cv2.imread(self.depth[item], cv2.IMREAD_ANYDEPTH)
        depth = depth.astype(np.float32) / 5.0

        if self.transform is not None:
            img, depth, mask = self.transform(img, depth)

        return img, depth, mask, img_number

    def __len__(self):
        return len(self.imgs)

    def get_intrinsics(self):
        intrinsics = np.array([[525.0, 0, 319.5], [0, 525.0, 239.5], [0.0, 0.0, 1.0]])
        intrinsics[:2,:3] *= self.scale
        return intrinsics




