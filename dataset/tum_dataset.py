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
        # generate look-up table for synchronization
        self.depth_lookup = [os.path.basename(l).split('.png')[0] for l in self.depth]
        self.depth_lookup = np.asarray([int(l.split('.')[0] + l.split('.')[1]) for l in self.depth_lookup])
        #assert len(self.imgs) == len(self.depth)
        assert len(self.imgs) > 0

        self.transform = ResizeRGBD(img_size)
        img = cv2.cvtColor(cv2.imread(self.imgs[0]), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        self.scale = max(img_size[0] / w, img_size[1] / h)

    def _find_closest_timestamp(self, item):
        query = os.path.basename(self.imgs[item]).split('.png')[0]
        query = int(query.split('.')[0] + query.split('.')[1])
        return np.argmin((self.depth_lookup - query)**2)

    def __getitem__(self, item):
        img = cv2.cvtColor(cv2.imread(self.imgs[item]), cv2.COLOR_BGR2RGB)
        img_number = os.path.basename(self.imgs[item]).split('.png')[0]

        # find depth map that has closed time index as depth and rgb are not synchronized
        depth = cv2.imread(self.depth[self._find_closest_timestamp(item)], cv2.IMREAD_ANYDEPTH)
        depth = depth.astype(np.float32) / 5.0

        img = torch.tensor(img).permute(2, 0, 1).float() / 255.0
        depth = torch.tensor(depth).unsqueeze(0)
        mask = torch.ones_like(depth, dtype=torch.bool)
        depth_noise = torch.exp(-depth)
        semantics = torch.ones_like(depth)
        data = self.transform(img, depth, depth_noise, mask, semantics)

        return (*data, img_number)

    def __len__(self):
        return len(self.imgs)

    def get_intrinsics(self):
        intrinsics = np.array([[525.0, 0, 319.5], [0, 525.0, 239.5], [0.0, 0.0, 1.0]])
        intrinsics[:2,:3] *= self.scale
        return intrinsics




