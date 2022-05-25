import cv2
import os
import numpy as np
import glob
import torch
import warnings
from torch.utils.data import Dataset
from dataset.transforms import ResizeRGBD
from typing import Tuple


class ScaredDataset(Dataset):
    def __init__(self, input_folder:str, baseline_orig:float, img_size: Tuple):
        super().__init__()
        self.imgs = sorted(glob.glob(os.path.join(input_folder, 'data','video_frames', '*l.png')))
        self.disparity = sorted(glob.glob(os.path.join(input_folder, 'data','disparity_frames', '*.pfm')))
        assert len(self.imgs) == len(self.disparity)
        assert len(self.imgs) > 0

        self.transform = ResizeRGBD(img_size)
        self.baseline = baseline_orig

    def __getitem__(self, item):
        img_l = cv2.cvtColor(cv2.imread(self.imgs[item]), cv2.COLOR_BGR2RGB)
        img_number = os.path.basename(self.imgs[item]).split('l.png')[0]
        img_r = cv2.cvtColor(cv2.imread(self.imgs[item].replace('l.png', 'r.png')), cv2.COLOR_BGR2RGB)
        # find depth map according to file-look up
        disparity = cv2.imread(self.disparity[item], cv2.IMREAD_UNCHANGED)
        depth = self.baseline / disparity  # get depth from disparity (fc * baseline) / disparity
        data = self.transform(img_l, depth, None, img_r, disparity)
        return (*data, img_number)

    def __len__(self):
        return len(self.imgs)



