import cv2
import os
import numpy as np
import glob
import torch
import warnings
from torch.utils.data import Dataset
from dataset.transforms import RGBDTransform


class ScaredDataset(Dataset):
    def __init__(self, input_folder:str, baseline:float, transform:RGBDTransform=None, ret_disparity=False):
        super().__init__()
        self.imgs = sorted(glob.glob(os.path.join(input_folder, 'video_frames', '*l.png')))
        self.disparity = sorted(glob.glob(os.path.join(input_folder, 'disparity_frames', '*.pfm')))
        assert len(self.imgs) == len(self.disparity)
        assert len(self.imgs) == len(self.semantics)
        assert len(self.imgs) > 0

        self.transform = transform
        self.baseline = baseline
        self.ret_disparity = ret_disparity

    def __getitem__(self, item):
        img = cv2.cvtColor(cv2.imread(self.imgs[item]), cv2.COLOR_BGR2RGB)
        img_number = int(os.path.basename(self.imgs[item]).split('l.png')[0])
        # find depth map according to file-look up
        disparity = cv2.imread(self.disparity[item], cv2.IMREAD_UNCHANGED)

        if self.transform is not None:
            img, disparity, *_ = self.transform(img, disparity)

        if self.ret_disparity:
            return img, disparity
        else:
            # get depth from disparity (fc * baseline) / disparity
            depth = self.baseline / disparity
            return img, depth

    def __len__(self):
        return len(self.imgs)



