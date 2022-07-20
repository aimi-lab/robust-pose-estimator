import cv2
import os
import numpy as np
import glob
import torch
import warnings
from torch.utils.data import Dataset
from dataset.transforms import ResizeRGBD
from typing import Tuple, List
import json
from alley_oop.utils.trajectory import save_freiburg


class ScaredDataset(Dataset):
    def __init__(self, input_folder:str, baseline_orig:float, img_size: Tuple):
        super().__init__()
        imgs = sorted(glob.glob(os.path.join(input_folder, 'data','video_frames', '*l.png')))
        self.disparity = sorted(glob.glob(os.path.join(input_folder, 'data','disparity_frames_psmnet', '*.pfm')))
        disparity_names = [os.path.basename(d).split('d')[0] for d in self.disparity]
        self.imgs = []
        for img in imgs:
            img_name = os.path.basename(img).split('l.png')[0]
            if img_name not in disparity_names:
                print(f'missing disparity {img_name}. skip frame')
            else:
                self.imgs.append(img)
        assert len(self.imgs) == len(self.disparity)
        assert len(self.imgs) > 0

        self.transform = ResizeRGBD(img_size)
        self.baseline = baseline_orig

    def __getitem__(self, item):
        img_l = cv2.cvtColor(cv2.imread(self.imgs[item]), cv2.COLOR_BGR2RGB)
        img_number = os.path.basename(self.imgs[item]).split('l.png')[0]
        img_r = cv2.cvtColor(cv2.imread(self.imgs[item].replace('l.png', 'r.png')), cv2.COLOR_BGR2RGB)
        # find depth map according to file-look up
        disparity = cv2.imread(self.disparity[item], cv2.IMREAD_UNCHANGED) / 2
        warnings.warn("somehow disparity is scaled by factor 2. Why????", UserWarning)
        depth = self.baseline / disparity  # get depth from disparity (fc * baseline) / disparity
        data = self.transform(img_l, depth, np.ones(depth.shape, dtype=np.uint8), img_r, disparity)
        return (*data, img_number)

    def __len__(self):
        return len(self.imgs)


def load_scared_kinematics(fnames:List) -> np.ndarray:

    # if kinematics data collect all files
    pose_list = []
    for i, fname in enumerate(fnames):
        with open(str(fname), 'r') as f: pose_elem = json.load(f)
        pose = np.array(pose_elem['camera-pose'])
        pose_list.append({'camera-pose': pose, 'timestamp': f'{i:06d}'})
    return pose_list


def scared2freiburg(folder:str):
    fnames = sorted(glob.glob(os.path.join(folder, '*.json')))
    pose_list = load_scared_kinematics(fnames)

    save_freiburg(pose_list, folder)
    import shutil
    shutil.move(os.path.join(folder, 'trajectory.freiburg'), os.path.join(folder, '..', '..', 'groundtruth.txt'))
