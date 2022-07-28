from typing import Tuple, List
import json
from alley_oop.utils.trajectory import save_freiburg
import cv2
import os
import numpy as np
import glob
import torch
from torch.utils.data import Dataset
from dataset.transforms import ResizeStereo
from typing import Tuple


class ScaredDataset(Dataset):
    def __init__(self, input_folder:str, img_size:Tuple):
        super().__init__()
        self.imgs = sorted(glob.glob(os.path.join(input_folder, 'data', 'video_frames*', '*l.png')))
        assert len(self.imgs) > 0
        self.transform = ResizeStereo(img_size)

    def __getitem__(self, item):
        img_l = cv2.cvtColor(cv2.imread(self.imgs[item]), cv2.COLOR_BGR2RGB)
        img_r = cv2.cvtColor(cv2.imread(self.imgs[item].replace('l.png', 'r.png')), cv2.COLOR_BGR2RGB)
        img_number = os.path.basename(self.imgs[item]).split('l.png')[0]
        # to torch tensor
        img_l = torch.tensor(img_l).permute(2,0,1).float()/255.0
        img_r = torch.tensor(img_r).permute(2, 0, 1).float() / 255.0
        mask = torch.ones((1, *img_l.shape[-2:]), dtype=torch.bool)
        semantics = torch.zeros((1, *img_l.shape[-2:]), dtype=torch.long)
        data = self.transform(img_l, img_r, mask, semantics)
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
