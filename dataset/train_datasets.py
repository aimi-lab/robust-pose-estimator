import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import Dataset
from torchvision.transforms import Resize, InterpolationMode
import os
from glob import glob
from typing import Tuple
import cv2

from core.utils.trajectory import read_freiburg
from dataset.rectification import StereoRectifier


def get_data(config: dict, img_size: Tuple, depth_cutoff: float):
    # check the format of the calibration file
    torch.manual_seed(1234)
    np.random.seed(1234)
    img_size = tuple(img_size)
    baseline = []
    intrinsics = []
    for i in range(len(config['sequences'])):
        calib_path = os.path.join(config['basepath'], config['sequences'][i], 'keyframe_1')
        if not os.path.exists(calib_path):
            calib_path = os.path.join(config['basepath'], config['sequences'][i])
        if os.path.isfile(os.path.join(calib_path, 'camcal.json')):
            calib_file = os.path.join(calib_path, 'camcal.json')
        elif os.path.isfile(os.path.join(calib_path, 'camera_calibration.json')):
            calib_file = os.path.join(calib_path, 'camera_calibration.json')
        elif os.path.isfile(os.path.join(calib_path, 'StereoCalibration.ini')):
            calib_file = os.path.join(calib_path, 'StereoCalibration.ini')
        elif os.path.isfile(os.path.join(calib_path, 'endoscope_calibration.yaml')):
            calib_file = os.path.join(calib_path, 'endoscope_calibration.yaml')
        else:
            raise ValueError(f"no calibration file found in {calib_path}")
        rect = StereoRectifier(calib_file, img_size_new=(img_size[1], img_size[0]), mode='conventional')
        calib = rect.get_rectified_calib()
        baseline.append(calib['bf'].astype(np.float32))
        intrinsics.append(calib['intrinsics']['left'].astype(np.float32))

    # glob data
    # for nested scared dataset
    ds = [sorted(glob(os.path.join(config['basepath'], s, 'keyframe_*'))) for s in config['sequences']]
    # others
    if len(ds[0]) == 0:
        ds = [[os.path.join(config['basepath'], s)] for s in config['sequences']]

    # generate multi-sequence dataset
    subsets = []
    for i, s in enumerate(config['sequences']):
        for d in ds[i]:
            if os.path.isfile(os.path.join(d, 'groundtruth.txt')):
                try:
                    subsets.append(PoseDataset(d, baseline[i], intrinsics[i], depth_cutoff, 0.0, config['step'],
                                               img_size, config['samples']))
                except AssertionError: # do not append if something went wrong
                    pass
    dataset = data.ConcatDataset(subsets)
    return dataset


class PoseDataset(Dataset):
    def __init__(self, root, baseline, intrinsics, depth_cutoff=300.0,conf_thr=0.0, step=(1,10), img_size=(512, 640), samples=-1):
        super(PoseDataset, self).__init__()
        images_l = sorted(glob(os.path.join(root, 'video_frames', '*l.png')))
        images_r = sorted(glob(os.path.join(root, 'video_frames', '*r.png')))
        masks = sorted(glob(os.path.join(root, 'masks', '*l.png')))
        gt_file = os.path.join(root, 'groundtruth.txt')
        poses = read_freiburg(gt_file)
        assert len(images_l) == len(images_r)
        assert len(images_l) > 0 , f'no images in {root}'
        n_list = images_l if len(masks) == 0 else masks
        sample_list = self._random_sample(step, samples, len(n_list))

        self.conf_thr = conf_thr
        self.depth_cutoff = depth_cutoff
        self.image_list = []
        self.image_list_r = []
        self.mask_list = []
        self.rel_pose_list = []
        if isinstance(step, int):
            step = (step, step)
        for i in sample_list:
            s = np.random.randint(*step) if step[0] < step[1] else step[0]  # select a random step in given range

            img_number1 = int(os.path.basename(n_list[i]).split('l.png')[0])
            img_number2 = int(os.path.basename(n_list[i+s]).split('l.png')[0])
            self.image_list.append([n_list[i].replace('masks', 'video_frames'),
                                    n_list[i+s].replace('masks', 'video_frames')])
            self.rel_pose_list.append(poses[img_number1-1].inv().mul(poses[img_number2-1]))
            self.image_list_r.append([n_list[i].replace('masks', 'video_frames').replace('l.png', 'r.png'),
                                      n_list[i+s].replace('masks', 'video_frames').replace('l.png', 'r.png')])
            if len(masks) == 0:
                self.mask_list.append([None, None])
            else:
                self.mask_list.append([n_list[i].replace('video_frames', 'masks'),
                                       n_list[i+s].replace('video_frames', 'masks')])
        self.resize = Resize(img_size)
        self.resize_msk = Resize(img_size, interpolation=InterpolationMode.NEAREST)
        self.scale = float(img_size[0])/float(cv2.imread(images_l[0]).shape[1])
        self.intrinsics = len(self.image_list) * [intrinsics]
        self.baseline = len(self.image_list) * [baseline]
        self.img_size = img_size

    def __getitem__(self, index):
        img1 = self._read_img(self.image_list[index][0])
        img2 = self._read_img(self.image_list[index][1])
        img1_r = self._read_img(self.image_list_r[index][0])
        img2_r = self._read_img(self.image_list_r[index][1])

        pose = self.rel_pose_list[index]
        pose = pose.scale(torch.tensor(1/self.depth_cutoff))  # scale translation for normalized depth
        baseline = float(self.baseline[index]/self.depth_cutoff) # scale stereo-baseline for normalized depth

        # generate mask
        mask1 = self._read_mask(self.mask_list[index][0])
        mask2 = self._read_mask(self.mask_list[index][1])

        return img1, img2, img1_r, img2_r, mask1, mask2, pose.vec(), self.intrinsics[index], baseline

    def _read_img(self, path):
        img = torch.from_numpy(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()
        return self.resize(img)

    def _read_mask(self, path):
        if path is not None:
            mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            mask = mask > 0
            mask = torch.from_numpy(mask).unsqueeze(0)
            return self.resize_msk(mask)
        else:
            return torch.ones(1,*self.img_size)

    def __len__(self):
        return len(self.image_list)

    def _random_sample(self, step, samples, total_samples):
        if isinstance(step, int):
            step = (step, step)
            # randomly sample n-frames
        if (samples > 0) & (samples < total_samples):
            sample_list = sorted(np.random.choice(total_samples - step[1], size=(samples,), replace=False))
        else:
            sample_list = np.arange(total_samples - step[1])
        return sample_list
