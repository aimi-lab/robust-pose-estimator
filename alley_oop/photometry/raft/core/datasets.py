import numpy as np
import torch
import torch.utils.data as data

import os
from glob import glob
from typing import Tuple
import cv2
from alley_oop.utils.pfm_handler import load_pfm
from alley_oop.utils.trajectory import read_freiburg
from alley_oop.geometry.lie_3d import lie_SE3_to_se3
from torchvision.transforms import Resize, InterpolationMode
from torch.utils.data import Dataset
from dataset.rectification import StereoRectifier


def get_data(input_path: str, sequences: str, img_size: Tuple, step: int=1):

    # check the format of the calibration file
    img_size = tuple(img_size)
    calib_path = os.path.join(input_path,sequences[0],'keyframe_1')
    if os.path.isfile(os.path.join(calib_path, 'camcal.json')):
        calib_file = os.path.join(calib_path, 'camcal.json')
    elif os.path.isfile(os.path.join(calib_path, 'camera_calibration.json')):
        calib_file = os.path.join(calib_path, 'camera_calibration.json')
    elif os.path.isfile(os.path.join(calib_path, 'StereoCalibration.ini')):
        calib_file = os.path.join(calib_path, 'StereoCalibration.ini')
    elif os.path.isfile(os.path.join(calib_path, 'endoscope_calibration.yaml')):
        calib_file = os.path.join(calib_path, 'endoscope_calibration.yaml')
    else:
        raise RuntimeError('no calibration file found')

    rect = StereoRectifier(calib_file, img_size_new=(img_size[1], img_size[0]), mode='conventional')
    calib = rect.get_rectified_calib()
    dataset = MultiSeqPoseDataset(root=input_path, seqs=sequences, baseline=calib['bf_orig'], conf_thr=0.0, step=step, img_size=img_size)

    intrinsics_lowres = torch.tensor(calib['intrinsics']['left']).float()
    intrinsics_lowres[:2,:3] /= 8
    return dataset, intrinsics_lowres


class PoseDataset(Dataset):
    def __init__(self, root, baseline, depth_cutoff=300.0,conf_thr=0.0, step=(1,10), img_size=(512, 640)):
        super(PoseDataset, self).__init__()
        images = sorted(glob(os.path.join(root, 'video_frames', '*l.png')))
        disparities = sorted(glob(os.path.join(root, 'disparity_frames', '*l.pfm')))
        gt_file = os.path.join(root, 'groundtruth.txt')
        poses = read_freiburg(gt_file)
        depth_noise = sorted(glob(os.path.join(root,'disparity_noise', '*l.pfm')))
        assert len(images) == len(disparities)
        assert len(images) == len(depth_noise)
        self.baseline = baseline
        self.conf_thr = conf_thr
        self.depth_cutoff = depth_cutoff
        self.image_list = []
        self.disp_list = []
        self.rel_pose_list = []
        self.depth_noise_list = []
        if isinstance(step, int):
            step = (step, step)
        for i in range(len(images)-step[1]):
            s = np.random.randint(*step) if step[0] < step[1] else step[0]  # select a random step in given range
            self.image_list.append([images[i], images[i+s]])
            self.disp_list.append([disparities[i], disparities[i+s]])
            self.rel_pose_list.append(np.linalg.inv(poses[i+s].astype(np.float64)) @ poses[i].astype(np.float64))
            self.depth_noise_list.append([depth_noise[i], depth_noise[i+s]])
        self.resize = Resize(img_size)
        self.resize_lowres = Resize((img_size[0]//8, img_size[1]//8))
        self.resize_lowres_msk = Resize((img_size[0] // 8, img_size[1] // 8), interpolation=InterpolationMode.NEAREST)

    def __getitem__(self, index):
        img1 = self._read_img(self.image_list[index][0])
        img2 = self._read_img(self.image_list[index][1])
        disp1 = self._read_disp(self.disp_list[index][0])
        disp2 = self._read_disp(self.disp_list[index][1])

        pose = torch.from_numpy(self.rel_pose_list[index]).clone()
        pose_se3 = lie_SE3_to_se3(pose)
        depth1 = self.baseline / disp1
        depth2 = self.baseline / disp2

        # generate mask
        # depth confidence threshold
        depth_conf1 = torch.special.erf(5e-3*250/torch.sqrt(self._read_disp(self.depth_noise_list[index][0])))
        depth_conf2 = torch.special.erf(5e-3*250/torch.sqrt(self._read_disp(self.depth_noise_list[index][1])))
        valid = depth_conf1 > self.conf_thr
        valid &= depth_conf2 > self.conf_thr
        # depth threshold
        valid &= depth1 < self.depth_cutoff
        valid &= depth2 < self.depth_cutoff
        valid &= depth1 > 1e-3
        valid &= depth2 > 1e-3
        # ToDo add tool mask!
        return img1, img2, depth1, depth2, depth_conf1, depth_conf2, self.resize_lowres_msk(valid), pose_se3.float()

    def _read_img(self, path):
        img = torch.from_numpy(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()
        return self.resize(img)

    def _read_disp(self, path):
        disp = torch.from_numpy(load_pfm(path)[0].copy()).unsqueeze(0).float()
        return self.resize(disp)

    def __len__(self):
        return len(self.image_list)

class DummyDataset(PoseDataset):
    def __getitem__(self, index):
        return super().__getitem__(0)

class MultiSeqPoseDataset(PoseDataset):
    def __init__(self, root, seqs, baseline, depth_cutoff=300.0, conf_thr=0.0, step=1, img_size=(512, 640)):
        datasets = [sorted(glob(os.path.join(root, s, 'keyframe_*'))) for s in seqs]
        datasets = [item for sublist in datasets for item in sublist]
        image_list1 = []
        disp_list1 = []
        rel_pose_list1 = []
        depth_noise_list1 = []
        for d in datasets:
            if os.path.isfile(os.path.join(d, 'groundtruth.txt')):
                super().__init__(d, baseline, depth_cutoff, conf_thr, step, img_size)
                image_list1 += self.image_list
                disp_list1 += self.disp_list
                rel_pose_list1 += self.rel_pose_list
                depth_noise_list1 += self.depth_noise_list
        self.image_list = image_list1
        self.disp_list = disp_list1
        self.rel_pose_list = rel_pose_list1
        self.depth_noise_list = depth_noise_list1
