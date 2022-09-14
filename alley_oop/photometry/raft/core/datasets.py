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
        # no calibration file found as we use TUM dataset
        print('TUM Dataset detected')
        dataset = TUMDataset(root=input_path, step=step, img_size=img_size)
        intrinsics = torch.tensor([[525.0, 0, 319.5], [0, 525.0, 239.5], [0.0, 0.0, 1.0]]).float()
        return dataset, intrinsics
    rect = StereoRectifier(calib_file, img_size_new=(img_size[1], img_size[0]), mode='conventional')
    calib = rect.get_rectified_calib()
    dataset = MultiSeqPoseDataset(root=input_path, seqs=sequences, baseline=calib['bf_orig'], conf_thr=0.0, step=step, img_size=img_size)

    intrinsics = torch.tensor(calib['intrinsics']['left']).float()
    return dataset, intrinsics


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

    def __getitem__(self, index):
        img1 = self._read_img(self.image_list[index][0])
        img2 = self._read_img(self.image_list[index][1])
        disp1 = self._read_disp(self.disp_list[index][0])
        disp2 = self._read_disp(self.disp_list[index][1])

        pose = torch.from_numpy(self.rel_pose_list[index]).clone()
        pose[:3,3] /= self.depth_cutoff  # normalize translation
        pose_se3 = lie_SE3_to_se3(pose)
        depth1 = self.baseline / disp1 / self.depth_cutoff  # normalize depth
        depth2 = self.baseline / disp2 / self.depth_cutoff  # normalize depth

        # generate mask
        # depth confidence threshold
        depth_conf1 = torch.special.erf(5e-3*250/torch.sqrt(self._read_disp(self.depth_noise_list[index][0])))
        depth_conf2 = torch.special.erf(5e-3*250/torch.sqrt(self._read_disp(self.depth_noise_list[index][1])))
        valid = depth_conf1 > self.conf_thr
        valid &= depth_conf2 > self.conf_thr
        # depth threshold
        valid &= depth1 < 1.0
        valid &= depth2 < 1.0
        valid &= depth1 > 1e-3
        valid &= depth2 > 1e-3
        # ToDo add tool mask!
        return img1, img2, depth1, depth2, depth_conf1, depth_conf2, valid, pose_se3.float()

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


class TUMDataset(Dataset):
    def __init__(self, root, depth_cutoff=8000.0,step=(1,10), img_size=(512, 640)):
        super(TUMDataset, self).__init__()
        images = sorted(glob(os.path.join(root, 'rgb', '*.png')))
        depth = sorted(glob(os.path.join(root, 'depth', '*.png')))
        gt_file = os.path.join(root, 'groundtruth.txt')
        poses, pose_lookup = read_freiburg(gt_file, ret_stamps=True)
        assert len(images) == len(depth)
        self.depth_cutoff = depth_cutoff
        self.image_list = []
        self.depth_list = []
        self.rel_pose_list = []
        # generate look-up table for synchronization
        depth_lookup = [os.path.basename(l).split('.png')[0] for l in depth]
        depth_lookup = np.asarray([int(l.split('.')[0] + l.split('.')[1]) for l in depth_lookup])
        if isinstance(step, int):
            step = (step, step)
        for i in range(len(images)-step[1]):
            s = np.random.randint(*step) if step[0] < step[1] else step[0]  # select a random step in given range
            self.image_list.append([images[i], images[i+s]])
            self.depth_list.append([depth[self._find_closest_timestamp(images[i], depth_lookup)],
                                    depth[self._find_closest_timestamp(images[i+s], depth_lookup)]])
            pose_cur = poses[self._find_closest_timestamp(images[i], pose_lookup)].astype(np.float64)
            pose_next = poses[self._find_closest_timestamp(images[i+s], pose_lookup)].astype(np.float64)
            self.rel_pose_list.append(np.linalg.inv(pose_next) @ pose_cur)
        self.resize = Resize(img_size)

    def _find_closest_timestamp(self, path, lookup):
        query = os.path.basename(path).split('.png')[0]
        query = int(query.split('.')[0] + query.split('.')[1])
        return np.argmin((lookup - query) ** 2)

    def __getitem__(self, index):
        img1, depth1 = self._read_img(index,0)
        img2, depth2 = self._read_img(index,1)

        # normalize depth
        depth1 /= self.depth_cutoff
        depth2 /= self.depth_cutoff

        pose = torch.from_numpy(self.rel_pose_list[index]).clone()
        pose[:3, 3] /= self.depth_cutoff  # normalize translation
        pose_se3 = lie_SE3_to_se3(pose)
        depth_conf1 = torch.ones_like(depth1)#torch.exp(-.5 * depth1 ** 2*10 )
        depth_conf2 = torch.ones_like(depth2)#torch.exp(-.5 * depth2 ** 2*10 )
        # generate mask
        # depth threshold
        valid = depth1 < 1.0
        valid &= depth2 < 1.0
        valid &= depth1 > 1e-3
        valid &= depth2 > 1e-3
        return img1, img2, depth1, depth2, depth_conf1, depth_conf2, valid, pose_se3.float()

    def _read_img(self, item, i):
        path = self.image_list[item][i]
        # find depth map that has closed time index as depth and rgb are not synchronized
        depth = cv2.imread(self.depth_list[item][i], cv2.IMREAD_ANYDEPTH)
        depth = torch.from_numpy(depth.astype(np.float32) / 5.0).unsqueeze(0)
        img = torch.from_numpy(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()
        return self.resize(img), self.resize(depth)

    def __len__(self):
        return len(self.image_list)
