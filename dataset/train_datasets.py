import numpy as np
import torch

import os
from glob import glob
from typing import Tuple
import cv2
from core.utils.trajectory import read_freiburg
from core.geometry.lie_3d_small_angle import small_angle_lie_SE3_to_se3
from torchvision.transforms import Resize, InterpolationMode
from torch.utils.data import Dataset
from dataset.rectification import StereoRectifier
import warnings


def get_data(config, img_size: Tuple, depth_cutoff: float):
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
    dataset = MultiSeqPoseDataset(root=config['basepath'], seqs=config['sequences'], baseline=baseline, intrinsics=intrinsics,conf_thr=0.0, step=config['step'],
                              img_size=img_size, depth_cutoff=depth_cutoff, samples=config['samples'])

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
        assert len(masks) == len(images_l)
        sample_list = self._random_sample(step, samples, len(images_l))

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
            self.image_list.append([images_l[i], images_l[i+s]])
            self.rel_pose_list.append(np.linalg.inv(poses[i+s].astype(np.float64)) @ poses[i].astype(np.float64))
            self.image_list_r.append([images_r[i], images_r[i+s]])
            self.mask_list.append([masks[i], masks[i+s]])
        self.resize = Resize(img_size)
        self.resize_msk = Resize(img_size, interpolation=InterpolationMode.NEAREST)
        self.scale = float(img_size[0])/float(cv2.imread(images_l[0]).shape[1])
        self.intrinsics = len(self.image_list) * [intrinsics]
        self.baseline = len(self.image_list) * [baseline]

    def __getitem__(self, index):
        img1 = self._read_img(self.image_list[index][0])
        img2 = self._read_img(self.image_list[index][1])
        img1_r = self._read_img(self.image_list_r[index][0])
        img2_r = self._read_img(self.image_list_r[index][1])

        pose = torch.from_numpy(self.rel_pose_list[index]).clone()
        pose[:3,3] /= self.depth_cutoff  # normalize translation
        pose_se3 = small_angle_lie_SE3_to_se3(pose)

        # generate mask
        mask1 = self._read_mask(self.mask_list[index][0])
        mask2 = self._read_mask(self.mask_list[index][1])

        return img1, img2, img1_r, img2_r, mask1, mask2, pose_se3.float(), self.intrinsics[index], float(self.baseline[index]/self.depth_cutoff)

    def _read_img(self, path):
        img = torch.from_numpy(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()
        return self.resize(img)

    def _read_mask(self, path):
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        mask = mask > 0
        mask = torch.from_numpy(mask).unsqueeze(0)
        return self.resize_msk(mask)

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


class StratifiedPoseDataset(PoseDataset):
    """
        Stratified sampling based on tool presence and camera motion (and their combinations)
    """
    def __init__(self, root, baseline, intrinsics, depth_cutoff=300.0,conf_thr=0.0, step=(1,10), img_size=(512, 640), samples=-1):
        #assert os.path.isfile(os.path.join(root, 'annotions.csv'))
        if os.path.isfile(os.path.join(root, 'annotions.csv')):
            with open(os.path.join(root, 'annotions.csv'), 'r') as f:
                annotations = f.readlines()[1:]
            annotations = np.genfromtxt(annotations, delimiter=',', dtype=bool).astype(int)
            annotations[:, 1] *= 2
            annotations = np.sum(annotations, axis=1)  # class 0, 1, 2, 3
            probs = 1/(np.bincount(annotations, minlength=4) +1)
            probs = probs/np.sum(probs)
            self.probs = np.zeros_like(annotations, dtype=float)
            for i, p in enumerate(probs):
                self.probs[annotations == i] = p
        else:
            self.probs = np.ones(len(glob(os.path.join(root, 'video_frames', '*l.png'))))
        super(StratifiedPoseDataset, self).__init__(root, baseline, intrinsics, depth_cutoff,conf_thr, step, img_size, samples)

    def _random_sample(self, step, samples, total_samples):
        if isinstance(step, int):
            step = (step, step)
            # randomly sample n-frames
        if (samples > 0) & (samples < total_samples):
            ids = np.arange(total_samples - step[1])
            probs = self.probs[:len(ids)] / np.sum(self.probs[:len(ids)])
            sample_list = sorted(np.random.choice(ids, size=(samples,), replace=False, p=probs))
        else:
            sample_list = np.arange(total_samples - step[1])
        return sample_list


class MultiSeqPoseDataset(StratifiedPoseDataset):
    def __init__(self, root, seqs, baseline, intrinsics, depth_cutoff=300.0, conf_thr=0.0, step=1, img_size=(512, 640), samples=-1):
        # for nested scared dataset
        datasets = [sorted(glob(os.path.join(root, s, 'keyframe_*'))) for s in seqs]
        # others
        if len(datasets[0]) == 0:
            datasets = [[os.path.join(root, s)] for s in seqs]
        image_list1 = []
        rel_pose_list1 = []
        image_list_r1 = []
        intrinsics_list = []
        baseline_list = []
        mask_list = []
        for i,s in enumerate(seqs):
            b = baseline[i]
            intr = intrinsics[i]
            for d in datasets[i]:
                if os.path.isfile(os.path.join(d, 'groundtruth.txt')):
                    try:
                        super().__init__(d, b, intr, depth_cutoff, conf_thr, step, img_size, samples)
                        image_list1 += self.image_list
                        image_list_r1 += self.image_list_r
                        mask_list += self.mask_list
                        rel_pose_list1 += self.rel_pose_list
                        intrinsics_list += self.intrinsics
                        baseline_list += self.baseline
                    except AssertionError:
                        pass
        self.image_list = image_list1
        self.image_list_r = image_list_r1
        self.mask_list = mask_list
        self.rel_pose_list = rel_pose_list1
        self.baseline = baseline_list
        self.intrinsics = intrinsics_list
