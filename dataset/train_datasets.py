import numpy as np
import torch
import torch.utils.data as data

import os
from glob import glob
from typing import Tuple
import cv2
from core.utils.pfm_handler import load_pfm
from core.utils.trajectory import read_freiburg
from core.geometry.lie_3d_pseudo import pseudo_lie_SE3_to_se3
from torchvision.transforms import Resize, InterpolationMode
from torch.utils.data import Dataset
from dataset.rectification import StereoRectifier
from dataset.semantic_dataset import RGBDecoder
import warnings


def get_data(config, img_size: Tuple, depth_cutoff: float):
    # check the format of the calibration file
    torch.manual_seed(1234)
    np.random.seed(1234)
    img_size = tuple(img_size)
    if config['type'] == 'TartanAir':
        dataset = TartainAir(root=config['basepath'], seqs=config['sequences'], step=config['step'], img_size=img_size, depth_cutoff=depth_cutoff)
    elif config['type'] == 'Intuitive':
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

    else:
        raise NotImplementedError

    return dataset


class PoseDataset(Dataset):
    def __init__(self, root, baseline, intrinsics, depth_cutoff=300.0,conf_thr=0.0, step=(1,10), img_size=(512, 640), samples=-1):
        super(PoseDataset, self).__init__()
        images_l = sorted(glob(os.path.join(root, 'video_frames', '*l.png')))
        images_r = sorted(glob(os.path.join(root, 'video_frames', '*r.png')))
        semantics = sorted(glob(os.path.join(root, 'semantic_predictions', '*l.png')))
        gt_file = os.path.join(root, 'groundtruth.txt')
        poses = read_freiburg(gt_file)
        assert len(images_l) == len(images_r)
        assert len(images_l) > 0 , f'no images in {root}'
        assert len(semantics) == len(images_l)
        sample_list = self._random_sample(step, samples, len(images_l))

        self.conf_thr = conf_thr
        self.depth_cutoff = depth_cutoff
        self.image_list = []
        self.image_list_r = []
        self.mask_list = []
        self.rel_pose_list = []
        self.rgb_decoder = RGBDecoder()
        if isinstance(step, int):
            step = (step, step)
        for i in sample_list:
            s = np.random.randint(*step) if step[0] < step[1] else step[0]  # select a random step in given range
            self.image_list.append([images_l[i], images_l[i+s]])
            self.rel_pose_list.append(np.linalg.inv(poses[i+s].astype(np.float64)) @ poses[i].astype(np.float64))
            self.image_list_r.append([images_r[i], images_r[i+s]])
            self.mask_list.append([semantics[i], semantics[i+s]])
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
        pose_se3 = pseudo_lie_SE3_to_se3(pose)

        # generate mask
        mask1 = self._read_mask(self.mask_list[index][0])
        mask2 = self._read_mask(self.mask_list[index][1])

        return img1, img2, img1_r, img2_r, mask1, mask2, pose_se3.float(), self.intrinsics[index], float(self.baseline[index]/self.depth_cutoff)

    def _read_img(self, path):
        img = torch.from_numpy(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()
        return self.resize(img)

    def _read_mask(self, path):
        mask = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        mask = self.rgb_decoder.getToolMask(mask)
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
            warnings.warn(f"{os.path.join(root, 'annotions.csv')} does not exist.",RuntimeWarning)
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

class DummyDataset(PoseDataset):
    def __getitem__(self, index):
        return super().__getitem__(0)

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
        pose_se3 = pseudo_lie_SE3_to_se3(pose)
        depth_conf1 = torch.exp(-.5 * depth1 ** 2*10 )
        depth_conf2 = torch.exp(-.5 * depth2 ** 2*10 )
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
        depth = cv2.bilateralFilter(depth.astype(np.float32), d=-1, sigmaColor=0.01, sigmaSpace=10)
        depth = torch.from_numpy(depth / 5.0).unsqueeze(0)
        img = torch.from_numpy(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()
        return self.resize(img), self.resize(depth)

    def __len__(self):
        return len(self.image_list)


class TartainAirSubset(Dataset):
    def __init__(self, root, depth_cutoff,step, img_size):
        super(TartainAirSubset, self).__init__()
        images = sorted(glob(os.path.join(root, 'image_left', '*.png')))
        depth = sorted(glob(os.path.join(root, 'depth_left', '*.png')))
        gt_file = os.path.join(root, 'groundtruth.txt')
        poses, pose_lookup = read_freiburg(gt_file, ret_stamps=True)
        assert len(images) == len(depth)
        self.depth_cutoff = depth_cutoff
        self.image_list = []
        self.depth_list = []
        self.rel_pose_list = []

        if isinstance(step, int):
            step = (step, step)
        for i in range(len(images)-step[1]):
            s = np.random.randint(*step) if step[0] < step[1] else step[0]  # select a random step in given range
            self.image_list.append([images[i], images[i+s]])
            self.depth_list.append([depth[i], depth[i+s]])
            self.rel_pose_list.append(np.linalg.inv(poses[i+s].astype(np.float64)) @ poses[i].astype(np.float64))
        self.resize = Resize(img_size)

    def __getitem__(self, index):
        img1, depth1 = self._read_img(index,0)
        img2, depth2 = self._read_img(index,1)

        # normalize depth
        depth1 /= self.depth_cutoff
        depth2 /= self.depth_cutoff

        pose = torch.from_numpy(self.rel_pose_list[index]).clone()
        pose[:3, 3] /= self.depth_cutoff  # normalize translation
        pose_se3 = pseudo_lie_SE3_to_se3(pose)
        depth_conf1 = torch.exp(-.5 * depth1 ** 2*10 )
        depth_conf2 = torch.exp(-.5 * depth2 ** 2*10 )
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
        depth = np.load(self.depth_list[item][i])
        depth = torch.from_numpy(depth).unsqueeze(0)
        img = torch.from_numpy(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()
        return self.resize(img), self.resize(depth)

    def __len__(self):
        return len(self.image_list)


class TartainAir(TartainAirSubset):
    def __init__(self, root, seqs, depth_cutoff, step, img_size):
        datasets = [sorted(glob(os.path.join(root, s, 'P*'))) for s in seqs]
        datasets = [item for sublist in datasets for item in sublist]
        image_list1 = []
        depth_list1 = []
        rel_pose_list1 = []
        for d in datasets:
            super().__init__(d, depth_cutoff, step, img_size)
            image_list1 += self.image_list
            depth_list1 += self.depth_list
            rel_pose_list1 += self.rel_pose_list
        self.image_list = image_list1
        self.depth_list = depth_list1
        self.rel_pose_list = rel_pose_list1
