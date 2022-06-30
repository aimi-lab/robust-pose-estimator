import cv2
import os
import json
from torch.utils.data import IterableDataset
from dataset.transforms import ResizeStereo, Compose
from typing import Tuple
from dataset.rectification import StereoRectifier
import numpy as np
import torch


class StereoVideoDataset(IterableDataset):
    def __init__(self, video_file:str, calib_file: str, pose_file: str=None, img_size:Tuple=None, rectify: bool=True, sample: int=1):
        super().__init__()
        self.video_file = video_file
        assert os.path.isfile(self.video_file)
        if rectify:
            assert os.path.isfile(calib_file)
            self.rectify = StereoRectifier(calib_file, img_size)
        else:
            self.rectify = None
        self.transform = ResizeStereo(img_size)
        vid_grabber = cv2.VideoCapture(self.video_file)
        self.length = int(vid_grabber.get(cv2.CAP_PROP_FRAME_COUNT)/sample)
        self.sample = sample

        self.poses = None
        if pose_file is not None:
            if os.path.isfile(pose_file):
                with open(str(pose_file), 'r') as f:
                    self.poses = np.array(json.load(f))

    def __iter__(self):
        return self._parse_video()

    def _parse_video(self):
        vid_grabber = cv2.VideoCapture(self.video_file)
        counter = 0
        while True:
            while True:
                ret, img = vid_grabber.read()
                counter += 1
                if not ret:
                    break
                if (counter-1)%self.sample == 0:
                    break
            if not ret:
                break
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_left, img_right = self._split_stereo_img(img)
            pose = self.poses[counter] if self.poses is not None else np.eye(4)
            if self.transform is not None:
                img_left, img_right = self.transform(img_left, img_right)
            if self.rectify is not None:
                img_left, img_right = self.rectify(img_left, img_right)
            img_left = torch.tensor(img_left).permute(2,0,1).float() / 255.0
            img_right = torch.tensor(img_right).permute(2,0,1).float() / 255.0
            yield img_left, img_right, pose, counter
        vid_grabber.release()

    def __len__(self):
        return self.length

    def _split_stereo_img(self, img):
        h, w = img.shape[:2]
        img_left = img[:h // 2]  # upper half
        img_right = img[h // 2:]  # lower half
        return img_left, img_right
