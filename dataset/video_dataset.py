import cv2
import os
import glob
from torch.utils.data import IterableDataset
from dataset.transforms import ResizeStereo
from typing import Tuple
from dataset.rectification import StereoRectifier


class StereoVideoDataset(IterableDataset):
    def __init__(self, input_folder:str, img_size:Tuple=None, rectify: bool=True):
        super().__init__()
        self.video_file = glob.glob(os.path.join(input_folder, '*.mp4'))
        assert len(self.video_file) == 1
        self.video_file = self.video_file[0]
        if rectify:
            calib_file = os.path.join(input_folder, 'camcal.json') if \
                os.path.isfile(os.path.join(input_folder, 'camcal.json')) \
                else os.path.join(input_folder, 'StereoCalibration.ini')
            assert os.path.isfile(calib_file)
            self.rectify = StereoRectifier(calib_file, img_size)
        else:
            self.rectify = None
        self.transform = ResizeStereo(img_size)
        vid_grabber = cv2.VideoCapture(self.video_file)
        self.length = int(vid_grabber.get(cv2.CAP_PROP_FRAME_COUNT))

    def __iter__(self):
        img_left, img_right = self._parse_video()
        return img_left, img_right

    def _parse_video(self):
        vid_grabber = cv2.VideoCapture(self.video_file)
        while True:
            ret, img = vid_grabber.read()
            if not ret:
                break
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_left, img_right = self.split_stereo_img(img)
            if self.img_transform is not None:
                img_left, img_right = self.img_transform(img_left, img_right)
            if self.rectify is not None:
                img_left, img_right = self.rectify(img_left, img_right)
            yield img_left, img_right
        vid_grabber.release()

    def __len__(self):
        return self.length

    def _split_stereo_img(self, img, left_is_top):
        h, w = img.shape[:2]
        img_left = img[:h // 2]  # upper half
        img_right = img[h // 2:]  # lower half
        return img_left, img_right



