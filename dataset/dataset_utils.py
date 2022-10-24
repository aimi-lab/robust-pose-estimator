import os
import glob
from dataset.scared_dataset import ScaredDataset
from dataset.tum_dataset import TUMDataset
from dataset.video_dataset import StereoVideoDataset
from dataset.semantic_dataset import StereoDataset, RGBDDataset
from dataset.rectification import StereoRectifier
from typing import Tuple
from torch.utils.data import Sampler


def get_data(input_path: str, img_size: Tuple, sample_video: int=1, rect_mode: str='conventional', force_stereo: bool=False):

    # check the format of the calibration file
    img_size = tuple(img_size)
    calib_file = None
    if os.path.isfile(os.path.join(input_path, 'camcal.json')):
        calib_file = os.path.join(input_path, 'camcal.json')
    elif os.path.isfile(os.path.join(input_path, 'camera_calibration.json')):
        calib_file = os.path.join(input_path, 'camera_calibration.json')
    elif os.path.isfile(os.path.join(input_path, 'StereoCalibration.ini')):
        calib_file = os.path.join(input_path, 'StereoCalibration.ini')
    elif os.path.isfile(os.path.join(input_path, 'endoscope_calibration.yaml')):
        calib_file = os.path.join(input_path, 'endoscope_calibration.yaml')
    else:
        # no calibration file found, then it could be a TUM Dataset
        try:
            dataset = TUMDataset(input_path, img_size)
            print(" TUM Dataset")
            calib = {'intrinsics': {'left': dataset.get_intrinsics()}, 'bf': 1.0}
        except AssertionError:
            raise RuntimeError('no calibration file found')

    if calib_file is not None:
        rect = StereoRectifier(calib_file, img_size_new=img_size, mode=rect_mode)
        calib = rect.get_rectified_calib()
        try:
            assert not force_stereo
            dataset = RGBDDataset(input_path, img_size=calib['img_size'])
            print(" RGBD Dataset with precomputed depth")
        except AssertionError:
            try:
                dataset = StereoDataset(input_path, img_size=calib['img_size'])
                print(" Stereo Dataset")
            except AssertionError:
                try:
                    dataset = ScaredDataset(input_path, img_size=calib['img_size'])
                    print(" SCARED Dataset")
                except AssertionError:
                    video_file = glob.glob(os.path.join(input_path, '*.mp4'))[0]
                    pose_file = os.path.join(input_path, 'groundtruth.txt')
                    dataset = StereoVideoDataset(video_file, pose_file, img_size=calib['img_size'], sample=sample_video, rectify=rect)
                    print(" Stereo Video Dataset")
    return dataset, calib


class SequentialSubSampler(Sampler):
    """
    Samples elements sequentially, always in the same order with a defined subsampling step.

        :param data_source: dataset to sample from
        :param step: subsample step
    """

    def __init__(self, data_source, start: int= 0, stop: int=-1, step: int=1):
        self.data_source = data_source
        self.start = start
        self.stop = stop
        self.step = step

    def __iter__(self):
        if self.stop > 0:
            l = min(self.stop, len(self.data_source))
        return iter(range(self.start, l, self.step))

    def __len__(self):
        return int(len(self.data_source)/self.step)
