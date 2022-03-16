import cv2
import os
import numpy as np
import glob
import torch
import warnings
import configparser
import json
from torch.utils.data import IterableDataset
from dataset.transforms import ResizeStereo
from stereo_slam.disparity.stereo_rectify import rectify_pair, get_rect_maps
from typing import Tuple


class VideoDataset(IterableDataset):
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


class StereoRectifier(object):
    def __init__(self, calib_file, img_size_new=None):
        if os.path.splitext(calib_file)[1] == '.json':
            cal = self._load_calib_json(calib_file)
        elif os.path.splitext(calib_file)[1] == '.ini':
            cal = self._load_calib_ini(calib_file)
        else:
            raise NotImplementedError

        if img_size_new is not None:
            # scale intrinsics
            scale = img_size_new[0]/cal['img_size'][0]
            assert scale == img_size_new[1]/cal['img_size'][1]
            cal['lkmat'] *= scale
            cal['rkmat'] *= scale
            cal['img_size'] = img_size_new

        self.maps, self.l_intr, self.r_intr = get_rect_maps(
            lcam_mat=cal['lkmat'],
            rcam_mat=cal['rkmat'],
            rmat=cal['R'],
            tvec=cal['T'],
            ldist_coeffs=cal['ld'],
            rdist_coeffs=cal['rd'],
            img_size=cal['img_size']
        )

    def __call__(self, img_left, img_right):
        return rectify_pair(img_left, img_right, self.maps)

    def get_rectified_calib(self):
        calib_rectifed = {}
        calib_rectifed['intrinsics']['left'] = self.l_intr
        calib_rectifed['intrinsics']['left'] = self.r_intr
        calib_rectifed['extrinsics'] = np.eye(4)
        calib_rectifed['extrinsics'][:3,3] = np.array([self.r_intr[0, 3] / self.r_intr[0, 0], 0., 0.]) # Tx*f, see cv2 website
        return calib_rectifed

    @staticmethod
    def _load_calib_json(fname):

        with open(fname, 'rb') as f: json_dict = json.load(f)

        lkmat = np.eye(3)
        lkmat[0, 0] = json_dict['data']['intrinsics'][0]['f'][0]
        lkmat[1, 1] = json_dict['data']['intrinsics'][0]['f'][1]
        lkmat[:2, -1] = json_dict['data']['intrinsics'][0]['c']

        rkmat = np.eye(3)
        rkmat[0, 0] = json_dict['data']['intrinsics'][1]['f'][0]
        rkmat[1, 1] = json_dict['data']['intrinsics'][1]['f'][1]
        rkmat[:2, -1] = json_dict['data']['intrinsics'][1]['c']

        ld = np.array(json_dict['data']['intrinsics'][0]['k'])
        rd = np.array(json_dict['data']['intrinsics'][1]['k'])

        tvec = np.array(json_dict['data']['extrinsics']['T'])
        rmat = cv2.Rodrigues(np.array(json_dict['data']['extrinsics']['om']))[0]

        img_size = (json_dict['data']['width'], json_dict['data']['height'])

        cal = {}
        cal['lkmat'] = lkmat
        cal['rkmat'] = rkmat
        cal['ld'] = ld
        cal['rd'] = rd
        cal['T'] = tvec
        cal['R'] = rmat
        cal['img_size'] = img_size

        return cal

    @staticmethod
    def _load_calib_ini(fname):
        config = configparser.ConfigParser()
        config.read(fname)
        img_size = (config['StereoLeft']['res_x'], config['StereoLeft']['res_y'])

        lkmat = np.eye(3)
        lkmat[0, 0] = config['StereoLeft']['fc_x']
        lkmat[1, 1] = config['StereoLeft']['fc_y']
        lkmat[0, 2] = config['StereoLeft']['cc_x']
        lkmat[2, 2] = config['StereoLeft']['cc_y']

        rkmat = np.eye(3)
        rkmat[0, 0] = config['StereoRight']['fc_x']
        rkmat[1, 1] = config['StereoRight']['fc_y']
        rkmat[0, 2] = config['StereoRight']['cc_x']
        rkmat[2, 2] = config['StereoRight']['cc_y']

        ld = np.array([config['StereoLeft']['kc_0'], config['StereoLeft']['kc_0'],config['StereoLeft']['kc_1'],
                       config['StereoLeft']['kc_2'],config['StereoLeft']['kc_3'],config['StereoLeft']['kc_4'],
                       config['StereoLeft']['kc_5'], config['StereoLeft']['kc_6'],config['StereoLeft']['kc_7']])
        rd = np.array([config['StereoRight']['kc_0'], config['StereoRight']['kc_0'], config['StereoRight']['kc_1'],
                       config['StereoRight']['kc_2'], config['StereoRight']['kc_3'], config['StereoRight']['kc_4'],
                       config['StereoRight']['kc_5'], config['StereoRight']['kc_6'], config['StereoRight']['kc_7']])

        tvec = np.array([config['StereoRight']['T_0'],config['StereoRight']['T_1'],config['StereoRight']['T_2']])
        rmat = np.zeros((3,3))
        rmat[0, 0] = config['StereoRight']['R_0']
        rmat[0, 1] = config['StereoRight']['R_1']
        rmat[0, 2] = config['StereoRight']['R_2']
        rmat[1, 0] = config['StereoRight']['R_3']
        rmat[1, 1] = config['StereoRight']['R_4']
        rmat[1, 2] = config['StereoRight']['R_5']
        rmat[2, 0] = config['StereoRight']['R_6']
        rmat[2, 1] = config['StereoRight']['R_7']
        rmat[2, 2] = config['StereoRight']['R_8']

        cal = {}
        cal['lkmat'] = lkmat
        cal['rkmat'] = rkmat
        cal['ld'] = ld
        cal['rd'] = rd
        cal['T'] = tvec
        cal['R'] = rmat
        cal['img_size'] = img_size

        return cal
