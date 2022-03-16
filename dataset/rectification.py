import cv2
import os
import numpy as np
import configparser
import json
from stereo_slam.disparity.stereo_rectify import rectify_pair, get_rect_maps


class StereoRectifier(object):
    def __init__(self, calib_file, img_size_new=None):
        if os.path.splitext(calib_file)[1] == '.json':
            cal = self._load_calib_json(calib_file)
        elif os.path.splitext(calib_file)[1] == '.ini':
            cal = self._load_calib_ini(calib_file)
        elif os.path.splitext(calib_file)[1] == '.yaml':
            cal = self._load_calib_yaml(calib_file)
        else:
            raise NotImplementedError

        if img_size_new is not None:
            # scale intrinsics
            scale = img_size_new[0]/cal['img_size'][0]
            assert scale == img_size_new[1]/cal['img_size'][1]
            cal['lkmat'] *= scale
            cal['rkmat'] *= scale
            cal['img_size'] = img_size_new
        self.img_size = cal['img_size']

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
        calib_rectifed = {'intrinsics': {}}
        calib_rectifed['intrinsics']['left'] = self.l_intr[:3,:3]
        calib_rectifed['intrinsics']['right'] = self.r_intr[:3,:3]
        calib_rectifed['extrinsics'] = np.eye(4)
        calib_rectifed['extrinsics'][:3,3] = np.array([self.r_intr[0, 3] / self.r_intr[0, 0], 0., 0.]) # Tx*f, see cv2 website
        calib_rectifed['bf'] = np.sqrt(np.sum(calib_rectifed['extrinsics'][:3, 3] ** 2))*self.l_intr[0, 0]
        calib_rectifed['img_size'] = self.img_size
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

    @staticmethod
    def _load_calib_yaml(fname):
        fs = cv2.FileStorage(fname, cv2.FILE_STORAGE_READ)
        img_size = (int(fs.getNode('Camera.width').real()), int(fs.getNode('Camera.height').real()))

        lkmat = fs.getNode('M1').mat()
        rkmat = fs.getNode('M2').mat()

        ld = fs.getNode('D1').mat()
        rd = fs.getNode('D2').mat()

        tvec = fs.getNode('T').mat()
        rmat = fs.getNode('R').mat()

        cal = {}
        cal['lkmat'] = lkmat
        cal['rkmat'] = rkmat
        cal['ld'] = ld
        cal['rd'] = rd
        cal['T'] = tvec
        cal['R'] = rmat
        cal['img_size'] = img_size

        return cal
