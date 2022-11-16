import cv2
import os
import numpy as np
import glob
import torch
import warnings
from torch.utils.data import Dataset
from dataset.transforms import ResizeStereo, ResizeRGBD
from typing import Tuple

from alley_oop.utils.pfm_handler import load_pfm

LABEL_LOOKUP = [
    {
        "name" : "background",
        "classid" : 0,
        "color" : [0,0,0]
    },
    {
        "name" : "instrument-shaft",
        "classid" : 1,
        "color" : [0,255,0]
    },
    {
        "name" : "instrument-clasper",
        "classid" : 2,
        "color" : [0,255,255]
    },
    {
        "name" : "instrument-wrist",
        "classid" : 3,
        "color" : [125,255,12]
    },
    {
        "name": "kidney-parenchyma",
        "classid": 4,
        "color" : [255, 55, 0]
    },
    {
        "name": "covered-kidney",
        "classid": 5,
        "color": [24, 55, 125]
    },
    {
        "name": "thread",
        "classid": 6,
        "color": [128, 155, 25]
    },
    {
        "name": "clamps",
        "classid": 7,
        "color": [0, 255, 125]
    },
    {
        "name" : "suturing-needle",
        "classid" : 8,
        "color" : [255,255,125]
    },
    {
        "name" : "suction-instrument",
        "classid" : 9,
        "color" : [123,15, 175]
    },
    {
        "name": "intestine",
        "classid": 10,
        "color": [124, 155, 5]
    },
    {
        "name" : "ultrasound-probe",
        "classid" : 11,
        "color" : [12, 255, 141]
    },
    {
        "name" : "cannula",
        "classid" : 12,
        "color" : [146, 142, 110]
    },
    {
        "name": "renal-vessel",
        "classid": 13,
        "color": [63, 21, 139]
    },
    {
        "name": "aorta-vena-cava",
        "classid": 14,
        "color": [152, 203, 116]
    },
    {
        "name": "ureter",
        "classid": 15,
        "color": [239, 103, 235]
    },
    {
        "name": "unidentified-tubular",
        "classid": 16,
        "color": [62, 52, 6]
    },
    {
        "name": "resectioned-tissue",
        "classid": 17,
        "color": [218, 69, 178]
    },
    {
        "name": "resectioned-cavity",
        "classid": 18,
        "color": [63, 122, 61]
    },
    {
        "name": "stomach",
        "classid": 19,
        "color": [121, 117, 248]
    },
    {
        "name": "large-intestine",
        "classid": 20,
        "color": [146, 240, 63]
    },
    {
        "name": "spleen",
        "classid": 21,
        "color": [47, 42, 31]
    },
    {
        "name": "liver",
        "classid": 22,
        "color": [224, 3, 117]
    },
    {
        "name": "clotted-blood",
        "classid": 23,
        "color": [94, 54, 212]
    },
    {
        "name": "smoke",
        "classid": 24,
        "color": [18, 238, 167]
    },
    {
        "name" : "vessel-clamps",
        "classid" : 25,
        "color" : [120,179,35]
    },
    {
        "name" : "gauze",
        "classid" : 26,
        "color" : [51,127,228]
    },
    {
        "name": "gallbladder",
        "classid": 27,
        "color": [105, 38, 209]
    },
    {
        "name": "pancreas",
        "classid": 28,
        "color": [210, 3, 104]
    },
    {
        "name": "cystic-artery",
        "classid": 29,
        "color": [146, 188, 38]
    },
    {
        "name" : "hernia-mesh",
        "classid" : 30,
        "color" : [119,170,204]
    },
    {
        "name": "hernia",
        "classid": 31,
        "color": [250, 254, 29]
    },
    {
        "name": "pooling-blood",
        "classid": 32,
        "color": [15, 175, 47]
    },
    {
        "name" : "elastic-band",
        "classid" : 33,
        "color" : [190,248,146]
    },
    {
        "name" : "plastic-tube",
        "classid" : 34,
        "color" : [82,136,253]
    },
    {
        "name" : "misc-device",
        "classid" : 35,
        "color" : [204,131,108]
    },
    {
        "name": "plastic-bag",
        "classid" : 36,
        "color" : [0, 243, 46]
    },
    {
        "name": "gel-cap",
        "classid": 37,
        "color": [143, 107, 120]
    },
    {
        "name": "stapler",
        "classid": 38,
        "color": [ 214, 14, 216]
    },
    {
        "name": "lung",
        "classid": 39,
        "color": [154, 114, 16]
    },
    {
        "name": "uterus",
        "classid": 40,
        "color": [12, 58, 116]
    },
    {
        "name": "ovary",
        "classid": 41,
        "color": [122, 158, 116]
    },
    {
        "name":"catheter",
        "classid":42,
        "color":[12,125,95]
    },
    {
        "name":"liver-retractor",
        "classid":43,
        "color":[245,34,95]
    },
    {
        "name": "undefined",
        "classid": 44,
        "color": [100, 100, 100]
    }
]


class RGBDecoder(object):
    def __init__(self):
        # Get keys and values
        k = np.array([k['color'] for k in LABEL_LOOKUP])
        self.backdecode = k
        self.decode_v = np.array([k['classid'] for k in LABEL_LOOKUP])
        # Setup scale array for dimensionality reduction
        s = 256 ** np.arange(3)
        # Reduce k to 1D
        self.decode_k1D = k.dot(s)
        # Get sorted k1D and correspondingly re-arrange the values array
        self.decode_sidx = self.decode_k1D.argsort()

    def rgbDecode(self, rgblabel):
        rgblabel = rgblabel.dot(256 ** np.arange(3))
        # set all values that are not in the list to background
        rgblabel[~np.isin(rgblabel, self.decode_k1D)] = 0
        decoded_label = self.decode_v[self.decode_sidx[np.searchsorted(self.decode_k1D, rgblabel,
                                                                       sorter=self.decode_sidx)]].astype(np.uint8)
        return decoded_label

    def getToolMask(self, rgblabel):
        decoded_label = self.rgbDecode(rgblabel)
        mask = np.isin(decoded_label, [1,2,3,8,9,11,12])
        decoded_label[mask] = 0
        decoded_label[~mask] = 1
        return decoded_label

    def colorize(self, multiclass):
        return self.backdecode[multiclass]

    def mergelabels(self, orig_label):
        # merge tool parts and other tools into tool_class
        orig_label[orig_label == 2] = 1  # clasper
        orig_label[orig_label == 3] = 1  # wrist
        orig_label[orig_label == 9] = 1  # suction instrument
        orig_label[orig_label == 11] = 1  # ultrasound probe
        orig_label[orig_label == 12] = 1  # cannula
        # intestine and large intestine
        #orig_label[orig_label == 20] = 10  # merge intestine classes
        orig_label[orig_label == 5] = 4  # merge kidney and covered kidney
        return orig_label


def mask_specularities(img, mask=None, spec_thr=0.96):
    spec_mask = img.sum(axis=-1) < (3 * 255*spec_thr)
    mask = mask & spec_mask if mask is not None else spec_mask
    mask = cv2.erode(mask.astype(np.uint8), kernel=np.ones((11, 11)))
    return mask


class StereoDataset(Dataset):
    def __init__(self, input_folder:str, img_size:Tuple):
        super().__init__()
        self.imgs = sorted(glob.glob(os.path.join(input_folder, 'video_frames*', '*l.png')))
        self.semantics = sorted(glob.glob(os.path.join(input_folder, 'semantic_predictions*', '*l.png')))
        assert len(self.imgs) == len(self.semantics)
        assert len(self.imgs) > 0

        self.transform = ResizeStereo(img_size)
        self.rgb_decoder = RGBDecoder()

    def __getitem__(self, item):
        img_l = cv2.cvtColor(cv2.imread(self.imgs[item]), cv2.COLOR_BGR2RGB)
        img_r = cv2.cvtColor(cv2.imread(self.imgs[item].replace('l.png', 'r.png')), cv2.COLOR_BGR2RGB)
        img_number = os.path.basename(self.imgs[item]).split('l.png')[0]
        semantic = cv2.cvtColor(cv2.imread(self.semantics[item]), cv2.COLOR_BGR2RGB)
        mask = self.rgb_decoder.getToolMask(semantic)
        semantic = self.rgb_decoder.mergelabels(self.rgb_decoder.rgbDecode(semantic))
        # mask specularities
        mask = mask_specularities(img_l, mask)
        # to torch tensor
        img_l = torch.tensor(img_l).permute(2, 0, 1).float()
        img_r = torch.tensor(img_r).permute(2, 0, 1).float()
        mask = torch.tensor(mask).unsqueeze(0)
        semantic = torch.tensor(semantic).unsqueeze(0)

        data = self.transform(img_l, img_r, mask, semantic)
        return (*data, img_number)

    def __len__(self):
        return len(self.imgs)


class RGBDDataset(Dataset):
    def __init__(self, input_folder:str, img_size:Tuple):
        super().__init__()
        self.imgs = sorted(glob.glob(os.path.join(input_folder, 'video_frames', '*l.png')))
        self.semantics = sorted(glob.glob(os.path.join(input_folder, 'semantic_predictions', '*l.png')))
        self.depths = sorted(glob.glob(os.path.join(input_folder, 'depth', '*l.pfm')))
        assert len(self.imgs) == len(self.semantics)
        assert len(self.imgs) == len(self.depths)
        assert len(self.imgs) > 0

        self.transform = ResizeRGBD(img_size)
        self.rgb_decoder = RGBDecoder()

    def __getitem__(self, item):
        img_l = cv2.cvtColor(cv2.imread(self.imgs[item]), cv2.COLOR_BGR2RGB)
        depth = load_pfm(self.depths[item])[0]
        img_number = os.path.basename(self.imgs[item]).split('l.png')[0]
        semantic = cv2.cvtColor(cv2.imread(self.semantics[item]), cv2.COLOR_BGR2RGB)
        mask = self.rgb_decoder.getToolMask(semantic)
        semantic = self.rgb_decoder.mergelabels(self.rgb_decoder.rgbDecode(semantic))
        # to torch tensor
        img_l = torch.tensor(img_l).permute(2,0,1).float()
        mask = torch.tensor(mask).unsqueeze(0)
        semantic = torch.tensor(semantic).unsqueeze(0)
        depth = torch.tensor(depth.copy()).unsqueeze(0)*250
        # mask specularities
        mask = mask_specularities(img_l, mask)
        data = self.transform(img_l, depth, mask, semantic)
        return (*data, img_number)

    def __len__(self):
        return len(self.imgs)



