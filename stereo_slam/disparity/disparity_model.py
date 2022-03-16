import torch
import torch.nn as nn
import yaml
import os
from stereo_slam.disparity.sttr_light.sttr import STTR


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


class DisparityModel(nn.Module):
    def __init__(self, config_yaml, calibration, infer_depth=True):
        super().__init__()
        with open(config_yaml, 'r') as ymlfile:
            config = Struct(**yaml.load(ymlfile, Loader=yaml.SafeLoader))

        self.model = self._load(config)
        self.infer_depth = infer_depth
        self.baseline_f = torch.nn.Parameter(calibration['bf'])

    def forward(self, limg, rimg):
        with torch.no_grad():
            out = self.model(limg, rimg)
            if self.infer_depth:
                out = self.baseline_f / out
        return out

    def _load(self, config):
        model = STTR(config)
        if os.path.isfile(config.loadmodel):
            checkpoint = torch.load(config.loadmodel)
            model.load_state_dict(checkpoint['model_dict'])
        return model
