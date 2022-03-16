import torch
import torch.nn as nn
import yaml
import os
from stereo_slam.disparity.sttr_light.sttr import STTR
from torchvision.transforms import ToTensor


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


class DisparityModel(nn.Module):
    def __init__(self, calibration, device=torch.device('cpu'), infer_depth=True):
        super().__init__()
        with open('stereo_slam/disparity/sttr_light/STTR.yaml', 'r') as ymlfile:
            config = Struct(**yaml.load(ymlfile, Loader=yaml.SafeLoader))

        self.model = self._load(config)
        self.infer_depth = infer_depth
        self.baseline_f = torch.nn.Parameter(calibration['bf'])
        self.device = device
        self.model.eval()
        self.to(device)
        self.tensor = ToTensor()

    def forward(self, limg, rimg):
        is_numpy = False
        if not torch.is_tensor(limg):
            limg = self.tensor(limg).to(self.device)
            rimg = self.tensor(rimg).to(self.device)
            is_numpy = True
        if limg.ndim == 3:
            limg = limg.unsqueeze(0)
            rimg = rimg.unsqueeze(0)
        with torch.no_grad():
            out = self.model(limg, rimg)
            if self.infer_depth:
                out = self.baseline_f / out
        if is_numpy:
            out = out.cpu().numpy()
        return out.squeeze()

    def _load(self, config):
        model = STTR(config)
        if os.path.isfile(config.loadmodel):
            checkpoint = torch.load(config.loadmodel)
            model.load_state_dict(checkpoint['model_dict'])
        return model
