import torch
import torch.nn as nn
import yaml
import os
from stereo_slam.disparity.sttr_light.sttr import STTR
from stereo_slam.disparity.sttr_light.utilities.misc import NestedTensor
from torchvision.transforms import ToTensor
import warnings


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


class DisparityModel(nn.Module):
    def __init__(self, calibration, device=torch.device('cpu'), infer_depth=True, depth_clipping=(-torch.inf, torch.inf)):
        super().__init__()
        with open('stereo_slam/disparity/sttr_light/STTR.yaml', 'r') as ymlfile:
            config = Struct(**yaml.load(ymlfile, Loader=yaml.SafeLoader))

        self.model = self._load(config)
        self.infer_depth = infer_depth
        self.baseline_f = torch.nn.Parameter(torch.tensor(calibration['bf']))
        self.device = device
        self.model.eval()
        self.to(device)
        self.tensor = ToTensor()
        self.clipping = depth_clipping

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
            out = self.model(NestedTensor(limg, rimg))
            out = out['disp_pred']
            clip_mask = torch.ones_like(out, dtype=bool)
            if self.infer_depth:
                out = self.baseline_f / out
                out = torch.clip(out, self.clipping[0], self.clipping[1])
                clip_mask = (out != self.clipping[0]) & (out != self.clipping[1])
        if is_numpy:
            out = out.cpu().numpy()
            clip_mask = clip_mask.cpu().numpy()
        return out.squeeze(), clip_mask.squeeze()

    def _load(self, config):
        model = STTR(config)
        if os.path.isfile(config.loadmodel):
            checkpoint = torch.load(config.loadmodel)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            warnings.warn('no STTR model loaded', UserWarning)
        return model
