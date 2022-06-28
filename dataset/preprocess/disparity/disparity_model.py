import torch
import torch.nn as nn
import yaml
import os
from dataset.preprocess.disparity.psmnet.stacked_hourglass import PSMNet
from torchvision.transforms import ToTensor, Resize
import warnings


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


class DisparityModel(nn.Module):
    def __init__(self, calibration, device=torch.device('cpu'), infer_depth=True, depth_clipping=(-float('inf'), float('inf'))):
        super().__init__()
        with open('stereo_slam/disparity/psmnet/PSMNet.yaml', 'r') as ymlfile:
            config = Struct(**yaml.load(ymlfile, Loader=yaml.SafeLoader))

        self.model = self._load(config)
        self.infer_depth = infer_depth
        self.baseline_f = torch.nn.Parameter(torch.tensor(calibration['bf']))
        self.device = device
        self.model.eval()
        self.to(device)
        self.tensor = ToTensor()
        self.clipping = depth_clipping
        # network input size
        self.th, self.tw = 256, 512
        self.upscale_factor = calibration['img_size'][0]/self.tw
        self.resize_for_inf = Resize((self.th, self.tw))
        self.resize_to_orig = Resize((calibration['img_size'][1], calibration['img_size'][0]))

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
            limg = self.resize_for_inf(limg)
            rimg = self.resize_for_inf(rimg)

            out = self.model(limg, rimg)
            out = self.resize_to_orig(out)*self.upscale_factor
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
        model = PSMNet(config)
        assert os.path.isfile(config.loadmodel), 'no PSMNet model loaded'
        checkpoint = torch.load(config.loadmodel)
        model.load_state_dict(checkpoint['state_dict'])
        return model
