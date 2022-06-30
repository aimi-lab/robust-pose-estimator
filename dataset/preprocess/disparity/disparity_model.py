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
    def __init__(self, calibration, device=torch.device('cpu')):
        super().__init__()
        with open('../dataset/preprocess/disparity/psmnet/PSMNet.yaml', 'r') as ymlfile:
            #config = Struct(**yaml.load(ymlfile, Loader=yaml.SafeLoader))
            config = yaml.load(ymlfile, Loader=yaml.SafeLoader)

        self.model = self._load(config)
        self.baseline_f = torch.nn.Parameter(torch.tensor(calibration['bf']))
        self.device = device
        self.model.eval()
        self.to(device)
        self.tensor = ToTensor()
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

            disp = self.model(limg, rimg)
            disp = self.resize_to_orig(disp)*self.upscale_factor
            clip_mask = torch.ones_like(disp, dtype=bool)
            depth = self.baseline_f / disp
        if is_numpy:
            disp = disp.cpu().numpy()
            depth = depth.cpu().numpy()
        return disp, depth

    def _load(self, config):
        model = PSMNet(config)
        assert os.path.isfile(config['loadmodel']), 'no PSMNet model loaded'
        checkpoint = torch.load(config['loadmodel'], map_location='cpu')
        for key in list(checkpoint['state_dict'].keys()):
            checkpoint['state_dict'][key.replace('model.module.', '')] = checkpoint['state_dict'].pop(key)
        model.load_state_dict(checkpoint['state_dict'])
        return model
