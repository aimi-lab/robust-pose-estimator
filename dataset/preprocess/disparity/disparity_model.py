import torch
import torch.nn as nn
import yaml
import os
from dataset.preprocess.disparity.psmnet.stacked_hourglass import PSMNet
from torchvision.transforms import ToTensor, Resize, Normalize
import torch.nn.functional as F
from alley_oop.metrics.projected_photo_metrics import disparity_photo_loss
import warnings

IMAGENET_STATS = {'mean': [0.485, 0.456, 0.406],
                   'std': [0.229, 0.224, 0.225]}

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
        self.normalize = Normalize(**IMAGENET_STATS)

        if config['conf_metric'] == 'matching_score':
            self.conf_func = lambda x: self._matching_score_measure(x, scale=config["score_scale"])
        elif config['conf_metric'] == 'maximum_likelihood':
            self.conf_func = lambda x: self._maximum_likelihood_measure(x, temperature=config["temperature"])
        elif config['conf_metric'] == 'entropy':
            self.conf_func = lambda x: self._entropy_measure(x, temperature=config["temperature"])
        else:
            raise NotImplementedError

    def forward(self, limg, rimg):
        with torch.no_grad():
            limg = self.normalize(self.resize_for_inf(limg))
            rimg = self.normalize(self.resize_for_inf(rimg))
            disp, cost_volume = self.model(limg, rimg)
            disp = self.resize_to_orig(disp)*self.upscale_factor
            # confidence prediction
            disp_noise = self.conf_func(cost_volume)
            disp_noise = self.resize_to_orig(disp_noise)
            depth = self.baseline_f / disp
            depth_noise = self._noise_propagation(disp_noise, depth, disp)

        return disp, depth, depth_noise

    def _load(self, config):
        model = PSMNet(config)
        assert os.path.isfile(config['loadmodel']), 'no PSMNet model loaded'
        checkpoint = torch.load(config['loadmodel'], map_location='cpu')
        for key in list(checkpoint['state_dict'].keys()):
            checkpoint['state_dict'][key.replace('module.', '')] = checkpoint['state_dict'].pop(key)
        model.load_state_dict(checkpoint['state_dict'])
        return model

    def _matching_score_measure(self, cost_volume, scale):
        # disparity noise estimation using MLM
        max_match_score = torch.sigmoid(torch.max(cost_volume, dim=1)[0])*scale #ToDo is there any other way to get a positive value?
        noise_var = 1 / (2 * torch.pi * max_match_score)
        return noise_var.unsqueeze(0)

    def _entropy_measure(self, cost_volume, temperature):
        # disparity noise using entropy
        sft_pred = F.softmax(cost_volume / temperature, dim=1)
        entropy = -torch.nansum(sft_pred * torch.log(sft_pred), dim=1)
        # assuming Gaussian distribution of the disparity we compute the noise variance
        noise_var = torch.exp(2*entropy-1)/(2*torch.pi)
        return noise_var.unsqueeze(0)

    def _maximum_likelihood_measure(self, cost_volume, temperature):
        # disparity noise estimation using MLM
        sft_pred = F.softmax(cost_volume / temperature, dim=1)
        max_sft_pred = torch.max(sft_pred, dim=1)[0]
        # assuming Gaussian distribution of the disparity we compute the noise variance
        noise_var = 1/(2*torch.pi*max_sft_pred**2)
        return noise_var.unsqueeze(0)

    def _lr_consistency(self, cost_volume, disp, img_l, img_r):
        # disparity noise estimation using MLM
        conf = disparity_photo_loss(img_l, img_r, disp, alpha=5.437).unsqueeze(0)
        return 1/(2*torch.pi*conf**2)


    def _noise_propagation(self, disparity_noise, depth, disp):
        # theroetically we propagate the noise to depth sigma_depth = |depth*sigma_disp/disp|
        # however we observe that the noise is not at all Gaussian and the predicted depth may be way off,
        # leading to high confidence when depth is small
        return disparity_noise #depth * disparity_noise/ disp