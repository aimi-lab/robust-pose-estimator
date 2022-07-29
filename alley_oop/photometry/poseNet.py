from alley_oop.photometry.EndoSfMLearner.models.PoseResNet import PoseResNet
import torch
from torch import nn
from torchvision.transforms import Normalize
from alley_oop.photometry.EndoSfMLearner.inverse_warp import pose_vec2mat


def _get_poseNet(ckp='../alley_oop/photometry/EndoSfMLearner/exp_pose_model_best.pth.tar'):
    # load pretrained network
    model = PoseResNet()
    model.load_state_dict(torch.load(ckp)['state_dict'])
    return model


class PoseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _get_poseNet()
        self.model.eval()
        self.transform = Normalize(0.45, 0.225)

    def forward(self, ref_img, trg_img):
        ref_img = self.transform(ref_img)
        trg_img = self.transform(trg_img)

        pose_vec = self.model(ref_img, trg_img)
        pose = torch.eye(4,dtype=pose_vec.dtype, device=pose_vec.device)
        pose[:3,:4] = pose_vec2mat(pose_vec).squeeze()
        return pose
