from alley_oop.pose.defPoseN import *
from alley_oop.geometry.pinhole_transforms import inv_transform
import torch
from alley_oop.fusion.surfel_map import SurfelMap, FrameClass
from alley_oop.geometry.lie_3d_pseudo import pseudo_lie_se3_to_SE3
from collections import OrderedDict
from torchvision.transforms import Resize
import warnings
from typing import Tuple


class RAFTPoseEstimator(torch.nn.Module):
    def __init__(self, intrinsics: torch.Tensor, baseline: float, checkpoint: str, img_shape: Tuple,
                 frame2frame: bool=False, init_pose: torch.tensor=torch.eye(4), scale: float=1.0):
        """

        """
        super(RAFTPoseEstimator, self).__init__()
        checkp = torch.load(checkpoint)
        checkp['config']['model']['image_shape'] = img_shape
        def load():
            new_state_dict = OrderedDict()
            state_dict = checkp['state_dict']
            for k, v in state_dict.items():
                name = k.replace('module.', '')  # remove `module.`
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        try:
            model = PoseN(checkp['config']['model'])
            load()
        except RuntimeError:
            model = DefPoseN(checkp['config']['model'])
            load()
        model.eval()
        self.model = model
        self.intrinsics = intrinsics.unsqueeze(0).float()
        self.last_pose= torch.nn.Parameter(init_pose.double(), requires_grad=False)
        self.cost = [0]
        self.last_frame = None
        self.frame2frame = frame2frame
        self.baseline = torch.tensor(baseline).unsqueeze(0).float().to(intrinsics.device)
        self.scale = scale

    def estimate(self, frame: FrameClass, scene: SurfelMap):
        success = True
        flow, crsp_list = None, None
        if self.frame2frame:
            if self.last_frame is not None:
                if isinstance(self.model, DefPoseN):
                    # we should not mask tools but provide a tool-mask to the conf-head.
                    # ToDo we consider the mask as tools but it can also be invalid depth or similar.
                    flow, rel_pose_se3, *_, conf_1, conf_2 = self.model(255*self.last_frame.img, 255*frame.img,
                                                                        self.intrinsics, self.baseline,
                                                                        image1r=self.last_frame.rimg, image2r=frame.rimg,
                                                                        toolmask1=self.last_frame.mask, toolmask2=frame.mask,
                                                                        ret_confmap=True)
                else:
                    flow, rel_pose_se3, *_, conf_1, conf_2 = self.model(255 * self.last_frame.img, 255 * frame.img,
                                                                        self.intrinsics, self.baseline,
                                                                        depth1=self.last_frame.depth,
                                                                        depth2=frame.depth,
                                                                        mask1=self.last_frame.mask, mask2=frame.mask,
                                                                        flow1=self.last_frame.flow, flow2=frame.flow,
                                                                        ret_confmap=True)
                rel_pose_se3 = rel_pose_se3.squeeze(0)
                frame.confidence = conf_2
                self.last_frame.confidence = conf_1
                flow = flow[-1]
                if (torch.isnan(rel_pose_se3).any()) | (torch.abs(rel_pose_se3) > 1.0e-1).any():
                    # pose estimation failed, keep last image as reference and skip this one
                    warnings.warn('pose estimation not converged, skip.', RuntimeWarning)
                    rel_pose = torch.eye(4, dtype=torch.float64, device=self.last_pose.device)
                    success = False
                else:
                    rel_pose = pseudo_lie_se3_to_SE3(rel_pose_se3.double())
                ret_frame = self.last_frame
            else:
                rel_pose = torch.eye(4, dtype=torch.float64, device=self.last_pose.device)
                ret_frame = None
            self.last_frame = frame
        else:
            # transform scene to last camera pose coordinates
            if isinstance(self.model, DefPoseN):
                raise NotImplementedError
            scene_tlast = scene.transform_cpy(inv_transform(self.last_pose.float()))
            model_frame, crsp_list = scene_tlast.render(self.intrinsics.squeeze())
            flow, rel_pose_se3, *_, conf_1, conf_2  = self.model(255*model_frame.img, 255*frame.img, self.intrinsics, self.baseline,
                                      depth1=model_frame.depth, depth2=frame.depth,
                                      mask1=model_frame.mask, mask2=frame.mask,
                                      flow1=model_frame.flow, flow2=frame.flow, ret_confmap=True)
            rel_pose_se3 = rel_pose_se3.squeeze(0)
            flow = flow[-1]
            frame.confidence = conf_2
            model_frame.confidence = conf_1
            if (torch.isnan(rel_pose_se3).any()) | (torch.abs(rel_pose_se3) > 1.0e-1).any():
                # pose estimation failed, keep last image as reference and skip this one
                warnings.warn('pose estimation not converged, skip.', RuntimeWarning)
                rel_pose = torch.eye(4, dtype=torch.float64, device=self.last_pose.device)
                success = False
            else:
                rel_pose = pseudo_lie_se3_to_SE3(rel_pose_se3.double())
            ret_frame = model_frame
        self.last_pose.data = self.last_pose.data @ rel_pose
        return self.last_pose.float(), ret_frame, success, flow, crsp_list

    @property
    def device(self):
        return self.last_pose.device

    def estimate_depth(self, img_l, img_r):
        return self.model.flow2depth(255*img_l, 255*img_r, self.baseline*self.scale)

