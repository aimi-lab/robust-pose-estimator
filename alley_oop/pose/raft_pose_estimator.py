from alley_oop.pose.PoseN import PoseN
from alley_oop.geometry.pinhole_transforms import inv_transform
import torch
from alley_oop.fusion.surfel_map import SurfelMap, FrameClass
from alley_oop.geometry.lie_3d import lie_se3_to_SE3
from collections import OrderedDict
from torchvision.transforms import Resize


class RAFTPoseEstimator(torch.nn.Module):
    def __init__(self, intrinsics: torch.Tensor, baseline: float, checkpoint: str, frame2frame: bool=False, init_pose: torch.tensor=torch.eye(4)):
        """

        """
        super(RAFTPoseEstimator, self).__init__()
        checkp = torch.load(checkpoint)
        model = PoseN(checkp['config']['model'])
        new_state_dict = OrderedDict()
        state_dict = checkp['state_dict']
        for k, v in state_dict.items():
            name = k.replace('module.', '')  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        model.eval()
        self.model = model
        self.intrinsics = intrinsics.unsqueeze(0).float()
        self.last_pose= torch.nn.Parameter(init_pose.float(), requires_grad=False)
        self.cost = [0]
        self.last_frame = None
        self.frame2frame = frame2frame
        self.baseline = torch.tensor(baseline).unsqueeze(0).float()

    def estimate(self, frame: FrameClass, scene: SurfelMap):
        if self.frame2frame:
            if self.last_frame is not None:
                rel_pose_se3 = self.model(255*self.last_frame.img, 255*frame.img, self.intrinsics, self.baseline, depth1=self.last_frame.depth, depth2=frame.depth,
                                          mask1=self.last_frame.mask, mask2=frame.mask)[1].squeeze(0)
                rel_pose = lie_se3_to_SE3(rel_pose_se3)
                ret_frame = self.last_frame
            else:
                rel_pose = torch.eye(4, dtype=torch.float32, device=self.last_pose.device)
                ret_frame = None
            self.last_frame = frame
        else:
            # transform scene to last camera pose coordinates
            scene_tlast = scene.transform_cpy(inv_transform(self.last_pose))
            model_frame = scene_tlast.render(self.intrinsics)
            rel_pose_se3 = self.model(255*model_frame.img, 255*frame.img, depth1=model_frame.depth, depth2=frame.depth, mask1=model_frame.mask, mask2=frame.mask)[1].squeeze(0)
            rel_pose = lie_se3_to_SE3(rel_pose_se3)
            ret_frame = model_frame
        self.last_pose.data = self.last_pose.data @ rel_pose
        return self.last_pose, ret_frame

    @property
    def device(self):
        return self.last_pose.device

    def estimate_depth(self, img_l, img_r):
        return self.model.flow2depth(255*img_l, 255*img_r, self.baseline)

