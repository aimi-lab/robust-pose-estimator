from alley_oop.interpol.gauss_pyramid import FrameGaussPyramid
from alley_oop.pose.frame_class import FrameClass
from alley_oop.pose.icp_rgb_pose_estimation import RGBICPPoseEstimator
from alley_oop.pose.rotation_estimation import RotationEstimator
import torch
from typing import Tuple


class PyramidPoseEstimator(torch.nn.Module):
    """ This is an implementation of Elastic Fusion pose estimation
    It estimates the camera rotation and translation in three stages and pyramid levels
"""

    def __init__(self, intrinsics: torch.Tensor, config: dict):
        """

        """
        super(PyramidPoseEstimator, self).__init__()
        self.pyramid = FrameGaussPyramid(level_num=config['pyramid_levels']-1, intrinsics=intrinsics)
        self.config = config
        self.last_frame_pyr = None
        self.last_pose = torch.nn.Parameter(torch.eye(4, dtype=intrinsics.dtype))

    def estimate(self, frame:FrameClass, model):
        # transform model to last camera pose coordinates
        model.transform(torch.linalg.inv(self.last_pose))
        #frame.plot()
        # apply gaussian pyramid to current and rendered images
        frame_pyr, intrinsics_pyr = self.pyramid(frame)

        if self.last_frame_pyr is not None:
            # render view of model from last camera pose
            model_frame = model.render(self.pyramid._top_instrinsics)
            model_frame_pyr, _ = self.pyramid(model_frame)
            #model_frame.plot()
            # compute SO(3) pre-alignment from previous image to current image
            rot_estimator = RotationEstimator(frame_pyr[-1].shape, intrinsics_pyr[-1],
                                               self.config['rot']['n_iter'], self.config['rot']['Ftol']).to(self.device)
            R_last2cur, *_ = rot_estimator.estimate(frame_pyr[-1], self.last_frame_pyr[-1])

            T_last2cur = torch.eye(4, dtype=R_last2cur.dtype, device=R_last2cur.device)
            T_last2cur[:3,:3] = R_last2cur  # initial guess is rotation only
            # combined icp + rgb pose estimation
            for pyr_level in range(len(frame_pyr)-1, -1, -1):
                pose_estimator = RGBICPPoseEstimator(frame_pyr[pyr_level].shape, intrinsics_pyr[pyr_level],
                                                     config['icp_weight'],
                                                     config['n_iter'][pyr_level]).to(self.device)
                T_last2cur, *_ = pose_estimator.estimate_gn(frame_pyr[pyr_level], model_frame_pyr[pyr_level], model, init_pose=T_last2cur)
                self.plot(T_last2cur, ref_frame, model, self.pyramid._top_instrinsics)
            pose = self.last_pose @ T_last2cur
            self.last_pose.data = pose
        self.last_frame_pyr = frame_pyr
        return self.last_pose

    @property
    def device(self):
        return self.last_pose.device

    def plot(self, T, ref_frame, model, intrinsics):
        ref_pcl = PointCloud()
        ref_pcl.from_depth(ref_frame.depth, intrinsics, normals=ref_frame.normals)
        ref_pcl.set_colors(ref_frame.img)
        ref_pcl.transform(torch.linalg.inv(T))

        from viewer.view_render import Render
        viewer = Render(model.pcl2open3d(), blocking=True)
        viewer.render(np.eye(4),add_pcd=ref_pcl.pcl2open3d() )

from alley_oop.utils.pfm_handler import load_pfm
import cv2
from pathlib import Path
import numpy as np
from alley_oop.geometry.point_cloud import PointCloud
from scipy.spatial.transform import Rotation as R
from alley_oop.interpol.synth_view import synth_view

with torch.no_grad():
    config = {
        'pyramid_levels': 3,
        'rot': {'n_iter': 10, 'Ftol': 1e-3},
        'icp_weight': 0.0000001,
        'n_iter': [3,4,10]
    }
    scale = 2
    disparity, _ = load_pfm(str(Path.cwd()/'../..' / 'tests' / 'test_data' / '0006.pfm'))
    h, w = (int(disparity.shape[0] / scale), int(disparity.shape[1] / scale))
    disparity = cv2.resize(disparity, (w, h))
    # background is very far, make it appear closer
    rand_background = 17 + 6 * np.random.rand(int((disparity < 20).sum()))
    disparity[disparity < 20] = rand_background
    depth = torch.tensor(1050.0 / disparity).double()
    img = torch.tensor(cv2.cvtColor(cv2.resize(cv2.imread(str(Path.cwd() /'../..' / 'tests' / 'test_data' / '0006.png')),
                                   (w, h)), cv2.COLOR_RGB2BGR)).float() / 255.0

    # generate dummy intrinsics and dummy images
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    R_true = torch.tensor(R.from_euler('xyz', (0.0, 2.0, 5.0), degrees=True).as_matrix()).double()
    t_true = torch.tensor([0.0, 1.0, 3.0]).double()
    T_true = torch.eye(4).double()
    T_true[:3, :3] = R_true
    T_true[:3, 3] = t_true
    intrinsics = torch.tensor([[1050.0/scale, 0, 479.5/scale],
                                       [0, 1050.0/scale, 269.5/scale],
                                       [0, 0, 1]]).double()
    img = img.permute(2, 0, 1).unsqueeze(0)
    ref_frame = FrameClass(img.double(), depth.unsqueeze(0).unsqueeze(0), intrinsics=intrinsics)

    ref_pcl = PointCloud()
    ref_pcl.from_depth(depth, intrinsics, normals=ref_frame.normals)
    ref_pcl.set_colors(ref_frame.img)

    target_pcl = ref_pcl.transform_cpy(T_true)
    target_frame = target_pcl.render(intrinsics)
    mask = (target_frame.img_gray[0, 0] != 0)
    estimator = PyramidPoseEstimator(intrinsics, config).to(device)
    with torch.no_grad():
        T = estimator.estimate(target_frame.to(device), target_pcl.to(device))
        T = estimator.estimate(ref_frame.to(device), target_pcl.to(device))
    print(torch.linalg.inv(T), T_true)


