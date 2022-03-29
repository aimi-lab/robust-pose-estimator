import cv2
import numpy as np
from alley_oop.metrics.projected_photo_metrics import synth_view
import open3d
from viewer.view_render import Render
from scipy.spatial.transform import Rotation as R
from matplotlib import cm
import time

class SlamViewer(object):
    def __init__(self, camera_intrinsics, config, scale=2):
        self.config = config
        self.src_img = None
        self.src_kpts = None
        self.ref_img = None
        self.ref_pose = None
        self.intrinsics = camera_intrinsics
        self.intrinsics[:2, :] /= scale
        self.scale = scale

    def set_reference(self, ref_img, ref_depth, ref_pose=np.eye(4)):
        self.ref_img = ref_img[::self.scale, ::self.scale]
        self.ref_depth = ref_depth[::self.scale, ::self.scale]
        self.ref_pose = ref_pose

    def __call__(self, trg_img, trg_kpts, matches, pose):
        if not isinstance(trg_kpts[0], cv2.KeyPoint):
            trg_kpts = [cv2.KeyPoint(kp[0], kp[1], 1.0) for kp in trg_kpts]
        if self.config['matches'] & (self.src_img is not None):
            if not isinstance(matches[0], cv2.DMatch):
                matches = [cv2.DMatch(m[0], m[1], m[2]) for m in matches]

            out_img = cv2.drawMatches(cv2.cvtColor(self.src_img, cv2.COLOR_RGB2BGR),self.src_kpts,
                                      cv2.cvtColor(trg_img, cv2.COLOR_RGB2BGR), trg_kpts,
                                      matches, None, 0.5)
            if self.config['show']: cv2.imshow('matches', out_img)
        if self.config['view_synthesis'] & (self.ref_img is not None):
            rel_pose = np.linalg.pinv(pose)@self.ref_pose
            synth = synth_view(self.ref_img, self.ref_depth[None,...], rel_pose[:3,:3], rel_pose[:3,3][:,None], self.intrinsics)
            synth_img = np.empty((self.ref_img.shape[0], 3*self.ref_img.shape[1], 3), dtype=np.uint8)
            synth_img[:, :self.ref_img.shape[1]] = trg_img[::self.scale, ::self.scale]
            synth_img[:, self.ref_img.shape[1]:2*self.ref_img.shape[1]] = synth
            synth_img[:, 2*self.ref_img.shape[1]:] = self.ref_img
            if self.config['show']: cv2.imshow('view synth', cv2.cvtColor(synth_img, cv2.COLOR_RGB2BGR))
        if self.config['show']: cv2.waitKey(100)
        self.src_img = trg_img
        self.src_kpts = trg_kpts


class Viewer3d(object):
    def __init__(self, max_error: float=10.0, blocking=False):
        self.pcd_gt = open3d.geometry.PointCloud()
        self.pcd_predicted = open3d.geometry.PointCloud()
        self.render3d = Render(self.pcd_gt, blocking=blocking)
        self.max_error = max_error

    def __call__(self, gt_pcl, predicted_pcl, gt_colors=None, error_dists=None):
        self.pcd_gt.points = open3d.utility.Vector3dVector(gt_pcl)
        if gt_colors is not None:
            self.pcd_gt.colors = open3d.utility.Vector3dVector(gt_colors / 255.0)
        self.pcd_predicted.points = open3d.utility.Vector3dVector(predicted_pcl)
        if error_dists is not None:
            colormap = cm.get_cmap('CMRmap')
            error_colors = colormap(np.clip(error_dists, 0, self.max_error)/self.max_error)[:,:3]
            self.pcd_predicted.colors = open3d.utility.Vector3dVector(error_colors)
        pose = np.eye(4)
        pose[:3, :3] = R.from_euler('xyz', [10.0, 10.0, 0.0], degrees=True).as_matrix()
        pose[:3, 3] = [-10,10,0]
        _ = self.render3d.render(pose, self.pcd_gt, add_pcd=self.pcd_predicted)