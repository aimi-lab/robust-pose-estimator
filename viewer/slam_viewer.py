import cv2
import numpy as np
from alley_oop.metrics.projected_photo_metrics import synth_view


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
