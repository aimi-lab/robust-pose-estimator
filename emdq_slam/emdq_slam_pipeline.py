import cv2
import numpy as np
from alley_oop.geometry.absolute_pose_quarternion import align
from alley_oop.geometry.opencv_utils import kpts2npy
from emdq import pyEMDQ

class EmdqSLAM(object):
    def __init__(self, camera):
        self.detector_descriptor = cv2.xfeatures2d.SURF_create(nOctaves=1)
        self.matcher = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=False)
        self.last_descriptors = None
        self.camera = camera
        self.lowes_ratio = 1.0
        self.emdq = pyEMDQ(1.0)
        self.last_scale = 1.0
        self.last_pose = np.eye(4)

    def __call__(self, img, depth, mask=None):
        # extract key-points and descriptors
        kps, features = self.detector_descriptor.detectAndCompute(img, mask)
        # project key-points to 3d
        kps_depth = np.asarray([depth[int(kp.pt[1]), int(kp.pt[0])] for kp in kps])
        kps3d = self.camera.project3d(kpts2npy(kps).T, kps_depth).T
        pose = np.eye(4)

        if self.last_descriptors is not None:
            # perform brute-force matching
            rawMatches = self.matcher.knnMatch(features, self.last_descriptors[1], k=2, compactResult=True)
            # Lowe's ratio test
            filt_matches = []
            for m in rawMatches:
                if len(m) == 2 and m[0].distance < m[1].distance * self.lowes_ratio:
                    filt_matches.append((m[0].queryIdx, m[0].trainIdx, m[0].distance))

            # run EMDQ
            self.emdq.SetScale(self.last_scale) # do we need to set the scale in 3d?
            inliers = self.emdq.fit(kps3d, self.last_descriptors[0], filt_matches)
            deformationfield = self.emdq.predict(kps3d)
            deformed_kps3d = deformationfield[:,:3]
            sigma_error = deformationfield[:, 4]
            # factor rigid, non-rigid components
            query_pts = np.float32([deformed_kps3d[m[0], :] for m in filt_matches])
            reference_pts = np.float32([self.last_descriptors[0][m[1], :] for m in filt_matches])
            diff_pose, residuals, scale = align(reference=reference_pts.T, query=query_pts.T, ret_homogenous=True)
            self.last_scale = scale
            # chain poses
            pose = self.last_pose@diff_pose
            self.last_pose = pose
            # run ARAP

        self.last_descriptors = (kps3d, features)
        return pose


