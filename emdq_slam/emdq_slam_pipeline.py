import cv2
import numpy as np
from alley_oop.geometry.absolute_pose_quarternion import align
from alley_oop.geometry.opencv_utils import kpts2npy
from emdq import pyEMDQ

class EmdqSLAM(object):
    def __init__(self, camera):
        self.detector_descriptor = cv2.xfeatures2d.SURF_create(nOctaves=1, hessianThreshold=300)
        self.matcher = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=False)
        self.last_descriptors = None
        self.camera = camera
        self.lowes_ratio = 0.3
        self.emdq = pyEMDQ(1.0)
        self.last_pose = np.eye(4)
        self.matches = []
        self.img_kps = []

    def __call__(self, img, depth, mask=None):
        # extract key-points and descriptors
        img_kps, features = self.detector_descriptor.detectAndCompute(img, mask)
        # project key-points to 3d
        kps_depth = np.asarray([depth[int(kp.pt[1]), int(kp.pt[0])] for kp in img_kps])
        kps3d = self.camera.project3d(kpts2npy(img_kps).T, kps_depth).T
        pose = np.eye(4)
        inliers = -1
        emdq_matches = []

        if self.last_descriptors is not None:
            # perform brute-force matching
            rawMatches = self.matcher.knnMatch(self.last_descriptors[1], features, k=2, compactResult=True)
            # Lowe's ratio test
            filt_matches = []
            for m in rawMatches:
                if len(m) == 2 and m[0].distance < m[1].distance * self.lowes_ratio:
                    filt_matches.append((m[0].queryIdx, m[0].trainIdx, m[0].distance))

            # run EMDQ
            inliers = self.emdq.fit(self.last_descriptors[0], kps3d,  filt_matches)
            if inliers > 0:
                deformationfield = self.emdq.predict(self.last_descriptors[0])
                deformed_kps3d = deformationfield[:,:3]
                sigma_error = deformationfield[:, 4]
                emdq_matches = []
                for m in filt_matches:
                    if sigma_error[m[0]] < 20**2: emdq_matches.append(m)
                # factor rigid, non-rigid components
                #ToDo what do we use the deformation for?
                reference_pts = np.float32([deformed_kps3d[m[0], :] for m in emdq_matches])
                query_pts = np.float32([self.last_descriptors[0][m[0], :] for m in emdq_matches])
                diff_pose, residuals, _ = align(reference=reference_pts.T, query=query_pts.T, ret_homogenous=True)
                # chain poses
                pose = diff_pose@self.last_pose
                self.last_pose = pose
                # run ARAP
        self.matches = emdq_matches
        self.img_kps = img_kps
        self.last_descriptors = (kps3d, features)
        return pose, inliers

    def get_matching_res(self):
        return self.img_kps, self.matches


