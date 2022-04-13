import sys
sys.path.append('../..')
import cv2
import numpy as np
from alley_oop.geometry.absolute_pose_quarternion import align
from alley_oop.geometry.pinhole_transforms import create_img_coords_np
from alley_oop.utils.opencv import kpts2npy
from emdq import pyEMDQ
from scipy.spatial.distance import pdist
from scipy.spatial import cKDTree
from .superglue.superglue import SuperGlueMatcher
import torch


class EmdqSLAM(object):
    def __init__(self, camera, config):
        self.detector_descriptor = cv2.xfeatures2d.SURF_create(nOctaves=1, hessianThreshold=config['surf_thr'])
        self.matcher = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=config['match_cross_check'])
        self.last_descriptors = None
        self.camera = camera
        self.lowes_ratio = config['lowes_ratio']
        self.spacing = config['node_spacing']


        self.emdq = pyEMDQ(1.0)
        self.last_pose = np.eye(4)
        self.matches = []
        self.img_kps = []
        self.displacements = []
        self.node_spacing = np.inf
        self.nodes = None
        self.dist_tree = None  # cached k-dtree for efficient control node query

    def __call__(self, img, depth, mask=None):
        if self.nodes is None: self.init_nodes(depth)
        return self.track(img, depth, mask)

    def init_nodes(self, depth):
        ipts = create_img_coords_np(depth.shape[0], depth.shape[1],self.spacing)
        node_depth = depth[::self.spacing, ::self.spacing].reshape(2, -1)
        self.nodes = self.camera.project3d(ipts, node_depth).T
        self.displacements = np.zeros_like(self.nodes)
        self.node_spacing = 2*np.mean(pdist(self.nodes))

    def track(self, img, depth, mask=None):
        # extract key-points and descriptors
        img_kps, features = self.detector_descriptor.detectAndCompute(img, mask)
        # project key-points to 3d
        kps_depth = np.asarray([depth[int(kp.pt[1]), int(kp.pt[0])] for kp in img_kps])
        kps3d = self.camera.project3d(kpts2npy(img_kps).T, kps_depth).T
        pose = np.eye(4)
        inliers = -1
        filt_matches = []

        if self.last_descriptors is not None:
            # perform brute-force matching
            rawMatches = self.matcher.knnMatch(self.last_descriptors[1], features, k=2, compactResult=True)
            # Lowe's ratio test
            for m in rawMatches:
                if len(m) == 2 and m[0].distance < m[1].distance * self.lowes_ratio:
                    filt_matches.append((m[0].queryIdx, m[0].trainIdx, m[0].distance))

            # run EMDQ
            inliers = self.emdq.fit(self.last_descriptors[0], kps3d,  filt_matches)
            if inliers > 0:
                nodes_last_frame = self.warp_canonical_model()
                deformationfield = self.emdq.predict(nodes_last_frame)
                #sigma_error = deformationfield[:, 4]
                deformed_kps3d = deformationfield[:,:3]
                # factor rigid, non-rigid components
                diff_pose, residuals, _ , displacements = align(reference=deformed_kps3d.T, query=nodes_last_frame.T, ret_homogenous=True)
                # apply deformation by updating node displacements
                # (nodes = canonical model, displacements are the warping to current frame)
                self.displacements += (self.last_pose[:3, :3].T @ displacements).T
                # chain poses
                pose = diff_pose@self.last_pose
                self.last_pose = pose
                self.add_node(depth)

                # run ARAP
        self.matches = filt_matches
        self.img_kps = img_kps
        self.last_descriptors = (kps3d, features)
        return pose, inliers

    def get_matching_res(self):
        return self.img_kps, self.matches

    def warp_canonical_model(self, current_reference=True):
        if current_reference:
            return self.warp_rigid(self.nodes) + self.displacements
        else:
            return self.nodes + self.displacements

    def warp_rigid(self, points3d, inverse=False):
        if inverse:
            return (points3d - self.last_pose[:3, 3]) @ self.last_pose[:3, :3]
        else:
            return points3d @ self.last_pose[:3, :3].T + self.last_pose[:3, 3]

    def add_node(self, depth):
        # project image points to 3d
        ipts = create_img_coords_np(depth.shape[0], depth.shape[1], step=self.spacing)
        node_depth = depth[::self.spacing, ::self.spacing].reshape(2, -1)
        candidate_nodes = self.warp_rigid(self.camera.project3d(ipts, node_depth).T, inverse=True)
        # check if new nodes need to be added. We add a control point when there is no point in the neighbourhood
        if self.dist_tree is None:
            self.dist_tree = cKDTree(self.nodes)
        dists = self.dist_tree.query(candidate_nodes, distance_upper_bound=self.node_spacing, k=1)[0]
        candidate_nodes = candidate_nodes[dists > self.node_spacing]
        # add to canonical model
        if len(candidate_nodes) > 0:
            self.nodes = np.row_stack((self.nodes, self.warp_rigid(candidate_nodes, inverse=True)))
            self.displacements = np.row_stack((self.displacements, np.zeros((candidate_nodes.shape[0], 3))))
            self.dist_tree = cKDTree(self.nodes)


class EmdqGlueSLAM(EmdqSLAM):
    def __init__(self, camera, config, device=torch.device('cpu')):
        super().__init__(camera, config)
        self.matcher = SuperGlueMatcher(device)
        self.detector_descriptor = None
        self.kps3d_last = None

    def track(self, img, depth, mask=None):
        # extract key-points and descriptors
        img_kps, filt_matches = self.matcher(img, mask)
        # project key-points to 3d
        kps_depth = np.asarray([depth[int(kp[1]), int(kp[0])] for kp in img_kps])
        kps3d = self.camera.project3d(img_kps.T, kps_depth).T
        pose = np.eye(4)
        inliers = -1

        if self.kps3d_last is not None:
            # run EMDQ
            inliers = self.emdq.fit(self.kps3d_last, kps3d,  filt_matches)
            if inliers > 0:
                nodes_last_frame = self.warp_canonical_model()
                deformationfield = self.emdq.predict(nodes_last_frame)
                #sigma_error = deformationfield[:, 4]
                deformed_kps3d = deformationfield[:,:3]
                # factor rigid, non-rigid components
                diff_pose, residuals, _ , displacements = align(reference=deformed_kps3d.T, query=nodes_last_frame.T, ret_homogenous=True)
                # apply deformation by updating node displacements
                # (nodes = canonical model, displacements are the warping to current frame)
                self.displacements += (self.last_pose[:3, :3].T @ displacements).T
                # chain poses
                pose = diff_pose@self.last_pose
                self.last_pose = pose
                self.add_node(depth)

                # run ARAP
        self.matches = filt_matches
        self.img_kps = img_kps
        self.kps3d_last = kps3d
        return pose, inliers

