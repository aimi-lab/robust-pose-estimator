import sys
sys.path.append('../')
import os, glob
import torch
import numpy as np
from tqdm import tqdm
from dataset.dataset_utils import StereoVideoDataset, StereoRectifier
import warnings
from torch.utils.data import DataLoader
import wandb
import cv2
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from alley_oop.fusion.surfel_map import SurfelMap, FrameClass
from alley_oop.metrics.point_cloud_metrics import pcl_absolute_error
from dataset.preprocess.disparity.disparity_model import DisparityModel
import pickle
import open3d


COLORS = np.array([[252,252,252],[51, 87,34], [66,116, 147],[244,148,2],[77,245,136],[155,165,185],[251,252,20],[240, 252, 230]])


class MarkerTracker(object):

    def __init__(self, intrinsics, extrinsics, vertical_disp_thr=10):
        self.prjoection_matrix_l = intrinsics @ np.eye(4)[:3,:]
        self.prjoection_matrix_r = intrinsics @ extrinsics[:3, :]
        self.markers_3d = {"centroids": [], "deformed": [], "c-code": []}
        self.vertical_disp_thr = vertical_disp_thr

    def _thresholding(self, img, color_thr=20, area_min=100, area_max=800, circ_thr=0.8):
        img = cv2.GaussianBlur(img, (5, 5), sigmaX=3)
        markers = {"centroids": [], "c-code": []}
        for i, c in enumerate(COLORS):
            mask = np.ones(img.shape[:2], dtype=bool)
            mask &= ((img > (c -color_thr)) & (img < (c +color_thr))).sum(axis=-1) == 3
            numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 4, cv2.CV_32S)
            w = stats[:, cv2.CC_STAT_WIDTH]
            h = stats[:, cv2.CC_STAT_HEIGHT]
            area = stats[:, cv2.CC_STAT_AREA]

            # filter by area
            filt = (area > area_min) & (area < area_max)
            # filter by ratio
            filt &= (w/h > circ_thr) & (h/w > circ_thr)
            centroids = centroids[filt, :]
            markers["centroids"] += [[c[0], c[1]] for c in centroids]
            markers["c-code"] += [i]*len(centroids)
        markers["centroids"] = np.asarray(markers["centroids"])
        markers["c-code"] = np.asarray(markers["c-code"])
        return markers

    def _assign_markers(self, markers_left, markers_right, max_dist=100000):
        # match markers based on color and minimal distance between left and right image
        dist_matrix = cdist(markers_left["centroids"], markers_right["centroids"])
        mask = np.repeat(markers_left["c-code"][...,None], dist_matrix.shape[1], axis=-1) != np.repeat(markers_right["c-code"][...,None], dist_matrix.shape[0], axis=-1).T
        dist_matrix[mask] = max_dist
        row_ind, col_ind = linear_sum_assignment(dist_matrix)

        # check left-right consistency and valid assignment
        col_ind1, row_ind1 = linear_sum_assignment(dist_matrix.T)
        row_ind_ch, col_ind_ch = [], []
        for i in range(len(row_ind)):
            q, v = row_ind[i], col_ind[i]
            j = np.where(row_ind1 == q)[0]
            if len(j) == 1:
                if (col_ind1[j] == v) & (dist_matrix[q,v] < max_dist):
                    row_ind_ch.append(q)
                    col_ind_ch.append(v)

        markers_left["centroids"] = markers_left["centroids"][row_ind_ch]
        markers_right["centroids"] = markers_right["centroids"][col_ind_ch]
        markers_left["c-code"] = markers_left["c-code"][row_ind_ch]
        markers_right["c-code"] = markers_right["c-code"][col_ind_ch]

        return markers_left, markers_right

    def __call__(self, img_left, img_right, pose, timestamp):
        # detect and match markers in left and right images
        markers_left = self._thresholding(img_left)
        markers_right = self._thresholding(img_right)
        if (len(markers_left["centroids"]) == 0) | (len(markers_right["centroids"]) == 0):
            warnings.warn("no markers found.", UserWarning)
            return self.markers_3d
        markers_left, markers_right = self._assign_markers(markers_left, markers_right)
        # sanity check for vertical disparity (it should be low as images are rectified)
        mask = markers_left["centroids"][:, 1] - markers_right["centroids"][:, 1] < self.vertical_disp_thr
        if mask.sum() < 1:
            warnings.warn("no markers matched.", UserWarning)
            return self.markers_3d

        markers_left["centroids"] = markers_left["centroids"][mask]
        markers_right["centroids"] = markers_right["centroids"][mask]
        markers_left["c-code"] = markers_left["c-code"][mask]
        markers_right["c-code"] = markers_right["c-code"][mask]
        # triangulate marker points in 3D space using DLT
        pts_3d = cv2.triangulatePoints(self.prjoection_matrix_l, self.prjoection_matrix_r, markers_left["centroids"].T, markers_right["centroids"].T)
        pts_3d = cv2.convertPointsFromHomogeneous(pts_3d.T).squeeze(1)

        # transform 3d point in local coordinate frame to world coordinates
        pts_3d_world = (pose[:3,:3] @ pts_3d.T + pose[:3, 3, None]).T
        markers_3d = {"centroids": pts_3d_world, "c-code": markers_left["c-code"]}

        # match points in 3D over time
        if len(self.markers_3d["centroids"]) > 0:
            canonical_markers, current_markers = self._assign_markers(self.markers_3d, markers_3d)
            warpfield = current_markers["centroids"] - canonical_markers["centroids"]
            self.markers_3d["deformed"].append({"timestamp": timestamp,
                                                "centroids": current_markers["centroids"],
                                                "warpfield": warpfield})
            if len(markers_3d["centroids"]) > len(current_markers["centroids"]):  # some points have not been matched an we need to add them
                print("I want to addd")
                #self.markers_3d["warpfield"] = current_markers["centroids"] - canonical_markers["centroids"]
                #self.markers_3d["centroids"]["deformed"] = current_markers["centroids"]
        else:
            self.markers_3d["centroids"] = pts_3d_world
            self.markers_3d["c-code"] = markers_left["c-code"]
            self.markers_3d["deformed"].append({"timestamp": timestamp,
                                                "centroids": pts_3d_world,
                                                "warpfield": np.zeros_like(pts_3d_world)})

        return self.markers_3d

    def evaluate(self, scene:SurfelMap, timestamp=None):
        if timestamp is None:
            idx = -1
        else:
            idx = np.where([k["timestamp"] for k in self.markers_3d["deformed"]] == timestamp)
            assert len(idx) > 0, f"timestamp {timestamp} not found"
        err = {}
        # canonical model error
        err["canoncial"] = pcl_absolute_error(self.markers_3d["centroids"], scene.opts.T.cpu().numpy())
        # deformed model error
        err["deformed"] = pcl_absolute_error(self.markers_3d["deformed"][idx]["centroids"], scene.opts.T.cpu().numpy())
        return err

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self.markers_3d, f)

    def load(self, path: str):
        assert len(self.markers_3d["centroids"]) == 0, "object not empty. loading prohibited."
        with open(path, 'rb') as f:
            self.markers_3d = pickle.load(f)

    def get_mesh(self, timestamp=None):
        if timestamp is None:
            idx = -1
        else:
            idx = np.where([k["timestamp"] for k in self.markers_3d["deformed"]] == timestamp)
            assert len(idx) > 0, f"timestamp {timestamp} not found"
        canonical = open3d.geometry.PointCloud(points=open3d.cuda.pybind.utility.Vector3dVector(self.markers_3d["centroids"]))
        deformed = open3d.geometry.PointCloud(points=open3d.cuda.pybind.utility.Vector3dVector(self.markers_3d["deformed"][idx]["centroids"]))
        return canonical, deformed


def main(input_path, output_path, visualize):

    video_file = glob.glob(os.path.join(input_path, '*.mp4'))[0]
    pose_file = os.path.join(input_path, 'groundthruth.txt')
    calib_file = os.path.join(input_path, 'camera_calibration.json')
    rect = StereoRectifier(calib_file, img_size_new=(640, 512), mode='conventional')
    calib = rect.get_rectified_calib()
    dataset = StereoVideoDataset(video_file, pose_file, img_size=calib['img_size'], rectify=rect)
    loader = DataLoader(dataset, num_workers=1)

    if visualize:
        from viewer.viewer3d import Viewer3D
        viewer = Viewer3D((1024, 1024), blocking=True)
        disp_model = DisparityModel(calibration=calib)

    tracker = MarkerTracker(calib['intrinsics']['left'], calib['extrinsics'])

    with torch.inference_mode():
        os.makedirs(output_path, exist_ok=True)
        for i, data in enumerate(tqdm(loader, total=len(dataset))):
            limg, rimg, pose_kinematics, img_number = data

            sparse_gt = tracker((255*limg.squeeze().permute(1,2,0).numpy()).astype(np.uint8),
                                (255*rimg.squeeze().permute(1,2,0).numpy()).astype(np.uint8),
                                pose_kinematics.squeeze(0).numpy(), int(img_number))

            # show depth and sparse gt
            if visualize:
                disparity, depth, noise = disp_model(limg, rimg)
                frame = FrameClass(limg, depth, intrinsics=torch.tensor(calib['intrinsics']['left']).float())
                pcd = SurfelMap(frame=frame, kmat=torch.tensor(calib['intrinsics']['left']).float(),
                                pmat=pose_kinematics.squeeze(), depth_scale=1)
                print(tracker.evaluate(pcd))
                viewer(pose=pose_kinematics.squeeze(), pcd=tracker.get_mesh()[0], add_pcd=pcd.pcl2open3d(stable=False), def_pcd=tracker.get_mesh()[1])
        tracker.save(os.path.join(output_path, "sparse3d_groundtruth.pckl"))
        print('finished')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='script to extract sparse GT from colored markers')

    parser.add_argument(
        'input',
        type=str,
        help='Path to input folder.'
    )
    parser.add_argument(
        '--outpath',
        type=str,
        help='Path to output folder. If not provided use input path instead.'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='visualize gt-mesh and point-cloud'
    )
    args = parser.parse_args()
    if args.outpath is None:
        args.outpath = os.path.join(args.input)

    main(args.input, args.outpath, args.visualize)
