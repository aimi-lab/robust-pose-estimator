from pathlib import Path
from typing import List
import numpy as np
import json
import argparse
import matplotlib as mpl
mpl.use('Qt5Agg')
from alley_oop.utils.paths import get_scared_abspath
from alley_oop.pose.trajectory_analyzer import TrajectoryAnalyzer


def load_scared_pose(fnames:List=None) -> np.ndarray:

    # if kinematics data collect all files
    if str(fnames[0]).lower().__contains__('frame_data') or str(fnames[0]).lower().__contains__('superglue'):
        pose_list = []
        for fname in fnames:
            with open(str(fname), 'r') as f: pose_elem = json.load(f)
            pose = np.array(pose_elem['camera-pose'])
            pose[0:3, 3] = -pose[0:3, 3] # neg. translation as Intuitive's coordinate system is inverted wrt. OpenCV
            pose[0:3, 0:3] = pose[0:3, 0:3].T # invert rotation as Intuitive's coordinate system is inverted wrt. OpenCV
            pose_list.append(pose)
    # all other pose estimation methods
    else:
        fname = fnames[0]
        with open(str(fname), 'r') as f: pose_list = json.load(f)
        pose_list = [pose_list[i]['camera-pose'] for i in range(len(pose_list))]
    
    pose_arrs = np.array(pose_list)[:, :4, :4]

    return pose_arrs


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Plot Trajectories')

    parser.add_argument(
        'base_path',
        type=str,
        help='Path to scared dataset.'
    )
    parser.add_argument(
        '--pred_folders',
        type=str,
        nargs='+',
        default='orbslam2_rgbd_results',
        help='Folder containing predictions.'
    )
    args = parser.parse_args()
    
    colors = ['b', 'g', 'r', 'k', 'm', 'y']
    d_idx = 1

    # iterate through keyframes
    for k_idx in range(1, 4):
        pose_plotter = TrajectoryAnalyzer(title='dataset_'+str(d_idx)+', keyframe_'+str(k_idx))
        for k, meth in enumerate(args.pred_folders):
            print(meth)
            fnames = sorted((Path(args.base_path) / 'data' / meth).rglob('*.json'))
            pose_arrs = load_scared_pose(fnames)
            pose_arrs[:, :3, -1] = np.cumsum(pose_arrs[:, :3, -1], axis=0)/-5 if meth.lower().__contains__('defslam') else pose_arrs[:, :3, -1]
            pose_arrs[:, :3, -1] = np.cumsum(pose_arrs[:, :3, -1]*np.array([-1, 1, -1]), axis=0) if meth.lower().__contains__('superglue') else pose_arrs[:, :3, -1]
            pose_arrs[:, :2, -1] = pose_arrs[:, :2, -1][:, ::-1] if meth.lower().__contains__('superglue') else pose_arrs[:, :2, -1]
            color = colors[k%len(colors)]
            pose_plotter.add_pose_trajectory(pose_arrs, label=meth, color=color)
        pose_plotter.legend()
        pose_plotter.get_rmse_by_idx(idx_a=-1, idx_b=-2, plot_opt=True) if len(args.base_path) > 1 else None
        pose_plotter.write_file()
        pose_plotter.show()
