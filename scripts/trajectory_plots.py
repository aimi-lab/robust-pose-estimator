from pathlib import Path
import numpy as np
import json

from alley_oop.utils.paths import get_scared_abspath
from alley_oop.pose.trajectory_analyzer import TrajectoryAnalyzer


def load_scared_pose(k_idx:int=1, meth='frame_data') -> np.ndarray:

    name_list = sorted((get_scared_abspath(1, k_idx) / 'data' / meth).rglob('*.json'))
    # if kinematics data collect all files
    if str(name_list[0]).lower().__contains__('frame_data'):
        pose_list = []
        for fname in name_list:
            with open(str(fname), 'r') as f: pose_elem = json.load(f)
            pose_list.append(pose_elem['camera-pose'])
    # all other pose estimation methods
    else:
        fname = name_list[0]
        with open(str(fname), 'r') as f: pose_list = json.load(f)
        pose_list = [pose_list[i]['camera-pose'] for i in range(len(pose_list))]
    
    pose_arrs = np.array(pose_list)[:, :3, :4]

    return pose_arrs


if __name__ == '__main__':

    data_path = Path.cwd() / 'tests' / 'test_data'
    meth_dirs = ['orbslam2_rgbd_results', 'frame_data']#'orbslam2_stereo_results', 'defslam_results'
    swap_tvec = False
    colors = ['b', 'g', 'r', 'k', 'm', 'y']
    d_idx = 1

    # iterate through keyframes
    for k_idx in range(1, 4):
        pose_plotter = TrajectoryAnalyzer(title='dataset_'+str(d_idx)+', keyframe_'+str(k_idx))
        for k, meth in enumerate(meth_dirs):
            pose_arrs = load_scared_pose(k_idx, meth)
            if swap_tvec:
                rmat_arrs = pose_arrs[:3, 1:4]
                tvec_arrs = pose_arrs[:3, 0]
                pose_arrs = np.concatenate([pose_arrs[:, :3, 1:4], pose_arrs[:, :3, 0, np.newaxis]], axis=-1)
            color = colors[k%len(colors)]
            pose_plotter.add_pose_trajectory(pose_arrs, label=meth, color=color)
        pose_plotter.legend()
        pose_plotter.get_rmse_by_idx(idx_a=-1, idx_b=-2, plot_opt=True)
        pose_plotter.write_file()
        pose_plotter.show()
