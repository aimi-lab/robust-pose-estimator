from pathlib import Path
import numpy as np
import json

from alley_oop.utils.paths import get_scared_abspath
from alley_oop.pose.trajectory_analyzer import TrajectoryAnalyzer


if __name__ == '__main__':

    data_path = Path.cwd() / 'tests' / 'test_data'
    meth_dirs = ['orbslam2_rgbd_results', 'frame_data']#'orbslam2_stereo_results', 'defslam_results'
    swap_tvec = False
    colors = ['b', 'g', 'r', 'k', 'm', 'y']
    d_idx = 1

    # iterate through keyframes
    for k_idx in range(1, 4):
        data_path = get_scared_abspath(1, k_idx) / 'data'
        pose_plotter = TrajectoryAnalyzer(title='dataset_'+str(d_idx)+', keyframe_'+str(k_idx))
        # iterate through methods
        for k, meth in enumerate(meth_dirs):
            name_list = sorted((data_path / meth).rglob('*.json'))
            # if kinematics data collect all files
            if meth.lower().__contains__('frame_data'):
                pose_list = []
                for j, fname in enumerate(name_list):
                    with open(str(fname), 'r') as f: pose_elem = json.load(f)
                    pose_list.append(pose_elem['camera-pose'])
                pose_arrs = np.array(pose_list)[:, :3, :4]
            # all other pose estimator methods
            else:
                fname = name_list[0]
                with open(str(fname), 'r') as f: pose_list = json.load(f)
                pose_arrs = np.array([pose_list[i]['camera-pose'] for i in range(len(pose_list))])[:, :3, :4]
            if swap_tvec:
                rmat_arrs = pose_arrs[:3, 1:4]
                tvec_arrs = pose_arrs[:3, 0]
                pose_arrs = np.concatenate([pose_arrs[:, :3, 1:4], pose_arrs[:, :3, 0, np.newaxis]], axis=-1)
            color = colors[k%len(colors)]
            pose_plotter.add_pose_trajectory(pose_arrs, label=fname.parent.name, color=color)
        pose_plotter.legend()
        pose_plotter.get_rmse_by_idx(idx_a=-1, idx_b=-2, plot_opt=True)
        pose_plotter.write_file()
        pose_plotter.show()
