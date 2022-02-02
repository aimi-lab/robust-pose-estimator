from pathlib import Path
import numpy as np
import json
import argparse
from scipy.spatial.transform import Rotation

from alley_oop.utils.paths import get_scared_abspath
from alley_oop.pose.trajectory_analyzer import TrajectoryAnalyzer


def load_scared_pose(d_idx:int=1, k_idx:int=1, meth='frame_data') -> np.ndarray:

    name_list = sorted((get_scared_abspath(d_idx, k_idx) / 'data' / meth).rglob('*.json'))
    # if kinematics data collect all files
    if str(name_list[0]).lower().__contains__('frame_data'):
        pose_list = []
        for fname in name_list:
            with open(str(fname), 'r') as f: pose_elem = json.load(f)
            pose = np.array(pose_elem['camera-pose'])
            pose[0:3, 3] = -pose[0:3, 3] # neg. translation as Intuitive's coordinate system is inverted wrt. OpenCV
            pose[0:3, 0:3] = pose[0:3, 0:3].T # invert rotation as Intuitive's coordinate system is inverted wrt. OpenCV
            pose_list.append(pose)
    # all other pose estimation methods
    else:
        fname = name_list[0]
        with open(str(fname), 'r') as f: pose_list = json.load(f)
        pose_list = [pose_list[i]['camera-pose'] for i in range(len(pose_list))]
    
    pose_arrs = np.array(pose_list)[:, :4, :4]

    return pose_arrs


def load_elastic_fusion_pose(meth, scale=20.0):
    trajectory_file = list(Path(meth).glob('*.klg.freiburg'))
    assert len(trajectory_file) == 1
    with open(str(trajectory_file[0]), 'r') as f: pose_list = np.loadtxt(f)
    pose_arrs = []
    time_stamps = []
    for p in pose_list:
        rel_pose_arr = np.eye(4)
        rel_pose_arr[:3,3] = scale*p[1:4] # translation
        quaternion = Rotation.from_quat(p[4:8])
        rel_pose_arr[:3,:3] = quaternion.as_matrix()
        time_stamps.append(p[0])
        pose_arrs.append(np.linalg.pinv(rel_pose_arr))

    pose_arrs = np.asarray(pose_arrs)
    time_stamps = np.asarray(time_stamps)

    return pose_arrs


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='ORB SLAM example')

    parser.add_argument(
        'base_path',
        type=str,
        help='Path to scared dataset.'
    )
    parser.add_argument(
        '--pred_folder',
        type=str,
        nargs='+',
        default=['orbslam2_rgbd_results'],
        help='Folder containing predictions.'
    )

    try:
        args = parser.parse_args()
        meth_dirs = []
        for pred_folder in args.pred_folder:
            assert (Path(args.base_path) / pred_folder).exists()
            meth_dirs += [str(Path(args.base_path) / pred_folder)]
    except:
        meth_dirs = ['orbslam2_stereo_results', 'frame_data'] #, 'orbslam2_rgbd_results'
    
    colors = ['b', 'g', 'r', 'k', 'm', 'y']
    d_idx = 1

    # iterate through keyframes
    for k_idx in range(1, 4):
        pose_plotter = TrajectoryAnalyzer(title='dataset_'+str(d_idx)+', keyframe_'+str(k_idx))
        for k, meth in enumerate(meth_dirs):
            if 'elastic_fusion' in meth:
                pose_arrs = load_elastic_fusion_pose(meth)
            else:
                pose_arrs = load_scared_pose(d_idx=d_idx, k_idx=k_idx, meth=meth)
            print(len(pose_arrs))
            color = colors[k%len(colors)]
            pose_plotter.add_pose_trajectory(pose_arrs, label=meth, color=color)
        pose_plotter.legend()
        pose_plotter.get_rmse_by_idx(idx_a=-1, idx_b=-2, plot_opt=True) if len(meth_dirs) > 1 else None
        pose_plotter.write_file()
        pose_plotter.show()
