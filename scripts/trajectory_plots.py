import os
import numpy as np
import argparse
import matplotlib as mpl
mpl.use('Qt5Agg')
from alley_oop.pose.trajectory_analyzer import TrajectoryAnalyzer
from alley_oop.utils.trajectory import read_freiburg
from evaluation.evaluate_ate_freiburg import eval


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
        default=['gt'],
        help='Folder containing predictions.'
    )
    args = parser.parse_args()
    
    colors = ['b', 'g', 'r', 'k', 'm', 'y']
    d_idx = 1

    keyframe = os.path.basename(args.base_path)
    dataset = os.path.basename(os.path.dirname(args.base_path))
    pose_plotter = TrajectoryAnalyzer(title=dataset + '/' + keyframe)

    freiburg_paths = {m: os.path.join(args.base_path, 'data', m, 'trajectory.freiburg') for m in args.pred_folders}
    freiburg_paths.update({'gt': os.path.join(args.base_path, 'groundtruth.txt')})

    for k, meth in enumerate(freiburg_paths):
        print(meth)
        assert os.path.isfile(freiburg_paths[meth]), f'{meth} does not exist'
        ate_rmse, rpe_trans, rpe_rot, error = eval(freiburg_paths[meth], freiburg_paths['gt'])
        print('ATE-RMSE: ',ate_rmse, ' mm')
        print('RPE-trans: ', rpe_trans, ' mm')
        print('RPE_rot: ', rpe_rot)
        pose_arrs = np.stack(read_freiburg(freiburg_paths[meth]))
        color = colors[k%len(colors)]
        pose_plotter.add_pose_trajectory(pose_arrs, label=meth, color=color)
    pose_plotter.legend()
    #pose_plotter.get_rmse_by_idx(idx_a=-1, idx_b=-2, plot_opt=True) if len(args.base_path) > 1 else None
    pose_plotter.write_file()
    pose_plotter.show()
