import numpy as np
from core.metrics.trajectory_metrics import absolute_trajectory_error, relative_pose_error
from core.utils.trajectory import read_freiburg


def eval(gt_list:str, pred_list:str, delta:int=1, offset:int=0, ret_align_T=False):
    if not isinstance(gt_list, dict):
        gt_list, gt_stamps = read_freiburg(gt_list, ret_stamps=True)
        gt_list = {key: pose for key, pose in zip(gt_stamps, gt_list)}
    if not isinstance(pred_list, dict):
        pred_list, pred_stamps = read_freiburg(pred_list, ret_stamps=True)
        pred_list = {key: pose for key, pose in zip(pred_stamps, pred_list)}

    # we assume exact synchronization (same time-stamps in gt and prediction)
    pred_keys = sorted(list(pred_list.keys()))
    gt_keys = sorted(list(gt_list.keys()))
    pred_poses = []
    gt_poses = []
    for k in pred_keys:
        if (k+offset > 0) & (k+offset < max(gt_keys)):
            pred_poses.append(pred_list[k])
            gt_poses.append(gt_list[k+offset])
    pred_poses = np.stack(pred_poses)
    gt_poses = np.stack(gt_poses)

    ate_rmse, trans_error, transform = absolute_trajectory_error(gt_poses, pred_poses, ret_align_T=True)
    rpe_trans, rpe_rot = relative_pose_error(gt_poses, pred_poses, delta=delta)
    if ret_align_T:
        return ate_rmse, np.mean(rpe_trans), np.mean(rpe_rot), trans_error, rpe_trans, rpe_rot, transform, gt_poses
    return ate_rmse, np.mean(rpe_trans), np.mean(rpe_rot), trans_error, rpe_trans, rpe_rot


if __name__=="__main__":
    # parse command line
    import argparse
    parser = argparse.ArgumentParser(description="Compute Trajectory Metrics")
    parser.add_argument('gt_file', help='ground truth trajectory (format: timestamp tx ty tz qx qy qz qw)', type=str)
    parser.add_argument('pred_file', help='estimated trajectory (format: timestamp tx ty tz qx qy qz qw)', type=str)
    parser.add_argument('--delta', help='interval for relative pose error', type=int, default=1)
    args = parser.parse_args()

    ate_rmse, rpe_trans, rpe_rot, trans_error = eval(args.gt_file, args.pred_file, args.delta)


    print("compared_pose_pairs %d pairs"%(len(trans_error)))

    print("absolute_translational_error.rmse %f m"%np.sqrt(np.dot(trans_error,trans_error) / len(trans_error)))
    print("absolute_translational_error.mean %f m"%np.mean(trans_error))
    print("absolute_translational_error.median %f m"%np.median(trans_error))
    print("absolute_translational_error.std %f m"%np.std(trans_error))
    print("absolute_translational_error.min %f m"%np.min(trans_error))
    print("absolute_translational_error.max %f m"%np.max(trans_error))

