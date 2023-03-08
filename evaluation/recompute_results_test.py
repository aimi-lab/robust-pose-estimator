import pandas as pd
import wandb
import seaborn as snb
import matplotlib.pyplot as plt
import os
import numpy as np

api = wandb.Api()
import argparse
import glob
import os
from evaluation.evaluate_ate_freiburg import eval, get_traj_length


parser = argparse.ArgumentParser(description='Inspect WandB results')
parser.add_argument(
    '--project',
    type=str,
    default="/home/mhayoz/research/innosuisse_surgical_robot/01_Datasets/05_slam/intuitive_human",
    help='Path to input folder.'
)

parser.add_argument(
    '--method',
    type=str,
    default='ours',
    help='Path to input folder.'
)
args = parser.parse_args()

datasets = sorted(glob.glob(os.path.join(args.project, '*/')))
ate_rmse_list = []
rpe_rot_list = []
rpe_trans_list = []

for dataset in datasets:
    files = glob.glob(os.path.join(dataset, 'data', 'test_*', args.method, 'trajectory.freiburg'))
    gt_file = os.path.join(dataset, 'groundtruth.txt')
    if not 'dataset_4' in dataset:
        ate_rmse_list = []
        rpe_rot_list = []
        rpe_trans_list = []
        valid_list = []
    # if len(files) == 0:
    #     print(dataset)
    #     print('ttl: ', get_traj_length(gt_file))
    if len(files) > 0:
        for file in files:
            print(dataset)
            print(file)
            # print('ttl: ', get_traj_length(gt_file, file))
            ate_rmse, rpe_trans, rpe_rot, trans_error, rpe_trans_e, rpe_rot_e, *_, valid = eval(gt_file, file, offset=0, ignore_failed_pos=True, ret_align_T=True)
            ate_rmse_list.append(ate_rmse)
            rpe_rot_list.append(rpe_rot_e)
            rpe_trans_list.append(rpe_trans_e)
            valid_list.append(valid)
        if (not 'dataset_4' in dataset) | ('dataset_4_8' in dataset):
            print(dataset)
            print('ate: ', np.asarray(ate_rmse_list).mean())
            print('trans: ', np.concatenate(rpe_trans_list).mean())
            print('rot: ', np.concatenate(rpe_rot_list).mean()*180/np.pi)
            print('valid: ', np.concatenate(valid_list).mean())



