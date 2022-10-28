import pandas as pd
import wandb
import seaborn as snb
import matplotlib.pyplot as plt
import os
import numpy as np
import glob
from evaluation.evaluate_ate_freiburg import eval


base = '/home/mhayoz/Intuitive/02-Data/experiments'
methods = glob.glob(os.path.join(base, '*/'))
for method in methods:
    print(method)
    files = sorted(glob.glob(os.path.join(method, '*.freiburg')))
    if len(files) == 0:
        continue
    rpe_trans = {'d2':[], 'd40':[], 'd41':[], 'd42':[], 'd43':[], 'd44':[], 'd45':[], 'd46':[], 'd47':[], 'd48':[], 'd5':[], 'd6':[]}
    rpe_rot = {'d2':[], 'd40':[], 'd41':[], 'd42':[], 'd43':[], 'd44':[], 'd45':[], 'd46':[], 'd47':[], 'd48':[], 'd5':[], 'd6':[]}
    for file in files:
        for key in rpe_trans:
            if key in file:
                gt_file = os.path.join(base,'gt',f'{key}.txt')
                ate_rmse, _, _, trans_error, rpe_trans_e, rpe_rot_e = eval(gt_file, file, offset=-4)
                rpe_trans[key].append(rpe_trans_e)
                rpe_rot[key].append(rpe_rot_e)
    rpe_trans_d4 = []
    rpe_rot_d4 = []
    for key in ['d40','d41', 'd42', 'd43', 'd44', 'd45', 'd46', 'd47', 'd48']:
        rpe_trans_d4.append(rpe_trans[key])
        rpe_rot_d4.append(rpe_rot[key])
    rpe_trans['d4'] = [np.column_stack(rpe_trans_d4).squeeze()]
    rpe_rot['d4'] = [np.column_stack(rpe_rot_d4).squeeze()]
    print('RPE-TRANS')
    means = []
    for key in rpe_trans:
        if key in ['d40','d41', 'd42', 'd43', 'd44', 'd45', 'd46', 'd47', 'd48']:
            continue
        mean = np.mean(np.concatenate(rpe_trans[key]))
        std = np.std(np.concatenate(rpe_trans[key]))
        print(f'{key}: {mean:.3f} +/- {std:.3f}')
        means.append(mean)
    print(f'average: {np.mean(means):.3f} +/- {np.std(means):.3f}')
    print('RPE-ROT')
    means = []
    for key in rpe_rot:
        if key in ['d40','d41', 'd42', 'd43', 'd44', 'd45', 'd46', 'd47', 'd48']:
            continue
        mean = np.mean(np.concatenate(rpe_rot[key]))*180/np.pi
        std = np.std(np.concatenate(rpe_rot[key]))*180/np.pi
        print(f'{key}: {mean:.3f} +/- {std:.3f}')
        means.append(mean)
    print(f'average: {np.mean(means):.3f} +/- {np.std(means):.3f}')
