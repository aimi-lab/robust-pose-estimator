import pandas as pd
import wandb
api = wandb.Api()

METHODS = ['alleyoop_scared', 'efusion', 'orbslam2']

# Download data from WANDB

# Project is specified by <entity/project-name>
runs = api.runs("hayoz/Alley-OOP")

summary_list = []
for run in runs:
    # .summary contains the output keys/values for metrics like accuracy.
    #  We call ._json_dict to omit large files
    all_dict = {}
    all_dict.update(run.summary._json_dict)

    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    all_dict.update(
        {k: v for k,v in run.config.items()
          if not k.startswith('_')})

    # .name is the human-readable name of the run.
    all_dict.update({"run_name": run.name})
    all_dict.update({"state": run.state})
    all_dict.update({"method": run._attrs['group']})
    all_dict.update({"id": run.id})
    summary_list.append(all_dict)

runs_df = pd.DataFrame(summary_list)

#########################
print("this is a fix for the scared Benchmarking. The GT files were not correct such that we have to recompute the error locally")
import os
from scripts.evaluate_ate_freiburg import main as evaluate
import numpy as np
for i, run in runs_df.iterrows():
    if run['method'] in METHODS:
        datapath = run['dataset'].replace('/storage/workspaces/artorg_aimi/ws_00000', '/home/mhayoz/research')
        gt_file = os.path.join(datapath, run['keyframe'], 'groundtruth.txt')
        method = run['method']
        method = method.replace('alleyoop_scared', 'alley_oop')
        pred_file = os.path.join(datapath, run['keyframe'], 'data', method ,'trajectory.freiburg')

        assert os.path.isfile(gt_file), f'missing {gt_file}'
        assert os.path.isfile(pred_file), f'missing {pred_file}'

        error = evaluate(pred_file, gt_file) * 1000
        runs_df.loc[i,'ATE/RMSE'] = np.sqrt(np.dot(error, error) / len(error))

#########################

runs_df.to_csv("project.csv")

# Group into methods and datasets
print('\n------------')
print('ATE-RMSE in mm')
for method in METHODS:
    print('\n------------')
    print(method)
    df = runs_df[runs_df.method.eq(method)]
    print('average duration in frames:', df['frame'].mean(), '+/-', df['frame'].std())
    df = pd.DataFrame({'mean': df.groupby('dataset').mean()['ATE/RMSE'], 'std':df.groupby('dataset').std()['ATE/RMSE']})
    print(df)
    print('average:', df.mean()['mean'],'+/-', df.std()['mean'])