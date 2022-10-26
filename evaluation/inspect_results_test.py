import pandas as pd
import wandb
import seaborn as snb
import matplotlib.pyplot as plt
import os
import numpy as np

api = wandb.Api()
import argparse

parser = argparse.ArgumentParser(description='Inspect WandB results')

parser.add_argument(
    '--methods',
    nargs='+',
    type=str,
    default=['test_orbslam2', 'test_efusion', 'test_f2f_nw', 'test_f2f_no_tools', 'test_f2f_tools', 'test_f2f_tools2'],
    help='Path to input folder.'
)
args = parser.parse_args()
METHODS = args.methods

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
runs_df = runs_df[runs_df.method.isin(METHODS)]
runs_df.method = runs_df.method.astype('category')
runs_df.method = runs_df.method.cat.set_categories(METHODS)
runs_df.sort_values(['method'], inplace=True)
runs_df.to_csv("project.csv")
runs_df['dataset'] = runs_df['keyframe']
runs_df["ATE/RMSE"] *= 1.0e3 #m to mm
runs_df["RPE/trans"] *= 1.0e3 #m to mm
runs_df["RPE/rot"] *= 180/np.pi # rad to deg

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
    print('macro average:', df.mean()['mean'],'+/-', df.std()['mean'])
    print('micro average:', runs_df[runs_df.method.eq(method)]['ATE/RMSE'].mean(), '+/-', runs_df[runs_df.method.eq(method)]['ATE/RMSE'].std())

print('\n------------')
print('RPE-rot in deg')
for method in METHODS:
    print('\n------------')
    print(method)
    df = runs_df[runs_df.method.eq(method)]
    print('average duration in frames:', df['frame'].mean(), '+/-', df['frame'].std())
    df = pd.DataFrame({'mean': df.groupby('dataset').mean()['RPE/rot'], 'std':df.groupby('dataset').std()['RPE/rot']})
    print(df)
    print('macro average:', df.mean()['mean'],'+/-', df.std()['mean'])
    print('micro average:', runs_df[runs_df.method.eq(method)]['RPE/rot'].mean(), '+/-', runs_df[runs_df.method.eq(method)]['RPE/rot'].std())

print('\n------------')
print('RPE-trans in mm')
for method in METHODS:
    print('\n------------')
    print(method)
    df = runs_df[runs_df.method.eq(method)]
    print('average duration in frames:', df['frame'].mean(), '+/-', df['frame'].std())
    df = pd.DataFrame(
        {'mean': df.groupby('dataset').mean()['RPE/trans'], 'std': df.groupby('dataset').std()['RPE/trans']})
    print(df)
    print('macro average:', df.mean()['mean'], '+/-', df.std()['mean'])
    print('micro average:', runs_df[runs_df.method.eq(method)]['RPE/trans'].mean(), '+/-',
          runs_df[runs_df.method.eq(method)]['RPE/trans'].std())
# Per Run info
print('\n------------')
print('ATE-RMSE in mm')
for run in runs_df.dataset.unique():

    df = runs_df[runs_df.dataset.eq(run)]
    for kf in df.keyframe.unique():
        print('\n------------')
        print(run, kf)
        df1 = df[df.keyframe.eq(kf)]
        print(df1[['method', 'ATE/RMSE']])

snb.violinplot(y='ATE/RMSE', x='dataset', hue='method', data=runs_df)
plt.show()