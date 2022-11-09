import pandas as pd
import wandb
import seaborn as snb
import matplotlib.pyplot as plt
import os

api = wandb.Api()
import argparse

parser = argparse.ArgumentParser(description='Inspect WandB results')

parser.add_argument(
    '--methods',
    nargs='+',
    type=str,
    default=['scenario_orbslam2', 'scenarios_f2f_nw', 'scenarios_f2f_no_tools', 'scenarios_f2f_tools', 'scenarios_f2f_tools2', 'scenario_efusion'],
    help='methods to inspect.'
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
    if run._attrs['group'] in ['scenario_orbslam2', 'scenario_efusion']:
        if (int(run.createdAt[5:7])*100+int(run.createdAt[8:10])) < (100*10+25):
            try:
                all_dict["ATE/RMSE"] *= 1000.0 #m to mm
                all_dict["RPE/trans"] *= 1000.0  # m to mm
            except KeyError:
                pass
    else:
        try:
            all_dict["ATE/RMSE"] *= 1000.0  # m to mm
            all_dict["RPE/trans"] *= 1000.0  # m to mm
        except KeyError:
            pass
    summary_list.append(all_dict)

runs_df = pd.DataFrame(summary_list)
runs_df = runs_df[runs_df.method.isin(METHODS)]
runs_df = runs_df[runs_df['scenario'] != 'A']
runs_df.replace('F', 'C', inplace=True)
runs_df.replace('G', 'B', inplace=True)
runs_df.method = runs_df.method.astype('category')
runs_df.method = runs_df.method.cat.set_categories(METHODS)
runs_df.sort_values(['method'], inplace=True)

runs_df['dataset'] = [os.path.basename(d) for d in runs_df['dataset']]


# Group into methods and datasets
print('\n------------')
print('ATE-RMSE in mm')
for method in METHODS:
    print('\n------------')
    print(method)
    df = runs_df[runs_df.method.eq(method)]
    print('samples:', len(df))
    df = pd.DataFrame({'mean': df.groupby('scenario').mean()['ATE/RMSE'], 'std':df.groupby('scenario').std()['ATE/RMSE']})
    print(df)
    print('macro average:', df.mean()['mean'],'+/-', df.std()['mean'])
    print('micro average:', runs_df[runs_df.method.eq(method)]['ATE/RMSE'].mean(), '+/-', runs_df[runs_df.method.eq(method)]['ATE/RMSE'].std())
print('\n------------')
print('RPE-trans in mm')
for method in METHODS:
    print('\n------------')
    print(method)
    df = runs_df[runs_df.method.eq(method)]
    df = pd.DataFrame(
        {'mean': df.groupby('scenario').mean()['RPE/trans'], 'std': df.groupby('scenario').std()['RPE/trans']})
    print(df)
    print('macro average:', df.mean()['mean'], '+/-', df.std()['mean'])
    print('micro average:', runs_df[runs_df.method.eq(method)]['RPE/trans'].mean(), '+/-',
          runs_df[runs_df.method.eq(method)]['RPE/trans'].std())
# print('\n------------')
# print('RPE-rot in deg')
# for method in METHODS:
#     print('\n------------')
#     print(method)
#     df = runs_df[runs_df.method.eq(method)]
#     df = pd.DataFrame({'mean': df.groupby('scenario').mean()['RPE/rot'], 'std':df.groupby('scenario').std()['RPE/rot']})
#     print(df)
#     print('macro average:', df.mean()['mean'],'+/-', df.std()['mean'])
# # # Per Run info
# print('\n------------')
# print('ATE-RMSE in mm')
# # for run in runs_df.dataset.unique():
# #
# #     df = runs_df[runs_df.dataset.eq(run)]
# #     for kf in df.keyframe.unique():
# #         print('\n------------')
# #         print(run, kf)
# #         df1 = df[df.keyframe.eq(kf)]
# #         print(df1[['method', 'ATE/RMSE']])
# #
# # snb.violinplot(y='ATE/RMSE', x='scenario', hue='method', data=runs_df)
# # plt.show()