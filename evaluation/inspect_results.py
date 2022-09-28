import pandas as pd
import wandb
import seaborn as snb
import matplotlib.pyplot as plt
import os

api = wandb.Api()

METHODS = ['orbslam2_raftdepth', 'scared_raftslam_f2f', 'scared_raftslam_f2f_w', 'scared_raftslam_f2m', 'scared_raftslam_f2m_w']

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
runs_df['dataset'] = [os.path.basename(d) for d in runs_df['dataset']]

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