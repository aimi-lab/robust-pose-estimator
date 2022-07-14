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
    summary_list.append(all_dict)

runs_df = pd.DataFrame(summary_list)
runs_df['ATE/RMSE'] *= 1000.0

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