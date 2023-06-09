import os
import argparse
import yaml
import pandas as pd
import sys

sys.path.append('../')
from scripts.infer_trajectory import main as infer_trajectory

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='script to benchmark pose estimation on test set')

    parser.add_argument(
        'input',
        type=str,
        help='Path to input folder.'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='../trained/poseNet_2xf8up4b.pth',
        help='Path to trained Pose Estimator Checkpoint.'
    )
    parser.add_argument(
        '--outpath',
        type=str,
        help='Path to output folder. If not provided use input path instead.'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='../configuration/infer_f2f.yaml',
        help='Configuration file.'
    )
    parser.add_argument(
        '--device',
        choices=['cpu', 'gpu'],
        default='gpu',
        help='select cpu or gpu to run slam.'
    )
    parser.add_argument(
        '--step',
        type=int,
        default=1,
        help='sub sampling interval.'
    )
    parser.add_argument(
        '--log',
        default=None,
        help='wandb group logging name. No logging if none set'
    )
    parser.add_argument(
        '--force_video',
        action="store_true",
        help='force to use video input and recompute depth'
    )
    parser.add_argument(
        '--viewer',
        default='none',
        choices=['none', '2d', '3d'],
        help='select viewer'
    )
    parser.add_argument(
        '--block_viewer',
        action="store_true",
        help='block viewer if viewer selected.'
    )
    args = parser.parse_args()
    with open(args.config, 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    assert os.path.isfile(args.checkpoint), 'no valid checkpoint file'

    assert os.path.isfile(os.path.join(args.input, 'test_split.csv'))
    df = pd.read_csv(os.path.join(args.input, 'test_split.csv'))
    for i, row in df.iterrows():
        args.start = row['start']
        args.stop = row['end']
        config.update({'start': args.start, 'end': args.stop})
        print(f'{args.start} -> {args.stop}')
        # read scenario file
        infer_trajectory(args, config)
