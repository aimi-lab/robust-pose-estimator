import sys
sys.path.append('../')
from alley_oop.slam import SLAM
import os
import torch
import numpy as np
from tqdm import tqdm
from viewer.viewer3d import Viewer3D
from other_slam_methods.stereo_slam.disparity.disparity_model import DisparityModel
from other_slam_methods.stereo_slam.segmentation_network.seg_model import SemanticSegmentationModel
import open3d
import matplotlib.pyplot as plt
from alley_oop.fusion.surfel_map import SurfelMap
from alley_oop.utils.trajectory import save_trajectory
from dataset.dataset_utils import get_data, StereoVideoDataset
from dataset.transforms import Compose
import warnings
from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity

def main(input_path, output_path, config, device_sel, nsamples):
    device = torch.device('cpu')
    if device_sel == 'gpu':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            warnings.warn('No GPU available, fallback to CPU')

    dataset, calib = get_data(input_path, config['img_size'])
    slam = SLAM(torch.tensor(calib['intrinsics']['left']), config['slam'], img_shape=config['img_size']).to(device)
    dataset.transform = Compose([dataset.transform, slam.pre_process])  # add pre-processing to data loading (CPU)
    loader = DataLoader(dataset, num_workers=1, pin_memory=True)


    with torch.inference_mode():
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for i, data in enumerate(tqdm(loader, total=len(dataset))):
                limg, depth, mask, img_number = data
                with record_function("model_inference"):
                    pose, scene = slam.processFrame(limg.to(device), depth.to(device), mask.to(device))
                if i > 2:
                    break
    prof.export_chrome_trace("trace.json")

    print('finished')


if __name__ == '__main__':
    import argparse
    import yaml
    parser = argparse.ArgumentParser(description='script to run EMDQ SLAM re-implementation')

    parser.add_argument(
        'input',
        type=str,
        help='Path to input folder.'
    )
    parser.add_argument(
        '--outpath',
        type=str,
        help='Path to output folder. If not provided use input path instead.'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='../configuration/alleyoop_slam.yaml',
        help='Configuration file.'
    )
    parser.add_argument(
        '--device',
        choices=['cpu', 'gpu'],
        default='cpu',
        help='select cpu or gpu to run slam.'
    )
    parser.add_argument(
        '--nsamples',
        type=int,
        default=10000000000,
        help='force use of CPU.'
    )
    args = parser.parse_args()
    with open(args.config, 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    if args.outpath is None:
        args.outpath = os.path.join(args.input, 'data','alleyoop')

    main(args.input, args.outpath, config, args.device, args.nsamples)
