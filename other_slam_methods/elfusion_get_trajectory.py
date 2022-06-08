import sys
sys.path.append('../..')
from dataset.dataset_utils import get_data, StereoVideoDataset, SequentialSubSampler
from ElasticFusion import pyElasticFusion
import os, glob
import torch
import numpy as np
from tqdm import tqdm
from alley_oop.utils.trajectory import save_trajectory
from dataset.preprocess.disparity.disparity_model import DisparityModel
from dataset.preprocess.segmentation_network.seg_model import SemanticSegmentationModel
import open3d
import warnings
import wandb
from torch.utils.data import DataLoader


def save_ply(pcl_array, path):
    pcl = open3d.geometry.PointCloud()
    pcl.points = open3d.utility.Vector3dVector(pcl_array[:, :3])
    pcl.colors = open3d.utility.Vector3dVector(pcl_array[:, 4:7])
    open3d.io.write_point_cloud(path, pcl)


def main(input_path, output_path, config, device_sel, start, stop, step, log):
    device = torch.device('cpu')
    if device_sel == 'gpu':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            warnings.warn('No GPU available, fallback to CPU')

    if log is not None:
        config.update({'data': os.path.split(input_path)[-1]})
        wandb.init(project='Alley-OOP', config=config, group=log)

    dataset, calib = get_data(input_path, config['img_size'])
    slam = pyElasticFusion(calib['intrinsics']['left'], config['img_size'][0], config['img_size'][1], 7.0, True, config['depth_scaling'])
    sampler = SequentialSubSampler(dataset, start, stop, step)
    loader = DataLoader(dataset, num_workers=0 if config['slam']['debug'] else 1, pin_memory=True, sampler=sampler)
    if isinstance(dataset, StereoVideoDataset):
        disp_model = DisparityModel(calibration=calib, device=device, depth_clipping=config['depth_clipping'])
        seg_model = SemanticSegmentationModel('stereo_slam/segmentation_network/trained/PvtB2_combined_TAM_fold1.pth',
                                              device)


    trajectory = []
    last_pose = np.eye(4)
    for i, data in enumerate(tqdm(loader, total=min(len(dataset), (stop-start)//step))):
        if isinstance(dataset, StereoVideoDataset):
            raise NotImplementedError
            limg, rimg, pose_kinematics, img_number = data
            depth, depth_valid = disp_model(limg, rimg)
            mask = seg_model.get_mask(limg)[0]
            mask &= depth_valid  # mask tools and non-valid depth
            diff_pose = np.linalg.pinv(last_pose)@pose_kinematics if config['slam']['kinematics'] != 'none' else np.eye(4)
            last_pose = pose_kinematics
        else:
            limg, depth, mask, rimg, disp, img_number = data
            diff_pose = np.eye(4)
            config['slam']['kinematics'] = 'fuse'

        if mask is None:
            mask = np.ones_like(depth)
        pose= slam.processFrame(limg, depth.astype(np.uint16),(mask == 0).astype(np.uint8) , i, diff_pose, config['slam']['kinematics'] == 'fuse')
        trajectory.append({'camera-pose': pose.tolist(), 'timestamp': img_number, 'residual': 0.0, 'key_frame': True})

    os.makedirs(output_path, exist_ok=True)
    save_trajectory(trajectory, output_path)
    pcl = slam.getPointCloud()
    if pcl is not None:
        save_ply(pcl, os.path.join(output_path, 'map.ply'))
        print(pcl.shape)
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
        default='configuration/efusion_tum.yaml',
        help='Configuration file.'
    )
    parser.add_argument(
        '--device',
        choices=['cpu', 'gpu'],
        default='cpu',
        help='select cpu or gpu to run slam.'
    )
    parser.add_argument(
        '--stop',
        type=int,
        default=10000000000,
        help='number of samples to run for.'
    )
    parser.add_argument(
        '--start',
        type=int,
        default=0,
        help='at which sample to start slam.'
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
    args = parser.parse_args()
    with open(args.config, 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    if args.outpath is None:
        args.outpath = os.path.join(args.input, 'data','efusion')

    main(args.input, args.outpath, config, args.device, args.start, args.stop, args.step, args.log)
