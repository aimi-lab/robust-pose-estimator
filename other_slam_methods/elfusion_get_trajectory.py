import sys
sys.path.append('../..')
from dataset.dataset_utils import get_data, StereoVideoDataset, SequentialSubSampler
from ElasticFusion import pyElasticFusion
import os, glob
import torch
import numpy as np
from tqdm import tqdm
from alley_oop.utils.trajectory import save_trajectory, read_freiburg
from dataset.preprocess.disparity.disparity_model import DisparityModel
from dataset.preprocess.segmentation_network.seg_model import SemanticSegmentationModel
import open3d
import warnings
import wandb
from torch.utils.data import DataLoader
from scipy.spatial.transform import Rotation as R
from scripts.evaluate_ate_freiburg import main as evaluate


def save_ply(pcl_array, path):
    pcl = open3d.geometry.PointCloud()
    pcl.points = open3d.utility.Vector3dVector(pcl_array[:, :3])
    pcl.colors = open3d.utility.Vector3dVector(pcl_array[:, 4:7])
    open3d.io.write_point_cloud(path, pcl)


def main(input_path, output_path, config, device_sel, start, stop, step, log, generate_map=True):
    device = torch.device('cpu')
    # if device_sel == 'gpu':
    #     if torch.cuda.is_available():
    #         device = torch.device('cuda')
    #     else:
    #         warnings.warn('No GPU available, fallback to CPU')

    if log is not None:
        config.update({'keyframe': os.path.split(input_path)[-1]})
        config.update({'dataset': os.path.split(input_path)[-2]})
        wandb.init(project='Alley-OOP', config=config, group=log)

    dataset, calib = get_data(input_path, config['img_size'])
    slam = pyElasticFusion(calib['intrinsics']['left'], config['img_size'][0], config['img_size'][1], 7.0, True, config['depth_scaling'])
    sampler = SequentialSubSampler(dataset, start, stop, step)
    loader = DataLoader(dataset, num_workers=0 if config['slam']['debug'] else 1, sampler=sampler)
    if isinstance(dataset, StereoVideoDataset):
        disp_model = DisparityModel(calibration=calib, device=device, depth_clipping=config['depth_clipping'])
        seg_model = SemanticSegmentationModel('stereo_slam/segmentation_network/trained/PvtB2_combined_TAM_fold1.pth',
                                              device)

    # check for ground-truth pose data for logging purposes
    gt_file = os.path.join(input_path, 'groundtruth.txt')
    gt_trajectory = read_freiburg(gt_file) if os.path.isfile(gt_file) else None
    trajectory = []
    last_pose = np.eye(4)
    for idx, data in enumerate(tqdm(loader, total=min(len(dataset), (stop-start)//step))):
        if isinstance(dataset, StereoVideoDataset):
            raise NotImplementedError
            limg, rimg, pose_kinematics, img_number = data
            depth, depth_valid = disp_model(limg, rimg)
            mask = seg_model.get_mask(limg)[0]
            mask &= depth_valid  # mask tools and non-valid depth
            diff_pose = np.linalg.pinv(last_pose)@pose_kinematics if config['slam']['kinematics'] != 'none' else np.eye(4)
            last_pose = pose_kinematics
        else:
            data = [data[i].numpy() for i in range(len(data)-1)] + [data[-1]]
            limg, depth, mask, rimg, disp, img_number = data
            diff_pose = np.eye(4)
            config['slam']['kinematics'] = 'fuse'

        if mask is None:
            mask = np.ones_like(depth)
        pose= slam.processFrame(limg, depth.astype(np.uint16),(mask == 0).astype(np.uint8) , idx, diff_pose, config['slam']['kinematics'] == 'fuse')
        trajectory.append({'camera-pose': pose.tolist(), 'timestamp': img_number[0], 'residual': 0.0, 'key_frame': True})
        if log:
            log_dict = {'frame': idx}
            if gt_trajectory is not None:
                if len(gt_trajectory) > idx:
                    pose = np.array(trajectory[-1]['camera-pose'])
                    tr_err = gt_trajectory[idx][:3, 3] - pose[:3, 3]
                    rot_err = (gt_trajectory[idx][:3, :3].T @ pose[:3, :3])
                    rot_err_deg = np.linalg.norm(R.from_matrix(rot_err).as_rotvec(degrees=True), ord=2)
                    log_dict.update({'error/x': tr_err[0],
                                     'error/y': tr_err[1],
                                     'error/z': tr_err[2],
                                     'error/rot': rot_err_deg})
            wandb.log(log_dict, step=idx)
    print('finished slam, save results...')
    os.makedirs(output_path, exist_ok=True)
    save_trajectory(trajectory, output_path)
    scene = slam.getPointCloud() if generate_map else None
    if scene is not None:
        save_ply(scene, os.path.join(output_path, 'map.ply'))

    if log is not None:
        wandb.save(os.path.join(output_path, 'trajectory.freiburg'))
        wandb.save(os.path.join(output_path, 'trajectory.json'))
        print('trajectories saved...')
        if scene is not None:
            wandb.save(os.path.join(output_path, 'map.ply'))
        if os.path.isfile(os.path.join(input_path, 'groundtruth.txt')):
            print('evaluated...')
            error = evaluate(os.path.join(input_path, 'groundtruth.txt'),
                             os.path.join(output_path, 'trajectory.freiburg'))
            print('save evaluation results...')
            wandb.define_metric('trans_error', step_metric='frame')
            for i, e in enumerate(error):
                wandb.log({'trans_error': e, 'frame': i})
            wandb.summary['ATE/RMSE'] = np.sqrt(np.dot(error, error) / len(error))
            wandb.summary['ATE/mean'] = np.mean(error)
    wandb.finish()
    print('finished')


if __name__ == '__main__':
    import argparse
    import yaml
    parser = argparse.ArgumentParser(description='script to run Elastic Fusion')

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
        default='configuration/efusion_scared.yaml',
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
