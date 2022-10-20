import sys
sys.path.append('../..')
from ElasticFusion import pyElasticFusion
import os
from tqdm import tqdm
from alley_oop.utils.trajectory import save_trajectory, read_freiburg
from dataset.dataset_utils import get_data, StereoVideoDataset, SequentialSubSampler, TUMDataset, RGBDDataset
from dataset.preprocess.segmentation_network.seg_model import SemanticSegmentationModel
from scipy.spatial.transform import Rotation as R
from torch.utils.data import DataLoader
import warnings
import torch
import wandb
import numpy as np
from evaluation.evaluate_ate_freiburg import main as evaluate
from alley_oop.pose.PoseN import DepthNet


def main(input_path, outpath, config, device_sel, start, stop, step, log, generate_map):
    device = torch.device('cpu')
    if device_sel == 'gpu':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            warnings.warn('No GPU available, fallback to CPU')

    if log is not None:
        config.update({'keyframe': os.path.split(input_path)[-1]})
        config.update({'dataset': os.path.split(input_path)[-2]})
        wandb.init(project='Alley-OOP', config=config, group=log)

    dataset, calib = get_data(input_path, config['img_size'], rect_mode=config['rect_mode'])
    baseline = torch.tensor(calib['bf']).float().unsqueeze(0)
    slam = pyElasticFusion(calib['intrinsics']['left'], config['img_size'][0], config['img_size'][1], 7.0, True, config['depth_scaling'])

    if not isinstance(dataset, StereoVideoDataset):
        sampler = SequentialSubSampler(dataset, start, stop, step)
    else:
        warnings.warn('start/stop arguments not supported for video dataset. ignored.', UserWarning)
        sampler = None
    loader = DataLoader(dataset, num_workers=0 if config['slam']['debug'] else 1, pin_memory=False, sampler=sampler)
    if isinstance(dataset, StereoVideoDataset):
        seg_model = SemanticSegmentationModel(
            '../dataset/preprocess/segmentation_network/trained/deepLabv3plus_trained_intuitive.pth',
            device, config['img_size'])

    # check for ground-truth pose data for logging purposes
    gt_file = os.path.join(input_path, 'groundtruth.txt')
    gt_trajectory = read_freiburg(gt_file) if os.path.isfile(gt_file) else None
    depth_network = DepthNet()

    trajectory = []
    with torch.inference_mode():
        for idx, data in enumerate(tqdm(loader, total=min(len(dataset), (stop-start)//step))):
            if isinstance(dataset, StereoVideoDataset):
                limg, rimg, pose_kinematics, img_number = data
                mask, semantics = seg_model.get_mask(limg.to(device))
                depth, _ = depth_network(255.0*limg.to(device), 255.0*rimg.to(device), baseline)
            elif isinstance(dataset, RGBDDataset) | isinstance(dataset, TUMDataset):
                limg, depth, mask, semantics, img_number = data
            else:
                limg, rimg, mask, semantics, img_number = data
                depth, _ = depth_network(255.0*limg.to(device), 255.0*rimg.to(device), baseline)
            limg = (255.0*limg.squeeze().permute(1,2,0)).numpy().astype(np.uint8)
            depth = depth.squeeze().numpy().astype(np.uint16)
            mask = (mask == 0).squeeze().numpy().astype(np.uint8)
            pose= slam.processFrame(limg, depth,mask , idx, np.eye(4), True)

            trajectory.append({'camera-pose': pose.tolist(), 'timestamp': img_number[0], 'residual': 0.0, 'key_frame': True})
            if log:
                log_dict = {'frame': idx*step}
                if gt_trajectory is not None:
                    if len(gt_trajectory) > idx*step:
                        pose = np.array(trajectory[-1]['camera-pose'])
                        tr_err = gt_trajectory[idx*step][:3, 3] - pose[:3, 3]
                        rot_err = (gt_trajectory[idx*step][:3, :3].T @ pose[:3, :3])
                        rot_err_deg = np.linalg.norm(R.from_matrix(rot_err).as_rotvec(degrees=True), ord=2)
                        log_dict.update({'error/x': tr_err[0],
                                         'error/y': tr_err[1],
                                         'error/z': tr_err[2],
                                         'error/rot': rot_err_deg})
                wandb.log(log_dict, step=idx*step)

    os.makedirs(outpath, exist_ok=True)
    save_trajectory(trajectory, outpath)
    if log is not None:
        wandb.save(os.path.join(outpath, 'trajectory.freiburg'))
        wandb.save(os.path.join(outpath, 'trajectory.json'))
        if os.path.isfile(os.path.join(input_path, 'groundtruth.txt')):
            error = evaluate(os.path.join(input_path, 'groundtruth.txt'),
                             os.path.join(outpath, 'trajectory.freiburg'))
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
    parser.add_argument(
        '--generate_map',
        action = 'store_true',
        help='set to generate dense surfel map'
    )
    args = parser.parse_args()
    with open(args.config, 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    if args.outpath is None:
        args.outpath = os.path.join(args.input, 'data','efusion')

    main(args.input, args.outpath, config, args.device, args.start, args.stop, args.step, args.log, args.generate_map is not None)
