import sys
sys.path.append('../')
from alley_oop.slam import SLAM
import os
import torch
import numpy as np
from tqdm import tqdm
from dataset.preprocess.disparity.disparity_model import DisparityModel
from dataset.preprocess.segmentation_network.seg_model import SemanticSegmentationModel
import matplotlib.pyplot as plt
from alley_oop.fusion.surfel_map import SurfelMap
from alley_oop.utils.trajectory import save_trajectory, read_freiburg
from dataset.dataset_utils import get_data, StereoVideoDataset, SequentialSubSampler, RGBDDataset
from dataset.transforms import Compose
import warnings
from torch.utils.data import DataLoader
import wandb
from evaluation.evaluate_ate_freiburg import main as evaluate
from alley_oop.metrics.projected_photo_metrics import disparity_photo_loss


def main(input_path, output_path, config, device_sel, stop, start, step, log, force_video, checkpoint):
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

    dataset, calib = get_data(input_path, config['img_size'], force_video=force_video)
    slam = SLAM(torch.tensor(calib['intrinsics']['left']), config['slam'], img_shape=config['img_size'],
                checkpoint=checkpoint).to(device)
    if not isinstance(dataset, StereoVideoDataset):
        sampler = SequentialSubSampler(dataset, start, stop, step)
    else:
        warnings.warn('start/stop arguments not supported for video dataset. ignored.', UserWarning)
        sampler = None
    loader = DataLoader(dataset, num_workers=0 if config['slam']['debug'] else 1, pin_memory=True, sampler=sampler)

    if isinstance(dataset, StereoVideoDataset):
        seg_model = SemanticSegmentationModel('../dataset/preprocess/segmentation_network/trained/deepLabv3plus_trained_intuitive.pth',
                                              device, config['img_size'])
    if not isinstance(dataset, RGBDDataset):
        disp_model = DisparityModel(calibration=calib, device=device)

    # check for ground-truth pose data for logging purposes
    gt_file = os.path.join(input_path, 'groundtruth.txt')
    gt_trajectory = read_freiburg(gt_file) if os.path.isfile(gt_file) else None
    slam.recorder.set_gt(gt_trajectory)
    with torch.inference_mode():
        viewer = None
        if config['viewer']['enable']:
            from viewer.viewer3d import Viewer3D
            viewer = Viewer3D((2 * config['img_size'][0], 2 * config['img_size'][1]),
                              blocking=config['viewer']['blocking'])

        trajectory = []
        os.makedirs(output_path, exist_ok=True)
        for i, data in enumerate(tqdm(loader, total=min(len(dataset), (stop-start)//step))):
            if isinstance(dataset, StereoVideoDataset):
                limg, rimg, pose_kinematics, img_number = data
                mask, semantics = seg_model.get_mask(limg.to(device))
                disparity, depth, depth_noise = disp_model(limg.to(device), rimg.to(device))
            elif isinstance(dataset, RGBDDataset):
                limg, depth, depth_noise, mask, semantics, img_number = data
            else:
                limg, rimg, mask, semantics, img_number = data
                disparity, depth, depth_noise = disp_model(limg.to(device), rimg.to(device))
            limg, depth, mask, confidence, depth_noise = slam.pre_process(limg, depth, mask, semantics, depth_noise)
            pose, scene, pose_relscale = slam.processFrame(limg.to(device), depth.to(device), mask.to(device),
                                                           confidence.to(device))

            if viewer is not None:
                curr_pcl = SurfelMap(frame=slam.get_frame(), kmat=torch.tensor(calib['intrinsics']['left']).float(),
                                     pmat=pose_relscale, depth_scale=scene.depth_scale).pcl2open3d(stable=False)

                canonical_scene = scene.pcl2open3d(stable=config['viewer']['stable'])
                print('Current Frame vs Scene (Canonical/Deformed)')
                viewer(pose.cpu(), canonical_scene, add_pcd=curr_pcl,
                       frame=slam.get_frame(), synth_frame=slam.get_rendered_frame())

            trajectory.append({'camera-pose': pose.tolist(), 'timestamp': img_number[0], 'residual': 0.0, 'key_frame': True})
            if (log is not None) & (i > 0):
                slam.recorder.log(step=i)

        save_trajectory(trajectory, output_path)

        scene.save_ply(os.path.join(output_path, 'stable_map.ply'), stable=True)
        scene.save_ply(os.path.join(output_path, 'all_map.ply'), stable=False)
        if log is not None:
            wandb.save(os.path.join(output_path, 'trajectory.freiburg'))
            wandb.save(os.path.join(output_path, 'trajectory.json'))
            wandb.save(os.path.join(output_path, 'map.ply'))
            if os.path.isfile(os.path.join(input_path, 'groundtruth.txt')):
                error = evaluate(os.path.join(input_path, 'groundtruth.txt'),
                                 os.path.join(output_path, 'trajectory.freiburg'))
                wandb.define_metric('trans_error', step_metric='frame')
                for i, e in enumerate(error):
                    wandb.log({'trans_error': e, 'frame': i})
                wandb.summary['ATE/RMSE'] = np.sqrt(np.dot(error,error) / len(error))
                wandb.summary['ATE/mean'] = np.mean(error)

        print('finished')


if __name__ == '__main__':
    import argparse
    import yaml
    parser = argparse.ArgumentParser(description='script to run Raft Pose SLAM')

    parser.add_argument(
        'input',
        type=str,
        help='Path to input folder.'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
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
        '--force_video',
        action="store_true",
        help='force to use video input and recompute depth'
    )
    args = parser.parse_args()
    with open(args.config, 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    if args.outpath is None:
        args.outpath = os.path.join(args.input, 'data','alleyoop')
    assert os.path.isfile(args.checkpoint), 'no valid checkpoint file'

    main(args.input, args.outpath, config, args.device, args.stop, args.start, args.step, args.log, args.force_video, args.checkpoint)
