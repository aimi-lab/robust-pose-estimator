import sys
sys.path.append('../')
import os
from tqdm import tqdm
import wandb
import torch
from torch.utils.data import DataLoader
import warnings

from core.fusion.surfel_map import SurfelMap
from core.utils.trajectory import save_trajectory, read_freiburg
from core.pose.pose_estimator import PoseEstimator
from core.utils.logging import InferenceLogger

from dataset.preprocess.segmentation_network.seg_model import SemanticSegmentationModel
from dataset.dataset_utils import get_data, StereoVideoDataset, SequentialSubSampler
from evaluation.evaluate_ate_freiburg import eval
from viewer.viewer3d import Viewer3D
from viewer.viewer2d import Viewer2D
from viewer.view_renderer import ViewRenderer


def main(args, config):
    device = torch.device('cpu')
    if args.device == 'gpu':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            warnings.warn('No GPU available, fallback to CPU')

    if args.log is not None:
        config.update({'keyframe': os.path.split(args.input)[-1]})
        config.update({'dataset': os.path.split(args.input)[-2]})
        wandb.init(project='Alley-OOP', config=config, group=args.log)
        if args.outpath is None:
            args.outpath = wandb.run.dir
    if args.outpath is None:
        try:
            args.outpath = os.path.join(args.input, 'data',f'{config["seq_number"]}', 'infer_trajectory')
        except KeyError:
            args.outpath = os.path.join(args.input, 'data', 'infer_trajectory')
    os.makedirs(args.outpath, exist_ok=True)

    dataset, calib = get_data(args.input, config['img_size'], force_stereo=True, rect_mode=config['rect_mode'])
    # check for ground-truth pose data for logging purposes
    gt_file = os.path.join(args.input, 'groundtruth.txt')
    gt_trajectory = read_freiburg(gt_file) if os.path.isfile(gt_file) else None
    init_pose = torch.tensor(gt_trajectory[args.start]) if gt_trajectory is not None else torch.eye(4)

    pose_estimator = PoseEstimator(config['slam'], torch.tensor(calib['intrinsics']['left']).to(device), baseline=calib['bf'],
                                    checkpoint=args.checkpoint, img_shape=config['img_size'], init_pose=init_pose).to(device)
    if not isinstance(dataset, StereoVideoDataset):
        sampler = SequentialSubSampler(dataset, args.start, args.stop, args.step)
    else:
        warnings.warn('start/stop arguments not supported for video dataset. ignored.', UserWarning)
        sampler = None
    loader = DataLoader(dataset, num_workers=0 if config['slam']['debug'] else 1, pin_memory=True, sampler=sampler)

    if isinstance(dataset, StereoVideoDataset):
        seg_model = SemanticSegmentationModel('../dataset/preprocess/segmentation_network/trained/deepLabv3plus_trained_intuitive.pth',
                                              device, config['img_size'])

    recorder = InferenceLogger()
    recorder.set_gt(gt_trajectory)
    with torch.no_grad():
        viewer = None
        if args.viewer == '3d':
            viewer = Viewer3D((2 * config['img_size'][0], 2 * config['img_size'][1]),blocking=args.block_viewer)
        elif args.viewer == '2d':
            viewer = Viewer2D(outpath=args.outpath, blocking=args.block_viewer)
        elif args.viewer == 'video':
            viewer = ViewRenderer((2*config['img_size'][1], 2*config['img_size'][0]), outpath=args.outpath)

        trajectory = [{'camera-pose': init_pose.tolist(), 'timestamp': args.start, 'residual': 0.0, 'key_frame': True}]
        for i, data in enumerate(tqdm(loader, total=min(len(dataset), (args.stop-args.start)//args.step))):
            if isinstance(dataset, StereoVideoDataset):
                limg, rimg, mask, pose_kinematics, img_number = data
                tool_mask, semantics = seg_model.get_mask(limg.to(device))
                mask &= tool_mask
            else:
                limg, rimg, mask, semantics, img_number = data

            pose, scene, flow = pose_estimator(limg.to(device), rimg.to(device), mask.to(device))

            # visualization
            if isinstance(viewer, Viewer3D) & (i > 0):
                curr_pcl = SurfelMap(frame=pose_estimator.get_frame(), kmat=torch.tensor(calib['intrinsics']['left']).float(),
                                     pmat=pose).pcl2open3d(stable=False)
                curr_pcl.paint_uniform_color([0.5, 0.5, 1.0])
                canonical_scene = scene.pcl2open3d(stable=False)
                viewer(pose.cpu(), canonical_scene, add_pcd=curr_pcl)
            elif isinstance(viewer, Viewer2D) & (i > 0):
                viewer(pose_estimator.get_frame(), pose_estimator.get_last_frame(), flow, i*args.step)
            elif isinstance(viewer, ViewRenderer) & (i > 0):
                canonical_scene = scene.pcl2open3d(stable=True)
                viewer(pose.cpu(), canonical_scene)
            trajectory.append({'camera-pose': pose.tolist(), 'timestamp': img_number[0], 'residual': 0.0, 'key_frame': True})

            # logging
            if (args.log is not None) & (i > 0):
                recorder(scene, pose, step=int(img_number[0]))

        save_trajectory(trajectory, args.outpath)
        if scene is not None:
            scene.save_ply(os.path.join(args.outpath, 'stable_map.ply'), stable=True)
            scene.save_ply(os.path.join(args.outpath, 'all_map.ply'), stable=False)
        if args.log is not None:
            wandb.save(os.path.join(args.outpath, 'trajectory.freiburg'))
            wandb.save(os.path.join(args.outpath, 'map.ply'))
            if os.path.isfile(os.path.join(args.input, 'groundtruth.txt')):
                ate_rmse, rpe_trans, rpe_rot, trans_error, rpe_trans_e, rpe_rot_e = eval(os.path.join(args.input, 'groundtruth.txt'),
                                 os.path.join(args.outpath, 'trajectory.freiburg'), offset=-4)
                wandb.define_metric('trans_error', step_metric='frame')
                wandb.define_metric('rpe_trans_e', step_metric='frame')
                wandb.define_metric('rpe_rot_e', step_metric='frame')
                for i, (e1,e2,e3) in enumerate(zip(trans_error,rpe_trans_e, rpe_rot_e)):
                    wandb.log({'trans_error': e1,'rpe_trans_e': e2,'rpe_rot_e': e3 , 'frame': i})
                wandb.summary['ATE/RMSE'] = ate_rmse
                wandb.summary['RPE/trans'] = rpe_trans
                wandb.summary['RPE/rot'] = rpe_rot

        print('finished')


if __name__ == '__main__':
    import argparse
    import yaml
    parser = argparse.ArgumentParser(description='script to run pose estmation')

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
        default='gpu',
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
    parser.add_argument(
        '--viewer',
        default='none',
        choices=['none', '2d', '3d', 'video'],
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

    main(args, config)
