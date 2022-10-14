import sys
sys.path.append('../')
from alley_oop.slam import SLAM
import os
from tqdm import tqdm
from dataset.preprocess.segmentation_network.seg_model import SemanticSegmentationModel
from alley_oop.fusion.surfel_map_deformable import *
from alley_oop.utils.trajectory import save_trajectory, read_freiburg
from dataset.dataset_utils import get_data, StereoVideoDataset, SequentialSubSampler, RGBDDataset, TUMDataset
import warnings
from torch.utils.data import DataLoader
import wandb
from evaluation.evaluate_ate_freiburg import main as evaluate
from viewer.viewer3d import Viewer3D
from viewer.viewer2d import Viewer2D


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

    dataset, calib = get_data(args.input, config['img_size'], force_video=args.force_video, rect_mode=config['rect_mode'])
    # check for ground-truth pose data for logging purposes
    gt_file = os.path.join(args.input, 'groundtruth.txt')
    gt_trajectory = read_freiburg(gt_file) if os.path.isfile(gt_file) else None

    slam = SLAM(torch.tensor(calib['intrinsics']['left']).to(device), config['slam'], img_shape=config['img_size'], baseline=calib['bf'],
                checkpoint=args.checkpoint, init_pose=torch.tensor(gt_trajectory[0]) if gt_trajectory is not None else torch.eye(4)).to(device)
    if not isinstance(dataset, StereoVideoDataset):
        sampler = SequentialSubSampler(dataset, args.start, args.stop, args.step)
    else:
        warnings.warn('start/stop arguments not supported for video dataset. ignored.', UserWarning)
        sampler = None
    loader = DataLoader(dataset, num_workers=0 if config['slam']['debug'] else 1, pin_memory=True, sampler=sampler)

    if isinstance(dataset, StereoVideoDataset):
        seg_model = SemanticSegmentationModel('../dataset/preprocess/segmentation_network/trained/deepLabv3plus_trained_intuitive.pth',
                                              device, config['img_size'])

    slam.recorder.set_gt(gt_trajectory)
    with torch.no_grad():
        viewer = None
        if args.viewer == '3d':
            viewer = Viewer3D((2 * config['img_size'][0], 2 * config['img_size'][1]),blocking=args.block_viewer)
        elif args.viewer == '2d':
            viewer = Viewer2D(outpath=args.outpath, blocking=args.block_viewer)

        trajectory = []
        os.makedirs(args.outpath, exist_ok=True)
        for i, data in enumerate(tqdm(loader, total=min(len(dataset), (args.stop-args.start)//args.step))):
            if isinstance(dataset, StereoVideoDataset):
                limg, rimg, pose_kinematics, img_number = data
                mask, semantics = seg_model.get_mask(limg.to(device))
                depth, flow, _ = slam.pose_estimator.estimate_depth(limg.to(device), rimg.to(device))
            elif isinstance(dataset, RGBDDataset) | isinstance(dataset, TUMDataset):
                raise NotImplementedError
            else:
                limg, rimg, mask, semantics, img_number = data
                depth, flow, _ = slam.pose_estimator.estimate_depth(limg.to(device), rimg.to(device))
            limg,rimg, depth, mask = slam.pre_process(limg, rimg, depth, mask, semantics)
            pose, scene, pose_relscale = slam.processFrame(limg.to(device), rimg.to(device), depth.to(device), mask.to(device), flow.to(device))

            if isinstance(viewer, Viewer3D):
                curr_pcl = SurfelMap(frame=slam.get_frame(), kmat=torch.tensor(calib['intrinsics']['left']).float(),
                                     pmat=pose_relscale, depth_scale=scene.depth_scale).pcl2open3d(stable=False)
                curr_pcl.paint_uniform_color([0.5, 0.5, 1.0])
                canonical_scene = scene.pcl2open3d(stable=config['viewer']['stable'])
                deformed_scene = scene.deform_cpy().pcl2open3d(stable=config['viewer']['stable']) if isinstance(scene, SurfelMapDeformable) else None
                print('Current Frame vs Scene (Canonical/Deformed)')
                viewer(pose.cpu(), canonical_scene, add_pcd=curr_pcl,
                       frame=slam.get_frame(), synth_frame=slam.get_rendered_frame(),
                       def_pcd=deformed_scene)
            elif isinstance(viewer, Viewer2D):
                viewer(slam.get_frame(), slam.get_rendered_frame(), i*args.step)
            trajectory.append({'camera-pose': pose.tolist(), 'timestamp': img_number[0], 'residual': 0.0, 'key_frame': True})
            if (args.log is not None) & (i > 0):
                slam.recorder.log(step=i*args.step)
            if args.store_map & ((i%50) == 49):
                scene.save_ply(os.path.join(args.outpath, f'stable_map_{i}.ply'), stable=True)
                scene.deform_cpy().save_ply(os.path.join(args.outpath, f'def_map_{i}.ply'), stable=True)

        save_trajectory(trajectory, args.outpath)

        scene.save_ply(os.path.join(args.outpath, 'stable_map.ply'), stable=True)
        scene.save_ply(os.path.join(args.outpath, 'all_map.ply'), stable=False)
        if args.log is not None:
            wandb.save(os.path.join(args.outpath, 'trajectory.freiburg'))
            wandb.save(os.path.join(args.outpath, 'trajectory.json'))
            wandb.save(os.path.join(args.outpath, 'map.ply'))
            if os.path.isfile(os.path.join(args.input, 'groundtruth.txt')):
                error = evaluate(os.path.join(args.input, 'groundtruth.txt'),
                                 os.path.join(args.outpath, 'trajectory.freiburg'))
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
    parser.add_argument(
        '--store_map',
        action="store_true",
        help='store intermediate maps'
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
    if args.outpath is None:
        args.outpath = os.path.join(args.input, 'data','alleyoop')
    assert os.path.isfile(args.checkpoint), 'no valid checkpoint file'

    main(args, config)
