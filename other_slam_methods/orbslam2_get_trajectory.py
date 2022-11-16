import orbslam2
import os
from tqdm import tqdm
from alley_oop.utils.trajectory import save_trajectory, read_freiburg
from dataset.dataset_utils import get_data, StereoVideoDataset, SequentialSubSampler, TUMDataset, RGBDDataset
from dataset.preprocess.segmentation_network.seg_model import SemanticSegmentationModel
from alley_oop.fusion.surfel_map import SurfelMap, FrameClass
from scipy.spatial.transform import Rotation as R
from torch.utils.data import DataLoader
import warnings
import torch
import wandb
import numpy as np
from evaluation.evaluate_ate_freiburg import eval
from alley_oop.pose.pose_net import DepthNet


def tuple2list(listpose):
    assert len(listpose) == 12
    array = np.asarray(list(listpose) + [0,0,0,1]).reshape(4,4)
    return array.tolist()


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
    settings_path = os.path.join(input_path, 'slam_config_640x480.yaml')
    assert os.path.isfile(settings_path)
    slam = orbslam2.System(os.path.join('Vocabulary', 'ORBvoc.txt'), settings_path, orbslam2.Sensor.RGBD)
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

    slam.set_use_viewer(config['viewer']['enable'])
    slam.initialize()
    # check for ground-truth pose data for logging purposes
    gt_file = os.path.join(input_path, 'groundtruth.txt')
    gt_trajectory = read_freiburg(gt_file) if os.path.isfile(gt_file) else None
    depth_network = DepthNet()

    trajectory = []
    scene = None
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
            limg = (limg.squeeze().permute(1,2,0)*255.0).numpy().astype(np.uint8)
            slam.process_image_rgbd(limg, depth.squeeze().numpy(), mask.squeeze().numpy(), float(idx))

            is_key_frame = (slam.map_changed() | (idx == 0))
            if slam.get_tracking_state() == orbslam2.TrackingState.OK:
                trajectory.append(
                    {'camera-pose': tuple2list(slam.get_pose()), 'timestamp': img_number[0], 'residual': slam.get_residual_error(), 'key_frame': is_key_frame})
            else:
                trajectory.append(
                    {'camera-pose': trajectory[-1]['camera-pose'], 'timestamp': img_number[0], 'residual': slam.get_residual_error(),
                     'key_frame': is_key_frame})

            if log:
                log_dict = {'frame': int(img_number[0]),
                            'surfels/total': scene.opts.shape[1] if scene is not None else 0,
                            'surfels/stable': (scene.conf >= 1.0).sum().item() if scene is not None else 0}
                log_dict.update({f'pyr0/cost': slam.get_residual_error()})
                if gt_trajectory is not None:
                    if len(gt_trajectory) > int(img_number[0]):
                        pose = np.array(trajectory[-1]['camera-pose'])
                        tr_err = gt_trajectory[int(img_number[0])][:3, 3] - pose[:3, 3]
                        rot_err = (gt_trajectory[int(img_number[0])][:3, :3].T @ pose[:3, :3])
                        rot_err_deg = np.linalg.norm(R.from_matrix(rot_err).as_rotvec(degrees=True), ord=2)
                        log_dict.update({'error/x': tr_err[0],
                                         'error/y': tr_err[1],
                                         'error/z': tr_err[2],
                                         'error/rot': rot_err_deg})
                wandb.log(log_dict, step=int(img_number[0]))

    os.makedirs(outpath, exist_ok=True)
    save_trajectory(trajectory, outpath)
    slam.shutdown()

    if scene is not None:
        scene.save_ply(os.path.join(outpath, 'map.ply'), stable=True)
    if log is not None:
        wandb.save(os.path.join(outpath, 'trajectory.freiburg'))
        wandb.save(os.path.join(outpath, 'trajectory.json'))
        if scene is not None:
            wandb.save(os.path.join(outpath, 'map.ply'))
        if os.path.isfile(os.path.join(input_path, 'groundtruth.txt')):
            ate_rmse, rpe_trans, rpe_rot, trans_error, *_ = eval(os.path.join(input_path, 'groundtruth.txt'),
                                                             os.path.join(outpath, 'trajectory.freiburg'), offset=-4)
            wandb.define_metric('trans_error', step_metric='frame')
            for i, e in enumerate(trans_error):
                wandb.log({'trans_error': e, 'frame': i})
            wandb.summary['ATE/RMSE'] = ate_rmse
            wandb.summary['RPE/trans'] = rpe_trans
            wandb.summary['RPE/rot'] = rpe_rot
    wandb.finish()
    print('finished')


if __name__ == '__main__':
    import argparse
    import yaml
    parser = argparse.ArgumentParser(description='script to run ORBSLAM 2')

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
        default='configuration/orbslam2.yaml',
        help='Path to config file.'
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
        args.outpath = os.path.join(args.input, 'data','orbslam')

    main(args.input, args.outpath, config, args.device, args.start, args.stop, args.step, args.log, args.generate_map is not None)
