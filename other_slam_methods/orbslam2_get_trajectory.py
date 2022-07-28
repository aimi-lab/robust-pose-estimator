import orbslam2
import os
from tqdm import tqdm
from alley_oop.utils.trajectory import save_trajectory, read_freiburg
from dataset.dataset_utils import get_data, StereoVideoDataset, SequentialSubSampler
from dataset.preprocess.disparity.disparity_model import DisparityModel
from dataset.preprocess.segmentation_network.seg_model import SemanticSegmentationModel
from alley_oop.fusion.surfel_map import SurfelMap, FrameClass
from scipy.spatial.transform import Rotation as R
from torch.utils.data import DataLoader
import warnings
import torch
import wandb
import numpy as np
from evaluation.evaluate_ate_freiburg import main as evaluate


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

    dataset, calib = get_data(input_path, config['img_size'])
    settings_path = os.path.join(input_path, 'slam_config_640x480.yaml')
    assert os.path.isfile(settings_path)
    slam = orbslam2.System(os.path.join('Vocabulary', 'ORBvoc.txt'), settings_path, orbslam2.Sensor.RGBD)
    sampler = SequentialSubSampler(dataset, start, stop, step)
    loader = DataLoader(dataset, num_workers=0 if config['slam']['debug'] else 1, pin_memory=False, sampler=sampler)
    disp_model = DisparityModel(calibration=calib, device=device)
    if isinstance(dataset, StereoVideoDataset):
        seg_model = SemanticSegmentationModel('stereo_slam/segmentation_network/trained/PvtB2_combined_TAM_fold1.pth',
                                              device, config['img_size'])

    slam.set_use_viewer(config['viewer']['enable'])
    slam.initialize()

    # check for ground-truth pose data for logging purposes
    gt_file = os.path.join(input_path, 'groundtruth.txt')
    gt_trajectory = read_freiburg(gt_file) if os.path.isfile(gt_file) else None

    trajectory = []
    scene = None
    for idx, data in enumerate(tqdm(loader, total=min(len(dataset), (stop-start)//step))):
        if isinstance(dataset, StereoVideoDataset):
            limg, rimg, pose_kinematics, img_number = data
            mask, semantics = seg_model.get_mask(limg)
        else:
            limg, depth, mask, rimg, disparity, semantics, img_number = data
        disparity, depth, depth_noise = disp_model(limg, rimg)
        limg, depth, mask, *_  = slam.pre_process(limg, depth, mask, semantics, depth_noise)

        slam.process_image_rgbd(limg.squeeze().numpy(), depth.squeeze().numpy(), mask.squeeze().numpy(), float(idx))

        is_key_frame = (slam.map_changed() | (idx == 0))
        if slam.get_tracking_state() == orbslam2.TrackingState.OK:
            trajectory.append(
                {'camera-pose': tuple2list(slam.get_pose()), 'timestamp': img_number[0], 'residual': slam.get_residual_error(), 'key_frame': is_key_frame})
        else:
            trajectory.append(
                {'camera-pose': trajectory[-1]['camera-pose'], 'timestamp': img_number[0], 'residual': slam.get_residual_error(),
                 'key_frame': is_key_frame})

        if generate_map:
            frame = FrameClass(limg.permute(0,3,1,2).float()/255.0, depth.unsqueeze(1), mask=mask.unsqueeze(1).to(torch.bool),
                               intrinsics=torch.tensor(calib['intrinsics']['left']).float())
            if scene is None:
                scene = SurfelMap(frame=frame, kmat=torch.tensor(calib['intrinsics']['left']).float(), upscale=1)
            else:
                scene.fuse(frame, torch.tensor(tuple2list(slam.get_pose())))
        if log:
            log_dict = {'frame': idx,
                        'surfels/total': scene.opts.shape[1] if scene is not None else 0,
                        'surfels/stable': (scene.conf >= 1.0).sum().item() if scene is not None else 0}
            log_dict.update({f'pyr0/cost': slam.get_residual_error()})
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
