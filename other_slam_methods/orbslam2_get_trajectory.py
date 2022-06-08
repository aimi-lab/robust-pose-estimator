import orbslam2
from dataset.dataset_utils import get_data, StereoVideoDataset, SequentialSubSampler
import json
import os
from tqdm import tqdm
from alley_oop.utils.trajectory import save_trajectory
from dataset.preprocess.disparity.disparity_model import DisparityModel
from dataset.preprocess.segmentation_network.seg_model import SemanticSegmentationModel
import yaml
import cv2
from torch.utils.data import DataLoader
import warnings
import torch
import wandb


def main(input_path, outpath, config, device_sel, start, stop, step, log):
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
    settings_path = os.path.join(input_path, 'slam_config_640x480.yaml')
    assert os.path.isfile(settings_path)
    slam = orbslam2.System(os.path.join('Vocabulary', 'ORBvoc.txt'), settings_path, orbslam2.Sensor.RGBD)
    sampler = SequentialSubSampler(dataset, start, stop, step)
    loader = DataLoader(dataset, num_workers=0 if config['slam']['debug'] else 1, pin_memory=True, sampler=sampler)
    if isinstance(dataset, StereoVideoDataset):
        disp_model = DisparityModel(calibration=calib, device=device, depth_clipping=config['depth_clipping'])
        seg_model = SemanticSegmentationModel('stereo_slam/segmentation_network/trained/PvtB2_combined_TAM_fold1.pth',
                                              device)

    slam.set_use_viewer(config['viewer']['enable'])
    slam.initialize()

    trajectory = []
    for idx, data in enumerate(tqdm(loader, total=min(len(dataset), (stop-start)//step))):
        if isinstance(dataset, StereoVideoDataset):
            raise NotImplementedError
            limg, rimg, pose_kinematics, img_number = data
            depth, depth_valid = disp_model(limg, rimg)
            mask = seg_model.get_mask(limg)[0]
            mask &= depth_valid  # mask tools and non-valid depth
        else:
            limg, depth, mask, rimg, disp, img_number = data
        slam.process_image_rgbd(limg.squeeze().numpy(), depth.squeeze().numpy(), mask.squeeze().numpy(), float(idx))

        is_key_frame = (slam.map_changed() | (idx == 0))
        if slam.get_tracking_state() == orbslam2.TrackingState.OK:
            trajectory.append(
                {'camera-pose': slam.get_pose(), 'timestamp': img_number, 'residual': slam.get_residual_error(), 'key_frame': is_key_frame})
        else:
            trajectory.append(
                {'camera-pose': trajectory[-1]['camera-pose'], 'timestamp': img_number, 'residual': slam.get_residual_error(),
                 'key_frame': is_key_frame})

    os.makedirs(outpath, exist_ok=True)
    save_trajectory(trajectory, outpath)
    slam.shutdown()

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
    args = parser.parse_args()
    with open(args.config, 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    if args.outpath is None:
        args.outpath = os.path.join(args.input, 'data','orbslam')

    main(args.input, args.outpath, config, args.device, args.start, args.stop, args.step, args.log)
