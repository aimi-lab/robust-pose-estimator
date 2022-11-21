import sys
sys.path.append('../')
from core.slam import SLAM
import os
import torch
import numpy as np
from tqdm import tqdm
from core.fusion.surfel_map import SurfelMap, Frame
from core.utils.trajectory import read_freiburg
from dataset.dataset_utils import get_data, StereoVideoDataset, SequentialSubSampler, RGBDDataset
from dataset.transforms import Compose
import warnings
from torch.utils.data import DataLoader
import tifffile
import cv2


def main(input_path, output_path, config, device_sel, stop, start, step, log, file):
    gt_depth = None
    if os.path.isfile(os.path.join(input_path, 'left_depth_map.tiff')):
        t1 = tifffile.imread(os.path.join(input_path, 'left_depth_map.tiff'))
        t1[(t1 == np.inf) | (t1 != t1) | (t1 < 0)] = 0
        gt_depth = t1[...,2]
        gt_depth = cv2.resize(gt_depth, (640, 512), interpolation=cv2.INTER_NEAREST)
        gt_depth_valid = gt_depth > 0

    device = torch.device('cpu')
    if device_sel == 'gpu':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            warnings.warn('No GPU available, fallback to CPU')

    dataset, calib = get_data(input_path, config['img_size'], rect_mode=config['rect_mode'])
    # check for ground-truth pose data for logging purposes
    gt_file = os.path.join(input_path, file)
    gt_trajectory = read_freiburg(gt_file) if os.path.isfile(gt_file) else None

    slam = SLAM(torch.tensor(calib['intrinsics']['left']).to(device), config['slam'],img_shape=config['img_size'], baseline=calib['bf'],
                checkpoint="../trained/dummy.pth",
                init_pose=torch.tensor(gt_trajectory[0]) if gt_trajectory is not None else None).to(device)

    if not isinstance(dataset, StereoVideoDataset):
        sampler = SequentialSubSampler(dataset, start, stop, step)
    else:
        warnings.warn('start/stop arguments not supported for video dataset. ignored.', UserWarning)
        sampler = None
    loader = DataLoader(dataset, num_workers=0 if config['slam']['debug'] else 1, pin_memory=True, sampler=sampler)

    with torch.inference_mode():
        from viewer.viewer3d import Viewer3D
        viewer = Viewer3D((2 * config['img_size'][0], 2 * config['img_size'][1]),
                              blocking=True)

        os.makedirs(output_path, exist_ok=True)
        for i, data in enumerate(tqdm(loader, total=min(len(dataset), (stop-start)//step))):
            if isinstance(dataset, StereoVideoDataset):
                limg, rimg, pose_kinematics, img_number = data
                depth, flow, _ = slam.pose_estimator.estimate_depth(limg.to(device), rimg.to(device))
            elif isinstance(dataset, RGBDDataset):
                limg, depth, tool_mask, semantic, img_number = data
            else:
                limg, rimg, tool_mask, semantics, img_number = data
                depth, flow, _ = slam.pose_estimator.estimate_depth(limg.to(device), rimg.to(device))
            frame = Frame(limg, limg, depth, intrinsics=torch.tensor(calib['intrinsics']['left']).float())

            pose_gt = torch.tensor(gt_trajectory[int(img_number[0])]).float()

            print(pose_gt)
            if i == 0:
                first_pcl = SurfelMap(frame=frame, kmat=torch.tensor(calib['intrinsics']['left']).float(),
                                      pmat=pose_gt, depth_scale=1).pcl2open3d(stable=False)
                if gt_depth is not None:
                    print("median depth error: ", np.median(((depth.squeeze().numpy() - gt_depth)[gt_depth_valid])))

            curr_pcl = SurfelMap(frame=frame, kmat=torch.tensor(calib['intrinsics']['left']).float(),
                                 pmat=pose_gt, depth_scale=1).pcl2open3d(stable=False)
            viewer(pose_gt.cpu(), first_pcl, add_pcd=curr_pcl)

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
        '--file',
        default='groundtruth.txt',
        help='path to freiburg file'
    )
    args = parser.parse_args()
    with open(args.config, 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    if args.outpath is None:
        args.outpath = os.path.join(args.input, 'data','infer_trajectory')

    main(args.input, args.outpath, config, args.device, args.stop, args.start, args.step, args.log, file=args.file)
