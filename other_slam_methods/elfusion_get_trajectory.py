import sys
sys.path.append('../..')
from dataset.dataset_utils import get_data
from dataset.video_dataset import StereoVideoDataset
from dataset.rectification import StereoRectifier
from ElasticFusion import pyElasticFusion
import os, glob
import json
import torch
import numpy as np
from tqdm import tqdm
from viewer.slam_viewer import SlamViewer
from other_slam_methods.stereo_slam.disparity.disparity_model import DisparityModel
from other_slam_methods.stereo_slam.segmentation_network.seg_model import SemanticSegmentationModel
import open3d
import warnings


def save_ply(pcl_array, path):
    pcl = open3d.geometry.PointCloud()
    pcl.points = open3d.utility.Vector3dVector(pcl_array[:, :3])
    pcl.colors = open3d.utility.Vector3dVector(pcl_array[:, 4:7])
    open3d.io.write_point_cloud(path, pcl)


def main(input_path, output_path, config, device_sel, nsamples):
    device = torch.device('cpu')
    if device_sel == 'gpu':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            warnings.warn('No GPU available, fallback to CPU')
    dataset, calib = get_data(input_path, config['img_size'])
    slam = pyElasticFusion(calib['intrinsics']['left'], calib['img_size'][0], calib['img_size'][1], 7.0, True, 20.0)

    if isinstance(dataset, StereoVideoDataset):
        disp_model = DisparityModel(calibration=calib, device=device, depth_clipping=config['depth_clipping'])
        seg_model = SemanticSegmentationModel('stereo_slam/segmentation_network/trained/PvtB2_combined_TAM_fold1.pth',
                                              device)


    trajectory = []
    last_pose = np.eye(4)
    for i, data in enumerate(tqdm(dataset, total=len(dataset))):
        if isinstance(dataset, StereoVideoDataset):
            limg, rimg, pose_kinematics, img_number = data
            depth, depth_valid = disp_model(limg, rimg)
            mask = seg_model.get_mask(limg)[0]
            mask &= depth_valid  # mask tools and non-valid depth
            diff_pose = np.linalg.pinv(last_pose)@pose_kinematics if config['slam']['kinematics'] != 'none' else np.eye(4)
            last_pose = pose_kinematics
        else:
            limg, depth, mask, img_number = data
            diff_pose = np.eye(4)
            config['slam']['kinematics'] = 'fuse'
        #if (i == 0) & (viewer is not None): viewer.set_reference(limg, depth)
        if mask is None:
            mask = np.ones_like(depth)
        pose= slam.processFrame(limg, depth.astype(np.uint16),(mask == 0).astype(np.uint8) , int(img_number), diff_pose, config['slam']['kinematics'] == 'fuse')
        trajectory.append({'camera-pose': pose.tolist(), 'timestamp': img_number, 'residual': 0.0, 'key_frame': True})
        if len(trajectory) > nsamples:
            break
        #if viewer is not None: viewer(limg, *slam.get_matching_res(), pose)
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, 'trajectory.json'), 'w') as f:
        json.dump(trajectory, f)
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
        default='stereo_slam/configuration/efusion_slam.yaml',
        help='Configuration file.'
    )
    parser.add_argument(
        '--force_cpu',
        help='force use of CPU.'
    )
    parser.add_argument(
        '--nsamples',
        type=int,
        default=10000000000,
        help='force use of CPU.'
    )
    args = parser.parse_args()
    with open(args.config, 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    if args.outpath is None:
        args.outpath = os.path.join(args.input, 'data','efusion')

    main(args.input, args.outpath, config, args.force_cpu is not None, args.nsamples)
