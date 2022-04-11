import sys
sys.path.append('..')
from dataset.semantic_dataset import RGBDDataset
from dataset.scared_dataset import ScaredDataset
from dataset.video_dataset import StereoVideoDataset
from dataset.rectification import StereoRectifier
from ElasticFusion import pyElasticFusion
import os, glob
import json
import torch
import numpy as np
from tqdm import tqdm
from viewer.slam_viewer import SlamViewer
from stereo_slam.disparity.disparity_model import DisparityModel
from stereo_slam.segmentation_network.seg_model import SemanticSegmentationModel
from emdq_slam.emdq_slam_pipeline import EmdqGlueSLAM
from alley_oop.geometry.camera import PinholeCamera


def main(input_path, output_path, config, force_cpu, nsamples):
    device = torch.device('cuda' if (torch.cuda.is_available() & (not force_cpu)) else 'cpu')
    if os.path.isfile(os.path.join(input_path, 'camcal.json')):
        calib_file = os.path.join(input_path, 'camcal.json')
    elif os.path.isfile(os.path.join(input_path, 'StereoCalibration.ini')):
        calib_file = os.path.join(input_path, 'StereoCalibration.ini')
    elif os.path.isfile(os.path.join(input_path, 'endoscope_calibration.yaml')):
        calib_file = os.path.join(input_path, 'endoscope_calibration.yaml')
    else:
        raise RuntimeError('no calibration file found')

    rect = StereoRectifier(calib_file, img_size_new=config['img_size'])
    calib = rect.get_rectified_calib()
    viewer = SlamViewer(calib['intrinsics']['left'], config['viewer']) if config['viewer']['enable'] else None

    try:
        dataset = RGBDDataset(input_path, calib['bf'], img_size=calib['img_size'])
    except AssertionError:
        try:
            dataset = ScaredDataset(input_path, calib['bf'], img_size=calib['img_size'])
        except AssertionError:
            video_file = glob.glob(os.path.join(input_path, '*.mp4'))[0]
            pose_file = os.path.join(input_path, 'camera-poses.json')
            dataset = StereoVideoDataset(video_file, calib_file, pose_file, img_size=calib['img_size'], sample=config['sample'])
            disp_model = DisparityModel(calibration=calib, device=device, depth_clipping=config['depth_clipping'])
            seg_model = SemanticSegmentationModel('stereo_slam/segmentation_network/trained/PvtB2_combined_TAM_fold1.pth', device)

    slam = pyElasticFusion(calib['intrinsics']['left'], calib['img_size'][0], calib['img_size'][1], 7.0, True, 15.0)
    camera = PinholeCamera(calib['intrinsics']['left'])
    slam2 = EmdqGlueSLAM(camera, config['emdq'], device)
    trajectory = []
    last_pose = np.eye(4)
    for i, data in enumerate(tqdm(dataset, total=len(dataset))):
        if isinstance(dataset, StereoVideoDataset):
            limg, rimg, pose_kinematics, img_number = data
            depth, depth_valid = disp_model(limg, rimg)
            mask = seg_model.get_mask(limg)[0]
            mask &= depth_valid  # mask tools and non-valid depth

        else:
            limg, depth, mask, img_number = data
            config['efusion']['kinematics'] = 'fuse'
        #if (i == 0) & (viewer is not None): viewer.set_reference(limg, depth)
        emdq_pose, inliers = slam2(limg, depth, mask)
        diff_pose = np.linalg.pinv(emdq_pose) @ last_pose  # that's the inverted relative pose, the inversion is required due to different coordinate systems of opencv and efusion
        last_pose = emdq_pose
        pose= slam.processFrame(limg, depth.astype(np.uint16), (mask == 0).astype(np.uint8), img_number, diff_pose, config['efusion']['kinematics'] == 'fuse')
        if (i%200) == 0:
            pcl = slam.getPointCloud()
            if pcl is not None:
                print(pcl.shape)
        trajectory.append({'camera-pose': pose.tolist(), 'timestamp': img_number, 'residual': 0.0, 'key_frame': True})
        if len(trajectory) > nsamples:
            break
        #if viewer is not None: viewer(limg, *slam.get_matching_res(), pose)
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, 'trajectory.json'), 'w') as f:
        json.dump(trajectory, f)
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
        default='configuration/combined.yaml',
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
        args.outpath = os.path.join(args.input, 'data','combined')

    main(args.input, args.outpath, config, args.force_cpu is not None, args.nsamples)
