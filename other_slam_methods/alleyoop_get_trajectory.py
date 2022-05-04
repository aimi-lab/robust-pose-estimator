import sys
sys.path.append('../..')
from dataset.semantic_dataset import RGBDDataset
from dataset.scared_dataset import ScaredDataset
from dataset.video_dataset import StereoVideoDataset
from dataset.rectification import StereoRectifier
from ElasticFusion import pyElasticFusion
from alley_oop.slam import SLAM
import os, glob
import json
import torch
import numpy as np
from tqdm import tqdm
from viewer.viewer3d import Viewer3D
from other_slam_methods.stereo_slam.disparity.disparity_model import DisparityModel
from other_slam_methods.stereo_slam.segmentation_network.seg_model import SemanticSegmentationModel
import open3d

def save_ply(pcl_array,colors,  path):
    pcl = open3d.geometry.PointCloud()
    pcl.points = open3d.utility.Vector3dVector(pcl_array.cpu().numpy())
    pcl.colors = open3d.utility.Vector3dVector(colors.repeat(1,3).cpu().numpy())
    open3d.io.write_point_cloud(path, pcl)


def main(input_path, output_path, config, force_cpu, nsamples):
    device = torch.device('cuda' if (torch.cuda.is_available() & (not force_cpu)) else 'cpu')
    with torch.no_grad():
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
        viewer = Viewer3D(config['img_size']) if config['viewer']['enable'] else None

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

        #slam = pyElasticFusion(calib['intrinsics']['left'], calib['img_size'][0], calib['img_size'][1], 7.0, True, 15.0)
        slam = SLAM(torch.tensor(calib['intrinsics']['left']), config['slam'])
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
            pose, scene = slam.processFrame(limg, depth.astype(np.uint16))
            viewer(pose, scene.pcl2open3d(), frame=slam.get_frame(), synth_frame=slam.get_rendered_frame())
            trajectory.append({'camera-pose': pose.tolist(), 'timestamp': img_number, 'residual': 0.0, 'key_frame': True})
            if len(trajectory) > nsamples:
                break
            #if viewer is not None: viewer(limg, *slam.get_matching_res(), pose)
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, 'trajectory.json'), 'w') as f:
            json.dump(trajectory, f)
        pcl = slam.getPointCloud()
        if pcl is not None:
            save_ply(*pcl, os.path.join(output_path, 'map.ply'))
            print(pcl[0].shape)
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
        default='stereo_slam/configuration/alleyoop_slam.yaml',
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
        args.outpath = os.path.join(args.input, 'data','alleyoop')

    main(args.input, args.outpath, config, args.force_cpu is not None, args.nsamples)
