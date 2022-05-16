import sys
sys.path.append('../')
from dataset.semantic_dataset import RGBDDataset
from dataset.scared_dataset import ScaredDataset
from dataset.tum_dataset import TUMDataset
from dataset.video_dataset import StereoVideoDataset
from dataset.rectification import StereoRectifier
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
import matplotlib.pyplot as plt
from alley_oop.fusion.surfel_map import SurfelMap
from alley_oop.utils.trajectory import save_trajectory
import warnings
from torch.utils.data import DataLoader


def main(input_path, output_path, config, device_sel, nsamples):
    device = torch.device('cpu')
    if device_sel == 'gpu':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            warnings.warn('No GPU available, fallback to CPU')

    with torch.inference_mode():
        calib_file = None
        if os.path.isfile(os.path.join(input_path, 'camcal.json')):
            calib_file = os.path.join(input_path, 'camcal.json')
        elif os.path.isfile(os.path.join(input_path, 'StereoCalibration.ini')):
            calib_file = os.path.join(input_path, 'StereoCalibration.ini')
        elif os.path.isfile(os.path.join(input_path, 'endoscope_calibration.yaml')):
            calib_file = os.path.join(input_path, 'endoscope_calibration.yaml')
        else:
            try:
                dataset = TUMDataset(input_path, config['img_size'])
                calib = {'intrinsics': {'left': dataset.get_intrinsics()}}
            except AssertionError:
                raise RuntimeError('no calibration file found')
        if calib_file is not None:
            rect = StereoRectifier(calib_file, img_size_new=config['img_size'])
            calib = rect.get_rectified_calib()

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
        viewer = Viewer3D((3 * config['img_size'][0], 3 * config['img_size'][1]),
                          blocking=config['viewer']['blocking']) if config['viewer']['enable'] else None

        slam = SLAM(torch.tensor(calib['intrinsics']['left']), config['slam']).to(device)
        trajectory = []
        for i, data in enumerate(tqdm(dataset, total=len(dataset))):
            if isinstance(dataset, StereoVideoDataset):
                limg, rimg, pose_kinematics, img_number = data
                depth, depth_valid = disp_model(limg, rimg)
                mask = seg_model.get_mask(limg)[0]
                mask &= depth_valid  # mask tools and non-valid depth
            else:
                limg, depth, mask, img_number = data
            pose, scene = slam.processFrame(limg, depth)
            if viewer is not None:
                curr_pcl = SurfelMap(frame=slam.get_frame(), kmat=torch.tensor(calib['intrinsics']['left']).float(), pmat=pose).pcl2open3d(stable=False)
                curr_pcl.paint_uniform_color([0.5,0.5,0.5])
                viewer(pose, scene.pcl2open3d(stable=config['viewer']['stable']), add_pcd=curr_pcl,
                       frame=slam.get_frame(), synth_frame=slam.get_rendered_frame(), optim_results=slam.get_optimization_res())
            trajectory.append({'camera-pose': pose.tolist(), 'timestamp': img_number, 'residual': 0.0, 'key_frame': True})
            if len(trajectory) > nsamples:
                break
            if (i%50) == 0:
                pcl = scene.pcl2open3d(stable=True)
                open3d.io.write_point_cloud(os.path.join(output_path, f'map_{i:04d}.ply'), pcl)
        os.makedirs(output_path, exist_ok=True)
        save_trajectory(trajectory, output_path)

        plt.close()
        fig, ax = slam.plot_recordings()
        plt.savefig(os.path.join(output_path, 'optimization_plot.pdf'))

        pcl = scene.pcl2open3d(stable=True)
        open3d.io.write_point_cloud(os.path.join(output_path, f'map.ply'), pcl)

        if viewer is not None:
            viewer.blocking = True
            viewer(pose, scene.pcl2open3d(stable=config['viewer']['stable']), frame=slam.get_frame(),
                   synth_frame=slam.get_rendered_frame())
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

    main(args.input, args.outpath, config, args.device is not None, args.nsamples)
