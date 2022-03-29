from alley_oop.geometry.camera import PinholeCamera
from dataset.semantic_dataset import RGBDDataset
from dataset.scared_dataset import ScaredDataset
from dataset.video_dataset import StereoVideoDataset
from dataset.rectification import StereoRectifier
from emdq_slam.emdq_slam_pipeline import EmdqSLAM, EmdqGlueSLAM
import os, glob
import json
import torch
from tqdm import tqdm
from viewer.slam_viewer import SlamViewer
from stereo_slam.disparity.disparity_model import DisparityModel
from stereo_slam.segmentation_network.seg_model import SemanticSegmentationModel


def main(input_path, output_path, config, force_cpu):
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
            dataset = StereoVideoDataset(video_file, calib_file, img_size=calib['img_size'], sample=config['sample'])
            disp_model = DisparityModel(calibration=calib, device=device, depth_clipping=config['depth_clipping'])
            seg_model = SemanticSegmentationModel('stereo_slam/segmentation_network/trained/PvtB2_combined_TAM_fold1.pth', device)

    camera = PinholeCamera(calib['intrinsics']['left'])
    slam = EmdqGlueSLAM(camera, config['slam'])
    if viewer is not None: viewer.set_reference(dataset[0][0], dataset[0][1])
    trajectory = []
    for data in tqdm(dataset, total=len(dataset)):
        if isinstance(dataset, StereoVideoDataset):
            limg, rimg, img_number = data
            depth, depth_valid = disp_model(limg, rimg)
            mask = seg_model.get_mask(limg)[0]
            mask &= depth_valid  # mask tools and non-valid depth
        else:
            limg, depth, mask, img_number = data
        pose, inliers = slam(limg, depth, mask)
        if inliers == 0:
            break
        trajectory.append({'camera-pose': pose.tolist(), 'timestamp': img_number, 'residual': 0.0, 'key_frame': True})
        if viewer is not None: viewer(limg, *slam.get_matching_res(), pose)
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
        default='configuration/emdq_slam.yaml',
        help='Configuration file.'
    )
    parser.add_argument(
        '--force_cpu',
        help='force use of CPU.'
    )

    args = parser.parse_args()
    with open(args.config, 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    if args.outpath is None:
        args.outpath = os.path.join(args.input, 'data','emdq_slam')

    main(args.input, args.outpath, config, args.force_cpu is not None)
