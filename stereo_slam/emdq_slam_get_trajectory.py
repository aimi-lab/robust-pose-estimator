from alley_oop.geometry.camera import PinholeCamera
from dataset.semantic_dataset import RGBDDataset
from dataset.scared_dataset import ScaredDataset
from dataset.video_dataset import StereoVideoDataset
from dataset.rectification import StereoRectifier
from emdq_slam.emdq_slam_pipeline import EmdqSLAM
import os
import json
from tqdm import tqdm
from viewer.slam_viewer import SlamViewer
from stereo_slam.disparity.disparity_model import DisparityModel


def main(input_path, output_path, config):
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
            dataset = StereoVideoDataset(input_path, img_size=calib['img_size'], sample=6)
            disp_model = DisparityModel(calibration=calib, device=device)

    camera = PinholeCamera(calib['intrinsics']['left'])
    slam = EmdqSLAM(camera, config['slam'])
    if viewer is not None: viewer.set_reference(dataset[0][0], dataset[0][1])
    trajectory = []
    for data in tqdm(dataset, total=len(dataset)):
        if isinstance(dataset, StereoVideoDataset):
            limg, rimg, img_number = data
            depth = disp_model(limg, rimg)
            mask = seg_model(limg)
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
        default='emdq_slam/configuration/default.yaml',
        help='Configuration file.'
    )

    args = parser.parse_args()
    with open(args.config, 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    if args.outpath is None:
        args.outpath = os.path.join(args.input, 'data','emdq_slam')

    main(args.input, args.outpath, config)
