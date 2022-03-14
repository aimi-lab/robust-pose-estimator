from alley_oop.geometry.camera import PinholeCamera
from dataset.camera_utils import readCalibJson
from dataset.semantic_dataset import RGBDDataset
from dataset.scared_dataset import ScaredDataset
from dataset.transforms import ResizeRGBD
from emdq_slam.emdq_slam_pipeline import EmdqSLAM
import os
import json
from tqdm import tqdm
from viewer.slam_viewer import SlamViewer


def main(input_path, output_path, config):

    width, height, bf, intrinsics = readCalibJson(os.path.join(input_path, 'slam_config_640x480.yaml')) #ToDo use rectified
    transform = ResizeRGBD((width, height))
    viewer = SlamViewer(intrinsics, config['viewer']) if config['viewer']['enable'] else None

    try:
        dataset = RGBDDataset(input_path, bf, transform=transform)
    except AssertionError:
        dataset = ScaredDataset(input_path, bf, transform=transform)

    camera = PinholeCamera(intrinsics)
    slam = EmdqSLAM(camera, config['slam'])
    if viewer is not None: viewer.set_reference(dataset[0][0], dataset[0][1])
    trajectory = []
    for img, depth, mask, img_number in tqdm(dataset, total=len(dataset)):
        pose, inliers = slam(img, depth, mask)
        if inliers == 0:
            break
        trajectory.append({'camera-pose': pose.tolist(), 'timestamp': img_number, 'residual': 0.0, 'key_frame': True})
        if viewer is not None: viewer(img, *slam.get_matching_res(), pose)
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
