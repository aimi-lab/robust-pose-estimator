from alley_oop.geometry.camera import PinholeCamera
from dataset.camera_utils import readCalibJson
from dataset.dataset_utils import RGBDDataset, ResizeRGBD
from emdq_slam.emdq_slam_pipeline import EmdqSLAM
import os
import json

input_path = '/home/mhayoz/research/innosuisse_surgical_robot/01_Datasets/02_segmentation/intuitive_segmentation/porcine_video/20180731_porcine_kidney_part0019'

width, height, bf, intrinsics = readCalibJson(os.path.join(input_path, 'slam_config_640x480.yaml'))
transform = ResizeRGBD((width, height))
dataset = RGBDDataset(input_path, bf, transform=transform)


camera = PinholeCamera(intrinsics)
slam = EmdqSLAM(camera)

trajectory = []
for img, depth, mask, img_number in dataset:
    pose = slam(img, depth, mask)
    trajectory.append({'camera-pose': pose.tolist(), 'timestamp': img_number, 'residual': 0.0, 'key_frame': True})


with open('trajectory.json', 'w') as f:
    json.dump(trajectory, f)