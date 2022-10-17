import glob
import os
import csv
import numpy as np
import cv2

from alley_oop.utils.trajectory import read_freiburg
from alley_oop.network_core.raft.core.datasets import RGBDecoder
from tqdm import tqdm

def annotate(root):
    rgb_decoder = RGBDecoder()
    def _read_mask(path):
        mask = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        mask = rgb_decoder.getToolMask(mask)
        return mask
    images_l = sorted(glob.glob(os.path.join(root, 'video_frames', '*l.png')))
    semantics = sorted(glob.glob(os.path.join(root, 'semantic_predictions', '*l.png')))
    gt_file = os.path.join(root, 'groundtruth.txt')
    poses = read_freiburg(gt_file)
    assert len(images_l) > 0 , f'no images in {root}'
    assert len(semantics) == len(images_l)

    annotations = []
    for i in tqdm(range(len(images_l)-1), total=len(images_l)):
        camera_motion = np.linalg.norm(np.linalg.inv(poses[i+1].astype(np.float64)) @ poses[i].astype(np.float64) - np.eye(4)) > 0.01
        tool_present = np.mean(_read_mask(semantics[i])) < 0.95
        annotations.append((camera_motion, tool_present))
    with open(os.path.join(root, 'annotions.csv'), 'w') as f:
        wr = csv.writer(f)
        wr.writerow(['camera_motion', 'tool_present'])
        wr.writerows(annotations)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='script to annotate video for camera motion and tool presence')

    parser.add_argument(
        'input',
        type=str,
        help='Path to input folder.'
    )
    args = parser.parse_args()

    print(args.input)
    if not os.path.isfile(os.path.join(args.input, 'annotions.csv')):
        annotate(args.input)