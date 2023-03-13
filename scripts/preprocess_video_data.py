import sys
sys.path.append('../')
import os
import torch
import numpy as np
from tqdm import tqdm
from dataset.dataset_utils import get_data, StereoVideoDataset
from torch.utils.data import DataLoader
import wandb
import cv2


def _check_valid(valid_list, n):
    if valid_list is None:
        return True
    valid = False
    for v in valid_list:
        if (n >= v[0]) & (n < v[1]):
            valid = True
    return valid


def main(input_path, output_path, step, log, rect_mode):
    # only extract valid frames for training
    if os.path.isfile(os.path.join(input_path, 'split.csv')):
        valid_list = np.genfromtxt(os.path.join(input_path, 'valid.csv'), header=1, delimiter=',')
    else:
        valid_list = None
    if log is not None:
        wandb.init(project='data extraction', group=log)

    dataset, calib = get_data(input_path, (1280, 1024), sample_video=step, rect_mode=rect_mode)
    assert isinstance(dataset, StereoVideoDataset)

    loader = DataLoader(dataset, num_workers=1)

    os.makedirs(os.path.join(output_path, 'video_frames'), exist_ok=True)

    with torch.inference_mode():
        for i, data in enumerate(tqdm(loader, total=len(dataset))):
            limg, rimg, pose_kinematics, img_number = data

            if _check_valid(valid_list, int(img_number[0])):
                # store images and depth and mask
                if torch.is_tensor(img_number):
                    img_name = f'{img_number.item():06d}'
                else:
                    img_name = f'{int(img_number[0]):06d}'
                cv2.imwrite(os.path.join(output_path, 'video_frames', img_name+'l.png'), cv2.cvtColor(255.0*limg.squeeze().permute(1,2,0).cpu().numpy(), cv2.COLOR_RGB2BGR).astype(np.uint8))
                cv2.imwrite(os.path.join(output_path, 'video_frames', img_name + 'r.png'),
                            cv2.cvtColor(255.0*rimg.squeeze().permute(1,2,0).cpu().numpy(), cv2.COLOR_RGB2BGR).astype(np.uint8))
        print('finished')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='script to extract stereo data')

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
        '--step',
        type=int,
        default=2,
        help='sub sampling interval.'
    )
    parser.add_argument(
        '--log',
        default=None,
        help='wandb group logging name. No logging if none set'
    )
    parser.add_argument(
        '--rect_mode',
        type=str,
        choices=['conventional', 'pseudo'],
        default='conventional',
        help='rectification mode, use pseudo for SCARED'
    )
    args = parser.parse_args()
    if args.outpath is None:
        args.outpath = args.input

    main(args.input, args.outpath, args.step, args.log, args.rect_mode)
