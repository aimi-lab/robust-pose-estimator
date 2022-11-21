import sys
sys.path.append('../')
from core.pose.pose_estimator import PoseEstimator
import os
import torch
import numpy as np
from tqdm import tqdm
from dataset.preprocess.segmentation_network.seg_model import SemanticSegmentationModel
from dataset.dataset_utils import get_data, StereoVideoDataset
from dataset.semantic_dataset import RGBDecoder
import warnings
from torch.utils.data import DataLoader
import wandb
import cv2
from core.utils.pfm_handler import save_pfm


def main(input_path, output_path, device_sel, step, log, rect_mode):
    device = torch.device('cpu')
    if device_sel == 'gpu':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            warnings.warn('No GPU available, fallback to CPU')

    if log is not None:
        wandb.init(project='data extraction', group=log)

    dataset, calib = get_data(input_path, (1280, 1024), sample_video=step, rect_mode=rect_mode)
    config={"frame2frame": True,
            "dist_thr": 0.05, # relative distance threshold for data association
            "depth_clipping":[1,250], # in mm
            "debug": False,
            "mask_specularities": True,
            "conf_weighing": True,
            "compensate_illumination": False,  # do not use this with noisy depth estimates
            "average_pts": False,
            "fuse_mode": "projective",
            "fuse_step": 1,
            "img_size": (1280, 1024)}
    pose_estimator = PoseEstimator(config, torch.tensor(calib['intrinsics']['left']).to(device),
                                   baseline=calib['bf'],
                                   checkpoint=args.checkpoint,
                                   img_shape=config['img_size']).to(device)
    assert isinstance(dataset, StereoVideoDataset)

    loader = DataLoader(dataset, num_workers=1, pin_memory=True)

    seg_model = SemanticSegmentationModel('../dataset/preprocess/segmentation_network/trained/deepLabv3plus_trained_intuitive.pth',
                                          device, (1280, 1024))

    rgb_decoder = RGBDecoder()

    os.makedirs(os.path.join(output_path, 'video_frames'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'semantic_predictions'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'depth'), exist_ok=True)

    with torch.inference_mode():
        for i, data in enumerate(tqdm(loader, total=len(dataset))):
            limg, rimg, pose_kinematics, img_number = data
            segmentation = seg_model.segment(limg.to(device)/255.0)[1]
            segmentation = rgb_decoder.colorize(segmentation.squeeze().cpu().numpy()).astype(np.uint8)
            depth, flow, _ = pose_estimator.flow2depth(limg.to(device), rimg.to(device), pose_estimator.baseline)
            depth = depth.squeeze().cpu().numpy()

            # store images and depth and semantics
            if torch.is_tensor(img_number):
                img_name = f'{img_number.item():06d}'
            else:
                img_name = f'{int(img_number[0]):06d}'
            cv2.imwrite(os.path.join(output_path, 'video_frames', img_name+'l.png'), cv2.cvtColor(255.0*limg.squeeze().permute(1,2,0).cpu().numpy(), cv2.COLOR_RGB2BGR).astype(np.uint8))
            cv2.imwrite(os.path.join(output_path, 'video_frames', img_name + 'r.png'),
                        cv2.cvtColor(255.0*rimg.squeeze().permute(1,2,0).cpu().numpy(), cv2.COLOR_RGB2BGR).astype(np.uint8))
            cv2.imwrite(os.path.join(output_path, 'semantic_predictions', img_name + 'l.png'), cv2.cvtColor(segmentation, cv2.COLOR_RGB2BGR))
            save_pfm(depth, os.path.join(output_path,'depth', f'{img_number[0]}l.pfm'))
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
        '--device',
        choices=['cpu', 'gpu'],
        default='cpu',
        help='select cpu or gpu to run slam.'
    )

    parser.add_argument(
        '--step',
        type=int,
        default=1,
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

    main(args.input, args.outpath, args.device, args.step, args.log, args.rect_mode)
