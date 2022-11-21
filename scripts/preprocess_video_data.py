import sys
sys.path.append('../')
from core.slam import SLAM
import os
import torch
import numpy as np
from tqdm import tqdm
from dataset.preprocess.disparity.disparity_model import DisparityModel
from dataset.preprocess.segmentation_network.seg_model import SemanticSegmentationModel
from dataset.dataset_utils import get_data, StereoVideoDataset
from dataset.semantic_dataset import RGBDecoder
import warnings
from torch.utils.data import DataLoader
import wandb
import cv2
from core.utils.pfm_handler import save_pfm


class ColorMatcher(object):
    """
        match color histograms of images
    """
    def __init__(self, template_src=None, template_trg=None):
        if (template_src is not None) & (template_trg is not None):
            self.interp_a_values = []
            for c in range(3):
                src_values, src_unique_indices, src_counts = np.unique(np.concatenate((template_src[..., c].ravel(), np.arange(0,255))),
                                                                       return_inverse=True,
                                                                       return_counts=True)
                src_counts -= 1

                tmpl_values, tmpl_counts = np.unique(template_trg[..., c].ravel(), return_counts=True)

                # calculate normalized quantiles for each array
                src_quantiles = np.cumsum(src_counts) / template_src[..., c].size
                tmpl_quantiles = np.cumsum(tmpl_counts) / template_trg[..., c].size

                self.interp_a_values.append(np.interp(src_quantiles, tmpl_quantiles, tmpl_values))
        else:
            self.interp_a_values = np.load('color_hist.npy')

    def __call__(self, srcl, srcr):
        return self._match(srcl), self._match(srcr)

    def _match(self, src):
        src = (src.permute(1,2,0).numpy()*255.0).astype(np.uint8)
        matched = np.empty_like(src)
        for c in range(3):
            src_values, src_unique_indices = np.unique(np.concatenate((src[..., c].ravel(), np.arange(0,255))),return_inverse=True)
            matched[..., c] = self.interp_a_values[c][src_unique_indices[:src[..., c].size]].reshape(src[..., c].shape)
        return torch.tensor(matched, dtype=torch.float).permute(2,0,1)/255.0


def main(input_path, output_path, device_sel, step, log, match_color, rect_mode):
    device = torch.device('cpu')
    if device_sel == 'gpu':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            warnings.warn('No GPU available, fallback to CPU')

    if log is not None:
        wandb.init(project='Alley-OOP-dataextraction', group=log)

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
            "fuse_step": 1}
    slam = SLAM(torch.tensor(calib['intrinsics']['left']).to(device), config, img_shape=(640, 512),
                baseline=calib['bf'],
                checkpoint="../trained/no_tools_22666t1j.pth", init_pose=torch.eye(4)).to(device)
    assert isinstance(dataset, StereoVideoDataset)
    if match_color:
        cmatcher = ColorMatcher()
        dataset.transform = cmatcher

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
            segmentation = seg_model.segment(limg.to(device))[1]
            segmentation = rgb_decoder.colorize(segmentation.squeeze().cpu().numpy()).astype(np.uint8)
            depth, flow, _ = slam.pose_estimator.estimate_depth(limg.to(device), rimg.to(device))
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
    import yaml
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
        '--match_color',
        action='store_true',
        help='match color of images with a template img. Enable this for phantom data'
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

    main(args.input, args.outpath, args.device, args.step, args.log, args.match_color, args.rect_mode)
