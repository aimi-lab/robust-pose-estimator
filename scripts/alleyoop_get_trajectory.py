import sys
sys.path.append('../')
from alley_oop.slam import SLAM
import os
import torch
import numpy as np
from tqdm import tqdm
from dataset.preprocess.disparity.disparity_model import DisparityModel
from dataset.preprocess.segmentation_network.seg_model import SemanticSegmentationModel
import matplotlib.pyplot as plt
from alley_oop.fusion.surfel_map import SurfelMap
from alley_oop.utils.trajectory import save_trajectory
from dataset.dataset_utils import get_data, StereoVideoDataset, SequentialSubSampler
from dataset.transforms import Compose
import warnings
from torch.utils.data import DataLoader
import wandb
from scripts.evaluate_ate_freiburg import main as evaluate


def main(input_path, output_path, config, device_sel, stop, start, step, log):
    device = torch.device('cpu')
    if device_sel == 'gpu':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            warnings.warn('No GPU available, fallback to CPU')

    if log is not None:
        config.update({'data': os.path.split(input_path)[-1]})
        wandb.init(project='Alley-OOP', config=config, group=log)

    dataset, calib = get_data(input_path, config['img_size'])
    slam = SLAM(torch.tensor(calib['intrinsics']['left']), config['slam'], img_shape=config['img_size']).to(device)
    dataset.transform = Compose([dataset.transform, slam.pre_process])  # add pre-processing to data loading (CPU)
    sampler = SequentialSubSampler(dataset, start, stop, step)
    loader = DataLoader(dataset, num_workers=0 if config['slam']['debug'] else 1, pin_memory=True, sampler=sampler)

    if isinstance(dataset, StereoVideoDataset):
        disp_model = DisparityModel(calibration=calib, device=device, depth_clipping=config['depth_clipping'])
        seg_model = SemanticSegmentationModel('stereo_slam/segmentation_network/trained/PvtB2_combined_TAM_fold1.pth',
                                              device)
    with torch.inference_mode():
        viewer = None
        if config['viewer']['enable']:
            import open3d
            from viewer.viewer3d import Viewer3D
            viewer = Viewer3D((2 * config['img_size'][0], 2 * config['img_size'][1]),
                              blocking=config['viewer']['blocking'])

        trajectory = []
        os.makedirs(output_path, exist_ok=True)
        for i, data in enumerate(tqdm(loader, total=min(len(dataset), (stop-start)//step))):
            if isinstance(dataset, StereoVideoDataset):
                raise NotImplementedError
                limg, rimg, pose_kinematics, img_number = data
                depth, depth_valid = disp_model(limg, rimg)
                mask = seg_model.get_mask(limg)[0]
                mask &= depth_valid  # mask tools and non-valid depth
            else:
                limg, depth, mask, confidence, img_number = data
            pose, scene, pose_relscale = slam.processFrame(limg.to(device), depth.to(device), mask.to(device),
                                                           confidence.to(device))

            if viewer is not None:
                curr_pcl = SurfelMap(frame=slam.get_frame(), kmat=torch.tensor(calib['intrinsics']['left']).float(),
                                     pmat=pose_relscale, depth_scale=scene.depth_scale).pcl2open3d(stable=False)
                if scene.patch_colors is not None:
                    curr_pcl.colors = open3d.utility.Vector3dVector(scene.patch_colors[slam.get_frame().mask.view(-1)])
                else:
                    curr_pcl.paint_uniform_color([0.5,0.5,0.5])
                viewer(pose.cpu(), scene.pcl2open3d(stable=config['viewer']['stable']), add_pcd=curr_pcl,
                       frame=slam.get_frame(), synth_frame=slam.get_rendered_frame(), optim_results=slam.get_optimization_res())
            trajectory.append({'camera-pose': pose.tolist(), 'timestamp': img_number[0], 'residual': 0.0, 'key_frame': True})
            if (log is not None) & (i > 0):
                slam.recorder.log(step=i)

        save_trajectory(trajectory, output_path)


        plt.close()
        fig, ax = slam.plot_recordings()
        plt.savefig(os.path.join(output_path, 'optimization_plot.pdf'))

        scene.save_ply(os.path.join(output_path, 'map.ply'), stable=True)
        if log is not None:
            wandb.save(os.path.join(output_path, 'trajectory.freiburg'))
            wandb.save(os.path.join(output_path, 'trajectory.json'))
            wandb.save(os.path.join(output_path, 'map.ply'))
            if os.path.isfile(os.path.join(input_path, 'groundtruth.txt')):
                error = evaluate(os.path.join(input_path, 'groundtruth.txt'),
                                 os.path.join(output_path, 'trajectory.freiburg'))
                wandb.define_metric('trans_error', step_metric='frame')
                for i, e in enumerate(error):
                    wandb.log({'trans_error': e, 'frame': i})
                wandb.summary['ATE/RMSE'] = np.sqrt(np.dot(error,error) / len(error))
                wandb.summary['ATE/mean'] = np.mean(error)

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
        '--stop',
        type=int,
        default=10000000000,
        help='number of samples to run for.'
    )
    parser.add_argument(
        '--start',
        type=int,
        default=0,
        help='at which sample to start slam.'
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
    args = parser.parse_args()
    with open(args.config, 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    if args.outpath is None:
        args.outpath = os.path.join(args.input, 'data','alleyoop')

    main(args.input, args.outpath, config, args.device, args.stop, args.start, args.step, args.log)
