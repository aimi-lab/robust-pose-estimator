import sys
sys.path.append('../')
import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler


from core.pose.pose_net import PoseN
import core.network_core.raft.core.datasets as datasets
from core.network_core.raft.losses import supervised_pose_loss
from core.utils.plotting import plot_res
import matplotlib.pyplot as plt

from core.geometry.lie_3d_pseudo import pseudo_lie_se3_to_SE3, pseudo_lie_se3_to_SE3_batch


def main(args, config, force_cpu):
    # general
    config['model']['image_shape'] = config['image_shape']
    device = torch.device('cuda') if (torch.cuda.is_available() & (not force_cpu)) else torch.device('cpu')

    # get data
    data_val, infer_depth = datasets.get_data(config['data']['val']['type'], config['data']['val']['basepath'],config['data']['val']['sequences'],
                                             config['image_shape'], config['data']['val']['step'], config['depth_scale'])
    val_loader = DataLoader(data_val, num_workers=4, pin_memory=True, batch_size=config['val']['batch_size'],
                            sampler=SubsetRandomSampler(torch.from_numpy(np.random.choice(len(data_val), size=(400,), replace=False))))

    # get model
    model = PoseN(config['model'])
    model.load_state_dict(torch.load(args.restore_ckpt)['state_dict'], strict=False)

    model.eval()
    model = model.to(device)

    if not os.path.isdir(args.outpath):
        os.mkdir(args.outpath)

    for i_batch, data_blob in enumerate(val_loader):
        if infer_depth:
            ref_img, trg_img, ref_img_r, trg_img_r, ref_mask, trg_mask, gt_pose, intrinsics, baseline = [
                x.to(device) for x in
                data_blob]
        else:
            ref_img, trg_img, ref_depth, trg_depth, ref_conf, trg_conf, valid, gt_pose, intrinsics, baseline = [
                x.to(device) for x in data_blob]

        # forward pass
        if infer_depth:
            flow_predictions, pose_predictions, trg_depth, ref_depth, conf1, conf2 = model(trg_img, ref_img,
                                                                                           intrinsics.float(), baseline.float(),
                                                                                           image1r=trg_img_r,
                                                                                           image2r=ref_img_r,
                                                                                           mask1=trg_mask,
                                                                                           mask2=ref_mask,
                                                                                           iters=config['model'][
                                                                                               'iters'],
                                                                                           ret_confmap=True)
        else:
            flow_predictions, pose_predictions, *_, conf1, conf2 = model(trg_img, ref_img, intrinsics.float(), baseline.float(),
                                                                         depth1=trg_depth,
                                                                         depth2=ref_depth,
                                                                         iters=config['model']['iters'],
                                                                         ret_confmap=True)
        # loss computations
        loss_pose = supervised_pose_loss(pose_predictions, gt_pose)

        # debug
        print("\n se3 pose")
        print(f"gt_pose: {gt_pose[0].detach().cpu().numpy()}\npred_pose: {pose_predictions[0].detach().cpu().numpy()}")
        print(" SE3 pose")
        print(f"gt_pose: {pseudo_lie_se3_to_SE3(gt_pose[0]).detach().cpu().numpy()}\npred_pose: {pseudo_lie_se3_to_SE3(pose_predictions[0]).detach().cpu().numpy()}\n")
        print(" trans loss: ", loss_pose[:, 3:].detach().mean().cpu().item())
        print(" rot loss: ", loss_pose[:, :3].detach().mean().cpu().item())

        pose = pose_predictions.clone()
        pose[:,3:] *= config['depth_scale']
        fig, ax = plot_res(ref_img, trg_img, flow_predictions[-1], trg_depth*config['depth_scale'], pseudo_lie_se3_to_SE3_batch(-pose), conf1, conf2, intrinsics)
        plt.savefig(os.path.join(args.outpath, f'{i_batch}.png'))
        if device == torch.device('cpu'):
            plt.show()
        else:
            plt.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    import yaml
    parser.add_argument('--name', default='RAFT-poseEstimator', help="name your experiment")
    parser.add_argument('--outpath', default='output', help="output path")
    parser.add_argument('--log', action="store_true")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--config', help="yaml config file", default='../configuration/train_raft.yaml')
    parser.add_argument('--force_cpu', action="store_true")
    parser.add_argument('--dbg', action="store_true")
    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)
    with open(args.config, 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    main(args, config, args.force_cpu)
