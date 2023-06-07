import sys
sys.path.append('../')
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from lietorch import SE3
import wandb

from core.pose.pose_net import PoseNet
from core.utils.logging import TrainLogger as Logger
import dataset.train_datasets as datasets


SUM_FREQ = 100
VAL_FREQ = 1000


def supervised_pose_loss(pose_pred, pose_gt):
    assert pose_pred.shape == pose_gt.tangent_shape
    l1 = (pose_pred - pose_gt.log()).abs()
    return l1


def val(model, dataloader, device, logger, key):
    model.eval()
    with torch.no_grad():
        for i_batch, data_blob in enumerate(dataloader):
            ref_img, trg_img, ref_img_r, trg_img_r, ref_mask, trg_mask, gt_pose, intrinsics, baseline = [x.to(device) for x in
                                                                                              data_blob]
            gt_pose = SE3(gt_pose)
            pose_predictions, trg_depth, ref_depth, weights = model(trg_img, ref_img,
                                                                   intrinsics.float(), baseline.float(),
                                                                   image1r=trg_img_r,
                                                                   image2r=ref_img_r,
                                                                   mask1=trg_mask.to(torch.bool),
                                                                   mask2=ref_mask.to(torch.bool),
                                                                   ret_confmap=True)
            loss_pose = supervised_pose_loss(pose_predictions, gt_pose)
            loss = torch.nanmean(loss_pose)
            loss_cpu = loss_pose.detach().cpu()
            metrics = {f"{key}/loss_rot": loss_cpu[:, 3:].sum(dim=-1).nanmean().item(),
                       f"{key}/loss_trans": loss_cpu[:, :3].sum(dim=-1).nanmean().item(),
                       f"{key}/loss_total": loss_cpu.sum(dim=-1).nanmean().item()}
            logger.push(metrics, len(dataloader))
        logger.flush()
    model.train()
    return loss.detach().mean().cpu().item()


def main(args, config, force_cpu):
    # general
    config['model']['image_shape'] = config['image_shape']
    device = torch.device('cuda') if (torch.cuda.is_available() & (not force_cpu)) else torch.device('cpu')

    # get data
    data_train = datasets.get_data(config['data']['train'], config['image_shape'], config['depth_scale'])
    data_val = datasets.get_data(config['data']['val'], config['image_shape'], config['depth_scale'])
    print(f"train: {len(data_train)} samples, val: {len(data_val)} samples")
    train_loader = DataLoader(data_train, num_workers=4, pin_memory=True, batch_size=config['train']['batch_size'], shuffle=True)
    val_loader = DataLoader(data_val, num_workers=4, pin_memory=True, batch_size=config['val']['batch_size'])

    # get model
    model = PoseNet(config['model'])
    model = model.init_from_raft(config['model']['pretrained'])
    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt)['state_dict'], strict=False)

    model.train()
    model = model.to(device)
    model.freeze_flow()
    parallel = False
    if (device != torch.device('cpu')) & (torch.cuda.device_count() > 1):
        parallel = True
        model = nn.DataParallel(model).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config['train']['learning_rate'],
                            weight_decay=config['train']['weight_decay'],
                            eps=config['train']['epsilon'])

    # training loop
    total_steps = 0
    scaler = GradScaler()
    logger = Logger(model, config, args.name, args.log)
    if args.log:
        args.outpath = wandb.run.dir
    if not os.path.isdir(args.outpath):
        os.mkdir(args.outpath)

    should_keep_training = True
    best_loss = 1000000.0
    while should_keep_training:

        for i_batch, data_blob in enumerate(train_loader):
            if i_batch == config['train']['freeze_flow_steps']:
                model.module.freeze_flow(False) if parallel else model.freeze_flow(False)
            optimizer.zero_grad()
            ref_img, trg_img, ref_img_r, trg_img_r, ref_mask, trg_mask, gt_pose, intrinsics, baseline = [
                x.to(device) for x in
                data_blob]
            gt_pose = SE3(gt_pose)
            # forward pass
            pose_predictions, trg_depth, ref_depth, weights = model(trg_img, ref_img,
                                                                   intrinsics.float(), baseline.float(),
                                                                   image1r=trg_img_r,
                                                                   image2r=ref_img_r,
                                                                   mask1=trg_mask.to(torch.bool),
                                                                   mask2=ref_mask.to(torch.bool),
                                                                   ret_confmap=True)
            # loss computations
            loss_pose = supervised_pose_loss(pose_predictions, gt_pose)
            loss = torch.mean(loss_pose)

            # update params
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)                
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['train']['grad_clip'])
            loss_cpu = loss_pose.detach().cpu()
            loss_tran = loss_cpu[:, :3].sum(dim=-1).mean().item()
            loss_rot = loss_cpu[:, 3:].sum(dim=-1).mean().item()
            # debug
            if args.dbg & (i_batch % SUM_FREQ == 0):
                print("\n se3 pose")
                print(f"gt_pose: {gt_pose[0].detach().cpu().numpy()}\npred_pose: {pose_predictions[0].detach().cpu().numpy()}")
                print(" trans loss: ", loss_tran)
                print(" rot loss: ", loss_rot)

            metrics = {"train/loss_rot": loss_rot,
                      "train/loss_trans": loss_tran,
                      "train/loss_total": loss_cpu.sum(dim=-1).mean().item()}

            scaler.step(optimizer)
            scaler.update()

            logger.push(metrics, SUM_FREQ)
            if total_steps % SUM_FREQ == SUM_FREQ - 1:
                logger.flush()

            if (total_steps % VAL_FREQ) == 0:
                val_loss = val(model, val_loader, device, logger, 'val')
                if torch.isnan(torch.tensor(val_loss)):
                    should_keep_training = False
                    break
                if val_loss < best_loss:
                    best_loss = val_loss
                    path = os.path.join(args.outpath, f'{args.name}.pth')
                    torch.save({"state_dict": model.state_dict(), "config": config}, path)
                    logger.save_model(path)
                path = os.path.join(args.outpath, f'{args.name}_last.pth')
                torch.save({"state_dict": model.state_dict(), "config": config}, path)
            total_steps += 1

            if total_steps > config['train']['epochs']:
                should_keep_training = False
                break

    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    import yaml
    parser.add_argument('--name', default='RAFT-poseEstimator', help="name your experiment")
    parser.add_argument('--outpath', default='output', help="output path")
    parser.add_argument('--log', action="store_true")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--config', help="yaml config file", default='../configuration/train.yaml')
    parser.add_argument('--force_cpu', action="store_true")
    parser.add_argument('--dbg', action="store_true")
    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)
    with open(args.config, 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    main(args, config, args.force_cpu)
