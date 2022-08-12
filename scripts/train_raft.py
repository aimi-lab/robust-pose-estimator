import sys
sys.path.append('../')
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler


from alley_oop.photometry.raft.core.PoseN import RAFT, PoseN
import alley_oop.photometry.raft.core.datasets as datasets
from alley_oop.photometry.raft.losses import geometric_2d_loss, geometric_3d_loss, supervised_pose_loss, seq_loss, l1_loss
from alley_oop.photometry.raft.utils.logger import Logger
from alley_oop.photometry.raft.utils.plotting import plot_res, plot_3d

import wandb
from torch.cuda.amp import GradScaler

from alley_oop.geometry.lie_3d import lie_se3_to_SE3, lie_se3_to_SE3_batch


# exclude extremly large displacements
SUM_FREQ = 100
VAL_FREQ = 1000

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(config, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'], eps=config['epsilon'])

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, config['learning_rate'], config['epochs']+100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler
    

def val(model, dataloader, device, loss_weights, intrinsics, logger):
    model.eval()
    with torch.no_grad():
        for i_batch, data_blob in enumerate(dataloader):
            ref_img, trg_img, ref_depth, trg_depth, ref_conf, trg_conf, valid, pose = [x.to(device) for x in data_blob]
            flow_predictions, pose_predictions = model(trg_img, ref_img, trg_depth/config['depth_scale'], ref_depth/config['depth_scale'], trg_conf, ref_conf,
                                                       iters=config['model']['iters'])

            ref_depth, trg_depth, ref_conf, trg_conf = [dataloader.dataset.resize_lowres(d) for d in
                                                        [ref_depth, trg_depth, ref_conf, trg_conf]]
            loss2d = geometric_2d_loss(flow_predictions[-1], pose_predictions[-1], intrinsics, trg_depth, trg_conf,
                                       valid)
            loss3d = geometric_3d_loss(flow_predictions[-1], pose_predictions[-1], intrinsics, trg_depth, ref_depth,
                                       trg_conf, ref_conf, valid)
            loss_pose = seq_loss(supervised_pose_loss, (pose_predictions, pose))
            loss = loss_weights['pose'] * loss_pose + loss_weights['2d'] * loss2d + loss_weights['3d'] * loss3d

            metrics = {"val/loss2d": loss2d.detach().mean().cpu().item(),
                       "val/loss3d": loss3d.detach().mean().cpu().item(),
                       "val/loss_pose": loss_pose.detach().mean().cpu().item(),
                       "val/loss_total": loss.detach().mean().cpu().item()}
            logger.push(metrics, len(dataloader))
        logger.flush()
        logger.log_plot(plot_res(ref_img, trg_img, flow_predictions[-1], trg_depth, lie_se3_to_SE3_batch(-pose), intrinsics)[0])
    model.train()
    return loss.detach().mean().cpu().item()


def main(args, config, force_cpu):
    # general
    loss_weights = config['train']['loss_weights']
    config['model']['image_shape'] = config['image_shape']
    device = torch.device('cuda') if (torch.cuda.is_available() & (not force_cpu)) else torch.device('cpu')

    # get data
    data_train, intrinsics = datasets.get_data(config['data']['train']['basepath'],config['data']['train']['sequences'],
                                               config['image_shape'], step=config['data']['train']['step'])
    data_val, intrinsics = datasets.get_data(config['data']['val']['basepath'],config['data']['val']['sequences'],
                                             config['image_shape'], step=config['data']['val']['step'])
    print(f"train: {len(data_train)} samples, val: {len(data_val)} samples")
    intrinsics = intrinsics.to(device)
    train_loader = DataLoader(data_train, num_workers=4, pin_memory=True, batch_size=config['train']['batch_size'], shuffle=True)
    val_loader = DataLoader(data_val, num_workers=4, pin_memory=True, batch_size=config['val']['batch_size'],
                            sampler=SubsetRandomSampler(torch.from_numpy(np.random.choice(len(data_val), size=(400,), replace=False))))

    # get model
    model = PoseN(config['model'], intrinsics)
    model, ref_model = model.init_from_raft(config['model']['pretrained'])
    ref_model = ref_model.to(device)
    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt)['state_dict'], strict=False)

    model.train()
    model = model.to(device)
    model.freeze_bn()
    model.freeze_flow()
    parallel = False
    if (device != torch.device('cpu')) & (torch.cuda.device_count() > 1):
        parallel = True
        model = nn.DataParallel(model).to(device)
    optimizer, scheduler = fetch_optimizer(config['train'], model)

    # training loop
    total_steps = 0
    scaler = GradScaler()
    logger = Logger(model, scheduler, config, args.name, args.log)
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
            ref_img, trg_img, ref_depth, trg_depth, ref_conf, trg_conf, valid, pose = [x.to(device)for x in data_blob]
            if config['train']['add_noise']:
                stdv = np.random.uniform(0.0, 5.0)
                ref_img = (ref_img + stdv * torch.randn(*ref_img.shape).to(device)).clamp(0.0, 255.0)
                trg_img = (trg_img + stdv * torch.randn(*trg_img.shape).to(device)).clamp(0.0, 255.0)

            # forward pass
            flow_predictions, pose_predictions = model(trg_img, ref_img, trg_depth/config['depth_scale'],
                                                       ref_depth/config['depth_scale'], trg_conf, ref_conf,
                                                       iters=config['model']['iters'])
            # prepare data for loss computaitons
            ref_depth, trg_depth, ref_conf, trg_conf = [train_loader.dataset.resize_lowres(d) for d in
                                                        [ref_depth, trg_depth, ref_conf, trg_conf]]

            with torch.inference_mode():
                ref_flow = ref_model(trg_img, ref_img, iters=config['model']['iters'])

            # loss computations
            loss_flow = seq_loss(l1_loss, (flow_predictions, ref_flow,))
            loss2d = geometric_2d_loss(flow_predictions[-1], pose_predictions[-1], intrinsics, trg_depth, trg_conf,
                                       valid)
            loss3d = geometric_3d_loss(flow_predictions[-1], pose_predictions[-1], intrinsics, trg_depth, ref_depth,
                                       trg_conf, ref_conf, valid)
            loss_pose = seq_loss(supervised_pose_loss, (pose_predictions, pose))
            loss = loss_weights['pose']*loss_pose+loss_weights['2d']*loss2d+loss_weights['3d']*loss3d + loss_weights['flow']*loss_flow

            # debug
            if args.dbg & (i_batch%SUM_FREQ == 0):
                print("\n se3 pose")
                print(f"gt_pose: {pose[0].detach().cpu().numpy()}\npred_pose: {pose_predictions[-1][0].detach().cpu().numpy()}")
                print(" SE3 pose")
                print(f"gt_pose: {lie_se3_to_SE3(pose[0]).detach().cpu().numpy()}\npred_pose: {lie_se3_to_SE3(pose_predictions[-1][0]).detach().cpu().numpy()}")
                print(" pose loss: ", loss_pose.detach().mean().cpu().item())
                print(" 2d loss: ", loss2d.detach().mean().cpu().item())
                print(" 3d loss: ", loss3d.detach().mean().cpu().item())
                print(" flow loss: ", loss_flow.detach().mean().cpu().item())
                if device == torch.device('cpu'):
                    fig, ax = plot_res(ref_img, trg_img, flow_predictions[-1], trg_depth, lie_se3_to_SE3_batch(-pose), intrinsics)
                    import matplotlib.pyplot as plt
                    plt.show()
                    plot_3d(ref_img, trg_img, ref_depth, trg_depth, lie_se3_to_SE3_batch(pose), intrinsics)
            # update params
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)                
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['train']['grad_clip'])
            metrics = {"train/loss2d": loss2d.detach().mean().cpu().item(),
                      "train/loss3d": loss3d.detach().mean().cpu().item(),
                      "train/loss_pose": loss_pose.detach().mean().cpu().item(),
                      "train/loss_flow": loss_flow.detach().mean().cpu().item(),
                      "train/loss_total": loss.detach().mean().cpu().item()}

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            logger.push(metrics, SUM_FREQ)
            if total_steps % SUM_FREQ == SUM_FREQ - 1:
                logger.flush()

            if total_steps % VAL_FREQ == VAL_FREQ - 1:
                val_loss = val(model, val_loader, device, loss_weights, intrinsics, logger)
                if val_loss < best_loss:
                    best_loss = val_loss
                    path = os.path.join(args.outpath, f'{args.name}.pth')
                    torch.save({"state_dict": model.state_dict(), "config": config}, path)
                    logger.save_model(path)
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
    parser.add_argument('--config', help="yaml config file", default='../configuration/train_raft.yaml')
    parser.add_argument('--force_cpu', action="store_true")
    parser.add_argument('--dbg', action="store_true")
    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)
    with open(args.config, 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    main(args, config, args.force_cpu)