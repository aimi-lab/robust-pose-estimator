import sys
sys.path.append('../')
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from alley_oop.photometry.raft.core.PoseN import RAFT, PoseN
import alley_oop.photometry.raft.core.datasets as datasets
from alley_oop.photometry.raft.losses import geometric_2d_loss, geometric_3d_loss, supervised_pose_loss, seq_loss

import wandb
from torch.cuda.amp import GradScaler

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
    

class Logger:
    def __init__(self, model, scheduler, config, project_name, log):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.log = log
        if log:
            wandb.init(project=project_name, config=config)

    def _print_training_status(self):
        metrics_data = [self.running_loss[k] for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        print(training_str + metrics_str)

        for k in self.running_loss:
            self.running_loss[k] = 0.0

    def push(self, metrics, freq):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]/freq

    def write_dict(self, results):
        wandb.log(results)

    def flush(self):
        if self.log:
            self.write_dict(self.running_loss)
        self._print_training_status()
        self.running_loss = {}

    def close(self):
        wandb.finish()


def val(model, dataloader, device, loss_weights, intrinsics, logger):
    model.eval()
    with torch.no_grad():
        for i_batch, data_blob in enumerate(dataloader):
            ref_img, trg_img, ref_depth, trg_depth, valid, pose = [x.to(device) for x in data_blob]
            flow_predictions, pose_predictions = model(ref_img, trg_img, iters=config['model']['iters'])

            loss2d = seq_loss(geometric_2d_loss, (flow_predictions, pose_predictions, intrinsics, trg_depth, valid,))
            loss3d = seq_loss(geometric_3d_loss,
                              (flow_predictions, pose_predictions, intrinsics, trg_depth, ref_depth, valid,))
            loss_pose = seq_loss(supervised_pose_loss, (pose_predictions, pose))
            loss = loss_weights['pose'] * loss_pose + loss_weights['2d'] * loss2d + loss_weights['3d'] * loss3d

            metrics = {"val/loss2d": loss2d.detach().mean().cpu().item(),
                       "val/loss3d": loss3d.detach().mean().cpu().item(),
                       "val/loss_pose": loss_pose.detach().mean().cpu().item(),
                       "val/loss_total": loss.detach().mean().cpu().item()}
            logger.push(metrics, len(dataloader))
        logger.flush()
    model.train()


def main(args, config):
    # general
    loss_weights = config['train']['loss_weights']
    config['model']['image_shape'] = config['image_shape']
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # get model
    model = PoseN(config['model'])
    model.init_from_raft(config['model']['pretrained'])
    #ref_model = nn.DataParallel(RAFT(config['model']))
    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt), strict=False)

    model =nn.DataParallel(model).to(device)
    model.train()
    model.module.freeze_bn()

    # get data
    data_train, intrinsics = datasets.get_data(config['data']['train']['basepath'],config['data']['train']['sequences'], config['image_shape'])
    data_val, intrinsics = datasets.get_data(config['data']['val']['basepath'],config['data']['val']['sequences'], config['image_shape'])
    print(f"train: {len(data_train)} samples, val: {len(data_val)} samples")
    intrinsics = intrinsics.to(device)
    train_loader = DataLoader(data_train, num_workers=4, pin_memory=True, batch_size=config['train']['batch_size'], shuffle=True)
    val_loader = DataLoader(data_val, num_workers=4, pin_memory=True, batch_size=config['val']['batch_size'])
    optimizer, scheduler = fetch_optimizer(config['train'], model)

    total_steps = 0
    scaler = GradScaler()
    logger = Logger(model, scheduler, config, args.name, args.log)

    should_keep_training = True
    while should_keep_training:

        for i_batch, data_blob in enumerate(train_loader):
            optimizer.zero_grad()
            ref_img, trg_img, ref_depth, trg_depth, valid, pose = [x.to(device)for x in data_blob]

            if config['train']['add_noise']:
                stdv = np.random.uniform(0.0, 5.0)
                ref_img = (ref_img + stdv * torch.randn(*ref_img.shape).to(device)).clamp(0.0, 255.0)
                trg_img = (trg_img + stdv * torch.randn(*trg_img.shape).to(device)).clamp(0.0, 255.0)

            flow_predictions, pose_predictions = model(ref_img, trg_img, ref_depth, trg_depth,
                                                       iters=config['model']['iters'])

            loss2d = seq_loss(geometric_2d_loss, (flow_predictions, pose_predictions, intrinsics, trg_depth, valid,))
            loss3d = seq_loss(geometric_3d_loss, (flow_predictions, pose_predictions, intrinsics, trg_depth, ref_depth, valid,))
            loss_pose = seq_loss(supervised_pose_loss, (pose_predictions, pose))
            loss = loss_weights['pose']*loss_pose+loss_weights['2d']*loss2d+loss_weights['3d']*loss3d
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)                
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['train']['grad_clip'])
            metrics = {"train/loss2d": loss2d.detach().mean().cpu().item(),
                      "train/loss3d": loss3d.detach().mean().cpu().item(),
                      "train/loss_pose": loss_pose.detach().mean().cpu().item(),
                      "train/loss_total": loss.detach().mean().cpu().item()}
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            logger.push(metrics, SUM_FREQ)
            if total_steps % SUM_FREQ == SUM_FREQ - 1:
                logger.flush()

            if total_steps % VAL_FREQ == VAL_FREQ - 1:
                val(model, val_loader, device, loss_weights, intrinsics, logger)

                PATH = 'checkpoints/%d_%s.pth' % (total_steps+1, args.name)
                torch.save(model.state_dict(), PATH)
            total_steps += 1

            if total_steps > config['train']['epochs']:
                should_keep_training = False
                break

    logger.close()
    PATH = 'checkpoints/%s.pth' % args.name
    torch.save(model.state_dict(), PATH)

    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    import yaml
    parser.add_argument('--name', default='RAFT-poseEstimator', help="name your experiment")
    parser.add_argument('--log', action="store_true")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--config', help="yaml config file", default='../configuration/train_raft.yaml')
    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)
    with open(args.config, 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')
    main(args, config)