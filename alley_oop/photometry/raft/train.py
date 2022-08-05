from __future__ import print_function, division
import sys
sys.path.append('core')

import argparse
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from alley_oop.photometry.raft.core.PoseN import RAFT, PoseN
import alley_oop.photometry.raft.core.datasets as datasets
from alley_oop.photometry.raft.losses import geometric_2d_loss, geometric_3d_loss, supervised_pose_loss, seq_loss

import wandb
from torch.cuda.amp import GradScaler

# exclude extremly large displacements
SUM_FREQ = 100
VAL_FREQ = 5000


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler
    

class Logger:
    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        #wandb.init(project='RAFT_PoseEstimator')

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        print(training_str + metrics_str)

        for k in self.running_loss:
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        wandb.log(results)

    def close(self):
        wandb.finish()


def train(args):
    loss_weights = {'pose':1.0, '2d':0.1, '3d':0.1}
    device = torch.device('cpu')
    model = PoseN(args)
    ref_model = nn.DataParallel(RAFT(args))
    print("Parameter Count: %d" % count_parameters(model))

    if args.restore_ckpt is not None:
        ref_model.load_state_dict(torch.load(args.restore_ckpt), strict=False)
        model.init_from_raft(args.restore_ckpt)

    #model =nn.DataParallel(model).to(device)
    model.train()
    #model.module.freeze_bn()
    model.freeze_bn()
    dataset, intrinsics = datasets.get_data(args.datapath, args.image_shape)
    train_loader = DataLoader(dataset, num_workers=1, pin_memory=True, batch_size=2)
    optimizer, scheduler = fetch_optimizer(args, model)

    total_steps = 0
    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(model, scheduler)

    VAL_FREQ = 5000

    should_keep_training = True
    while should_keep_training:

        for i_batch, data_blob in enumerate(train_loader):
            optimizer.zero_grad()
            ref_img, trg_img, ref_depth, trg_depth, valid, pose = [x.to(device)for x in data_blob]

            if args.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                ref_img = (ref_img + stdv * torch.randn(*ref_img.shape).to(device)).clamp(0.0, 255.0)
                trg_img = (trg_img + stdv * torch.randn(*trg_img.shape).to(device)).clamp(0.0, 255.0)

            flow_predictions, pose_predictions = model(ref_img, trg_img, iters=args.iters)

            loss2d = seq_loss(geometric_2d_loss, (flow_predictions, pose_predictions, intrinsics, trg_depth, valid,))
            loss3d = seq_loss(geometric_3d_loss, (flow_predictions, pose_predictions, intrinsics, trg_depth, ref_depth, valid,))
            loss_pose = seq_loss(supervised_pose_loss, (pose_predictions, pose))
            loss = loss_weights['pose']*loss_pose+loss_weights['2d']*loss2d+loss_weights['3d']*loss3d
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)                
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            metrics = {"loss2d": loss2d.detach().mean().cpu().item(),
                      "loss3d": loss3d.detach().mean().cpu().item(),
                      "loss_pose": loss_pose.detach().mean().cpu().item(),
                      "loss_total": loss.detach().mean().cpu().item()}
            
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            logger.push(metrics)

            if total_steps % VAL_FREQ == VAL_FREQ - 1:
                PATH = 'checkpoints/%d_%s.pth' % (total_steps+1, args.name)
                torch.save(model.state_dict(), PATH)

                results = {}
                logger.write_dict(results)
                
                model.train()
                if args.stage != 'chairs':
                    model.module.freeze_bn()
            
            total_steps += 1

            if total_steps > args.num_steps:
                should_keep_training = False
                break

    logger.close()
    PATH = 'checkpoints/%s.pth' % args.name
    torch.save(model.state_dict(), PATH)

    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training") 
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--validation', type=str, nargs='+')
    parser.add_argument('--datapath', type=str, default='/home/mhayoz/research/innosuisse_surgical_robot/01_Datasets/05_slam/intuitive_phantom/part0007')
    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_shape', type=int, nargs='+', default=[512, 640])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')
    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    train(args)