from typing import Any, List

import torch
from torchvision import transforms

from src.metrics import d1, photo_loss, lidar_loss, ds_loss, lr_loss
from src.models.modules.psmnet.stacked_hourglass import PSMNet
from src.callbacks.wandb_callbacks import LogDispPredictions


class PSMNetModel(nn.Module):
    """
    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        maxdisp: int = 128,
        lr: float = 0.001,
        lr_decay: float = 0,
        lr_period: list = [200, 300, 400],
        lr_mode: str = 'lambda',
        loadmodel: str = "",
        weight_decay: float = 0.0005,
        beta_lo: float = 0.9,
        beta_hi: float = 0.999,
    ):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        self.model = PSMNet(self.hparams)

        if self.hparams.loadmodel is not None:
            state_dict = torch.load(self.hparams.loadmodel)#, self.device
            try:
                self.model.load_state_dict(state_dict['state_dict'])
            except RuntimeError:
                #trunc_keys = [('.').join(key.split('.')[1:]) for key in state_dict['state_dict'].keys()]
                new_dict = self.model.state_dict()
                for key, val in zip(self.model.state_dict().keys(), state_dict['state_dict'].values()):
                    new_dict[key] = val
                self.model.load_state_dict(new_dict)
                #self.model.module.load_state_dict(state_dict['state_dict'])

        # loss function
        self.criterion = torch.nn.functional.smooth_l1_loss

        # network input size
        self.th, self.tw = 256, 512

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.d1_loss = d1
        self.pm_loss = photo_loss
        self.li_loss = lidar_loss
        self.ds_loss = ds_loss

        #if not str(self.model.device).lower().__contains__('cpu'):
        #    self.model = torch.nn.DataParallel(self.model)
        #print('Model device is %s' % str(self.model.device))
        self.model = torch.nn.DataParallel(self.model)

        # user-defined member variables
        self.epoch_idx = 0

    def forward(self, limg_b: torch.Tensor, rimg_b: torch.Tensor, train_opt: bool = None):
        
        limg_b = self.resize2input(limg_b)
        rimg_b = self.resize2input(rimg_b)

        outs = self.model(limg_b, rimg_b, train_opt)

        return outs

    def step(self, batch: Any, train_opt: bool = None):

        limg_b, rimg_b, l_gt_b, _ = batch[:4]

        outs = self.forward(limg_b, rimg_b, train_opt=train_opt)
        
        l_gt_b = torch.squeeze(l_gt_b, 1)  # squeeze redundant dimension
        l_ds_b = self.resize_gt(l_gt_b)
        mask = l_ds_b > 0
        if train_opt: # distinguish between training (3 outputs) and test (1 output)
            out1 = torch.squeeze(outs[0], 1)
            out2 = torch.squeeze(outs[1], 1)
            outs = torch.squeeze(outs[2], 1)
            loss = 0.5*self.criterion(out1[mask], l_ds_b[mask], size_average=True) + \
                    0.7*self.criterion(out2[mask], l_ds_b[mask], size_average=True) + \
                    self.criterion(outs[mask], l_ds_b[mask], size_average=True) 
        else:
            outs = torch.squeeze(outs,1)
            loss = self.criterion(outs[mask], l_ds_b[mask], size_average=True)

        return loss, outs, l_ds_b

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch, train_opt=True)
        limg_b = self.resize2input(batch[0])
        rimg_b = self.resize2input(batch[1])

        # train metrics
        d1 = self.d1_loss(preds, targets)
        pm_b = self.pm_loss(rimg_b, preds, DSSIM_window=11, alpha=0.5, image_borders=0)
        li_b = self.li_loss(preds, targets[0])
        ds_b = self.ds_loss(limg_b, preds)
        
        # log train metrics
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/d1", d1, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/pm_loss", pm_b.mean(), on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/li_loss", li_b.mean(), on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/ds_loss", ds_b.mean(), on_step=False, on_epoch=True, prog_bar=False)

        # log learning rate
        try:
            lr = self.lr_schedulers().get_lr()[0]
        except AttributeError:
            lr = self.lr_schedulers().in_cooldown
        self.log("train/lr", float(lr), on_step=False, on_epoch=True, prog_bar=False)

        # we can return here dict with any tensors
        # and then read it in some callback or in training_epoch_end() below
        # remember to always return loss from training_step, or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        self.epoch_idx += 1


    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch, train_opt=False)
        limg_b = self.resize2input(batch[0])
        rimg_b = self.resize2input(batch[1])

        # val metrics
        d1 = self.d1_loss(preds, targets)
        pm_b = self.pm_loss(rimg_b, preds, DSSIM_window=11, alpha=0.5, image_borders=0)
        li_b = self.li_loss(preds, targets[0])
        ds_b = self.ds_loss(limg_b, preds)

        # log val metrics
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/d1", d1, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/pm_loss", pm_b.mean(), on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/li_loss", li_b.mean(), on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/ds_loss", ds_b.mean(), on_step=False, on_epoch=True, prog_bar=False)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch, train_opt=False)
        limg_b = self.resize2input(batch[0])
        rimg_b = self.resize2input(batch[1])
        
        # test metrics
        d1 = self.d1_loss(preds, targets)
        pm_b = self.pm_loss(rimg_b, preds, DSSIM_window=11, alpha=0.5, image_borders=0)
        li_b = self.li_loss(preds, targets[0])
        ds_b = self.ds_loss(limg_b, preds)
        
        # log test metrics
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/d1", d1, on_step=False, on_epoch=True)
        self.log("test/pm_loss", pm_b.mean(), on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/li_loss", li_b.mean(), on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/ds_loss", ds_b.mean(), on_step=False, on_epoch=True, prog_bar=False)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self, plateau_mode: bool = False):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        opt_adam = torch.optim.Adam(
            params=self.model.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta_lo, self.hparams.beta_hi)#weight_decay=self.hparams.weight_decay
        )

        if self.hparams.lr_mode == 'plateau':
            lrs_adam = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_adam, mode='min', factor=self.hparams.lr_decay,  patience=self.hparams.lr_period)
        elif self.hparams.lr_mode == 'multi':
            lrs_adam = torch.optim.lr_scheduler.MultiStepLR(opt_adam, milestones=list(self.hparams.lr_period), gamma=0.1)
        elif self.hparams.lr_mode == 'lambda':
            lambda_r = lambda epoch: 0.95 ** epoch
            lrs_adam = torch.optim.lr_scheduler.LambdaLR(opt_adam, lambda_r)

        return {'optimizer': opt_adam, 'lr_scheduler': {'scheduler': lrs_adam, 'monitor': 'val/loss'}}

    def configure_callbacks(self):

        log_disp = LogDispPredictions(num_samples=3)

        return [log_disp]#super().configure_callbacks()

    def resize2input(self, img):
        return transforms.Resize((self.th, self.tw)).__call__(img)

    def resize_gt(self, disp):
        
        width_scale = disp.shape[-1] / self.tw

        disp = transforms.Resize(size=(self.th, self.tw), interpolation=transforms.InterpolationMode.NEAREST).__call__(disp) / width_scale

        return disp