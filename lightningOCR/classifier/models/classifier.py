import os
import torch
import torch.nn as nn

from lightningOCR.common import BaseLitModule
from lightningOCR.common import LIGHTNING_MODULE


@LIGHTNING_MODULE.register()
class Classifier(BaseLitModule):
    def __init__(
        self,
        data,
        strategy,
        architecture,
        loss,
        metric
    ):
        super(Classifier, self).__init__(
            data, strategy, architecture, loss, metric
        )
        assert self.metric is not None, 'metric should be Acc'
        self.register_buffer('train_corrects', torch.tensor(0.0))
        self.register_buffer('train_samples', torch.tensor(0.0))
        self.register_buffer('val_corrects', torch.tensor(0.0))
        self.register_buffer('val_samples', torch.tensor(0.0))

    def training_step(self, batch, batch_idx):
        x = batch['image']
        gt = batch['gt']
        pred = self.model(x)
        loss = self.loss(pred, gt)
        c, s = self.metric(pred, gt)

        # Log and Plot
        self.train_corrects += c
        self.train_samples += s
        self.log('acc/train', self.train_corrects / self.train_samples, prog_bar=True, logger=False)
        for j, para in enumerate(self.optimizers().param_groups):
            self.log(f'x/lr{j}', para['lr'], prog_bar=False, logger=True)
        if self.global_rank in [-1, 0] and self.global_step < 6 and hasattr(self.trainset, 'plot_batch'):
            # do plot
            os.makedirs(self.logger.log_dir, exist_ok=True)
            self.trainset.plot_batch(batch, os.path.join(self.logger.log_dir, f'train_batch_{self.global_step}.jpg'))

        return loss

    def training_epoch_end(self, training_step_outputs):
        train_corrects = self.all_gather(self.train_corrects)
        train_samples = self.all_gather(self.train_samples)
        self.log('acc/train', train_corrects.sum() / train_samples.sum(), prog_bar=False, logger=True)
        self.train_corrects.zero_()
        self.train_samples.zero_()

    def validation_step(self, batch, batch_idx):
        x = batch['image']
        gt = batch['gt']
        pred = self.model(x)
        c, s = self.metric(pred, gt)
        self.val_corrects += c
        self.val_samples += s

    def validation_epoch_end(self, val_step_outputs):
        val_corrects = self.all_gather(self.val_corrects)
        val_samples = self.all_gather(self.val_samples)
        self.log('acc/val', val_corrects.sum() / val_samples.sum(), prog_bar=True, logger=True, rank_zero_only=True)
        self.val_corrects.zero_()
        self.val_samples.zero_()