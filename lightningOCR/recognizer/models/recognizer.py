import torch
import torch.nn as nn

from lightningOCR.common import BaseLitModule
from lightningOCR.common import LIGHTNING_MODULE
from .crnn import CRNN


@LIGHTNING_MODULE.register()
class Recognizer(BaseLitModule):
    def __init__(self, data_cfg, strategy, architecture, loss_cfg, metric_loss):
        super(Recognizer, self).__init__(
            data_cfg, strategy, architecture, loss_cfg, metric_loss
        )

    def training_step(self, batch, batch_idx):
        x = batch['image']
        gt = batch['gt']
        pred = self.model(x)
        loss = self.loss(pred, gt)
        # metric = self.metric(pred, gt)

        # Log and Plot
        for j, para in enumerate(self.optimizers().param_groups):
            self.log(f'x/lr{j}', para['lr'], prog_bar=False, logger=True)
        if self.global_rank in [-1, 0] and self.global_step < 3:
            # do plot
            pass

        return loss

    # def training_epoch_end(self, training_step_outputs):
    #     self.log('acc/train', self.train_corrects / self.train_samples, prog_bar=False, logger=True)
    #     self.train_corrects = 0
    #     self.train_samples = 0

    # def validation_step(self, batch, batch_idx):
    #     x = batch['image']
    #     label = batch['label']
    #     logits = self.model(x)
    #     c, s = get_acc(logits, label)
    #     return c, s

    # def validation_step_end(self, val_step_outputs):
    #     if self.strategy['gpus'] <= 1:
    #         val_step_outputs = [val_step_outputs]
    #     for (c, s) in val_step_outputs:
    #         self.val_corrects += c
    #         self.val_samples += s

    # def validation_epoch_end(self, val_step_outputs):
    #     self.log('acc/val', self.val_corrects / self.val_samples, prog_bar=True, logger=True)