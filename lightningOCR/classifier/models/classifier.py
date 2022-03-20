import torch
import torch.nn as nn

from lightningOCR.common import BaseLitModule
from lightningOCR.common import LIGHTNING_MODULE, build_model, build_loss
from .resnet import *


def get_acc(pred, gt):
    """
    Args:
        logits (Tensor): shape (N,classes)
        labels (Tensor): shape (N,)

    Returns:
        corrects : correct number
        batch_num : batch number
    """
    logits = pred['logits']
    labels = gt['targets']
    _, pred_labels = torch.max(logits, dim=1)
    if len(labels.shape) > 1:
        _, labels = labels.max(dim=1)
    corrects = (pred_labels == labels).float().sum().item()
    batch_num = len(pred_labels)
    return corrects, batch_num


@LIGHTNING_MODULE.register()
class Classifier(BaseLitModule):
    def __init__(self, data_cfg, strategy, architecture, loss_cfg):
        super(Classifier, self).__init__(data_cfg, strategy)
        self.model = build_model(architecture)
        self.loss = build_loss(loss_cfg)
        self.train_corrects = 0
        self.train_samples = 0

    def training_step(self, batch, batch_idx):
        x = batch['image']
        gt = batch['gt']
        pred = self.model(x)
        loss = self.loss(pred, gt)

        # Log and Plot
        c, s = get_acc(pred, gt)
        self.train_corrects += c
        self.train_samples += s
        self.log('acc/train', self.train_corrects / self.train_samples, prog_bar=True, logger=False)
        for j, para in enumerate(self.optimizers().param_groups):
            self.log(f'x/lr{j}', para['lr'], prog_bar=False, logger=True)
        if self.global_rank in [-1, 0] and self.global_step < 3:
            # do plot
            pass

        return loss

    def training_epoch_end(self, training_step_outputs):
        self.log('acc/train', self.train_corrects / self.train_samples, prog_bar=False, logger=True)
        self.train_corrects = 0
        self.train_samples = 0

    def validation_step(self, batch, batch_idx):
        x = batch['image']
        gt = batch['gt']
        pred = self.model(x)
        c, s = get_acc(pred, gt)
        return c, s

    def validation_epoch_end(self, val_step_outputs):
        val_step_outputs = self.all_gather(val_step_outputs)
        val_corrects = 0
        val_samples = 0
        for (c, s) in val_step_outputs:
            val_corrects += c
            val_samples += s
        self.log('acc/val', val_corrects / val_samples, prog_bar=True, logger=True, rank_zero_only=True)