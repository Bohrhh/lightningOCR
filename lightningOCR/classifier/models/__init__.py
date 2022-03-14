import torch
import torch.nn as nn

from lightningOCR.common import BaseLitModule
from lightningOCR.common import LIGHTNING_MODULE, build_model, build_loss
from .resnet import *

@LIGHTNING_MODULE.register()
class Classifier(BaseLitModule):
    def __init__(self, strategy, architecture, loss_cfg):
        super(Classifier, self).__init__(strategy)
        self.model = build_model(architecture)
        self.loss = build_loss(loss_cfg)
        self.positives = 0
        self.samples = 0

    def acc(self, logits, labels):
        """
        Args:
            logits (Tensor): shape (N,classes)
            labels (Tensor): shape (N,)

        Returns:
            acc (np.float32): accuracy
        """
        _, pred_labels = torch.max(logits, dim=1)
        if len(labels.shape) > 1:
            _, labels = labels.max(dim=1)
        positive = (pred_labels == labels).float().sum().item()
        batch = len(pred_labels)
        return positive, batch

    def training_step(self, batch, batch_idx):
        x = batch['image']
        label = batch['label']
        logits = self.model(x)
        loss = self.loss(logits, label)
        p, s = self.acc(logits, label)
        self.samples += s
        self.positives += p
        self.log('acc', self.positives / self.samples, prog_bar=True, logger=False)
        return loss

    def training_epoch_end(self, training_step_outputs):
        self.log('acc', self.positives / self.samples, prog_bar=False, logger=True)
        self.positives = 0
        self.samples = 0

    def validation_step(self, batch, batch_idx):
        x = batch['image']
        label = batch['label']
        logits = self.model(x)
        p, s = self.acc(logits, label)
        return p, s

    def validation_epoch_end(self, val_step_outputs):
        positives = 0
        samples = 0
        for (p, s) in val_step_outputs:
            positives += p
            samples += s
        self.log('val_acc', positives / samples, prog_bar=True, logger=True)