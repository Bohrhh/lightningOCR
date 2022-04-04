import torch
import torch.nn as nn
import torch.nn.functional as F
from .registry import Registry

LOSSES = Registry('loss')


@LOSSES.register()
class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.loss_fun = nn.CrossEntropyLoss()
    
    def forward(self, pred, gt):
        logits = pred['logits']
        targets = gt['target']
        loss = self.loss_fun(logits, targets)
        return loss


@LOSSES.register()
class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1, loss_weight=1.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.loss_weight = loss_weight

    def forward(self, logits, target):
        logprobs = F.log_softmax(logits, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        loss = loss.mean()*self.loss_weight
        return loss


@LOSSES.register()
class CTCLoss(nn.Module):
    def __init__(self):
        super(CTCLoss, self).__init__()
        self.loss_func = nn.CTCLoss(blank=0, reduction='mean')

    def forward(self, pred, gt):
        logits = pred['logits']
        targets = gt['target']
        target_lengths = gt['target_length']

        N, T, _ = logits.shape
        x = torch.log_softmax(logits, dim=2)
        x_for_loss = x.permute(1, 0, 2).contiguous()  # T * N * C
        x_lengths = torch.full(size=(N, ), fill_value=T, dtype=torch.long, device=logits.device)
        target_lengths = torch.clamp(target_lengths, min=1, max=T).long()

        loss = self.loss_func(x_for_loss, targets, x_lengths, target_lengths)
        return loss