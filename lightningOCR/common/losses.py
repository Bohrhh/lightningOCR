import torch
import torch.nn as nn
import torch.nn.functional as F
from .registry import Registry

LOSSES = Registry('loss')

LOSSES.register(name='CrossEntropyLoss', obj=nn.CrossEntropyLoss)


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