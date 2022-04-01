import difflib
import torch
import torch.nn as nn
import torch.nn.functional as F
from .registry import Registry

METRICS = Registry('metric')


@METRICS.register()
class Acc(nn.Module):
    """
    Args:
        pred: {'logits': Tensor}, tensor shape (N,classes)
        gt: {'targets': Tensor} tensor shape (N,)

    Returns:
        corrects : correct number
        batch_num : batch number
    """
    def __init__(self):
        super(Acc, self).__init__()

    def forward(self, pred, gt):
        logits = pred['logits']
        labels = gt['targets']
        _, pred_labels = torch.max(logits, dim=1)
        if len(labels.shape) > 1:
            _, labels = labels.max(dim=1)
        corrects = (pred_labels == labels).float().sum().item()
        batch_num = len(pred_labels)
        return corrects, batch_num


@METRICS.register()
class RecAcc(nn.Module):
    """
    Args:
        pred: {'text': List of text}
        gt: {'text': List of text}

    Returns:
        corrects : correct number
        batch_num : batch number
    """
    def __init__(self):
        super(RecAcc, self).__init__()

    def forward(self, pred, gt):
        pred_text = pred['text']
        gt_text = gt['text']
        assert len(pred_text) == len(gt_text)
        batch_num = len(pred_text)
        corrects = 0
        for (i, j) in zip(pred_text, gt_text):
            if i == j:
                corrects += 1
        return corrects, batch_num


@METRICS.register()
class RecF1(nn.Module):
    """
    Args:
        pred: {'text': List of text}
        gt: {'text': List of text}

    Returns:
        match_chars
        gt_chars
        pred_chars
    """
    def __init__(self):
        super(RecF1, self).__init__()

    def forward(self, pred, gt):
        pred_text = pred['text']
        gt_text = gt['text']
        assert len(pred_text) == len(gt_text)

        match_chars = 0
        gt_chars = 0 
        pred_chars = 0

        for (i, j) in zip(gt_text, pred_text):
            matchs = difflib.SequenceMatcher(a=i, b=j).get_matching_blocks()
            match_chars += sum([k.size for k in matchs])
            gt_chars += len(i)
            pred_chars += len(j)
        return match_chars, gt_chars, pred_chars