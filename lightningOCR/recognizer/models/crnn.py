import torch
import torch.nn as nn
import torch.nn.functional as F

from lightningOCR.common import ARCHITECTURES
from .modules import MobileNetV1Enhance, SequenceEncoder, CTCHead


@ARCHITECTURES.register()
class CRNN(nn.Module):
    """CTC-loss based recognizer.
    
    Shape:
        x (Tensor): shape (N, C, H, W)
    
    Return:
        results (Dict): {'logits': Tensor of shape (N, T, D) e.g. (64, 80, 6625)
                         'feats': Tensor of shape (N, T, C) e.g. (64, 80, 96)} 
    """
    def __init__(self, return_feats=False):
        super(CRNN, self).__init__()
        self.backbone = MobileNetV1Enhance(scale=0.5)
        self.neck = SequenceEncoder(self.backbone.out_channels, encoder_type='rnn', hidden_size=64)
        self.head = CTCHead(self.neck.out_channels, mid_channels=96, return_feats=return_feats)
        self.return_feats = return_feats

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)

        if self.return_feats:
            feats, logits = self.head(x)
            return {'logits': logits, 'feats': feats}
        else:
            logits = self.head(x)
            return {'logits': logits}
