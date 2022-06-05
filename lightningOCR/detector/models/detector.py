import os
import torch
import torch.nn as nn

from lightningOCR.common import BaseLitModule
from lightningOCR.common import LIGHTNING_MODULE


@LIGHTNING_MODULE.register()
class Detector(BaseLitModule):
    def __init__(
        self,
        data,
        strategy,
        architecture,
        loss,
        metric=None,
        postprocess=None,
    ):
        super(Detector, self).__init__(
            data, strategy, architecture, loss, metric, postprocess
        )
        pass