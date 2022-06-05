import cv2
import lmdb
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from lightningOCR.common import Compose, DATASETS, BaseDataset

matplotlib.use('Agg')  # for writing to files only


@DATASETS.register()
class DetDataset(BaseDataset):
    """Dataset for detection.

    An example of file structure is as followed.

    data folder organization
    ${data_root}
    ├── images/
    ├── labels/

    Args:
        data_root (str or list of str): data root or list of data roots
        pipeline (list[dict]): Processing pipeline
        length (int or None): length of dataset
    """

    def __init__(self,
                 data_root,
                 pipeline,
                 length=None,
                 postprocess=None):
        super(DetDataset, self).__init__(pipeline, postprocess)
        pass
    
