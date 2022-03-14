import os
import torch
import collections
import albumentations as A
import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset

from .registry import Registry, build_from_cfg
from .pipelines import PIPELINES

DATASETS = Registry('dataset')


def build_pipeline(cfg, default_args=None):
    pipeline = build_from_cfg(cfg, PIPELINES, default_args)
    return pipeline


def build_dataset(cfg, default_args=None):
    pipeline = build_from_cfg(cfg, DATASETS, default_args)
    return pipeline


class Compose(object):
    """Compose multiple transforms sequentially.

    Args:
        transforms (Sequence[dict | callable]): Sequence of transform object or
            config dict to be composed.
    """

    def __init__(self, transforms, bbox_params=None, keypoint_params=None, additional_targets=None):
        assert isinstance(transforms, collections.abc.Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_pipeline(transform)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict')

        self.compose = A.Compose(
            self.transforms, 
            bbox_params=None if bbox_params is None else A.BboxParams(**bbox_params), 
            keypoint_params=None if keypoint_params is None else A.KeypointParams(**keypoint_params),
            additional_targets=additional_targets)

    def __call__(self, data):
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
           dict: Transformed data.
        """

        return self.compose(**data)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string


class BaseDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.batch_size_per_gpu
        self.num_workers = min(cfg.workers_per_gpu,
                               os.cpu_count() // max(torch.cuda.device_count(),1),
                               self.batch_size if self.batch_size > 1 else 0)
        self.pin_memory = cfg.pin_memory

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.trainset = build_dataset(self.cfg.train)
            self.valset = build_dataset(self.cfg.val)
            self.trainset_size = len(self.trainset)
            self.valset_size = len(self.valset)
        if stage == 'test' or stage is None:
            self.testset = build_dataset(self.cfg.test)

    def train_dataloader(self):
        return DataLoader(self.trainset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.valset,
                          batch_size=2*self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.testset,
                          batch_size=2*self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory)