import numpy as np
import collections
import albumentations as A
from torch.utils.data import Dataset
from abc import ABCMeta, abstractmethod

from .registry import build_from_cfg
from .pipelines import PIPELINES
from .litmodel import build_postprocess


def build_pipeline(cfg, default_args=None):
    pipeline = build_from_cfg(cfg, PIPELINES, default_args)
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


class BaseDataset(Dataset, metaclass=ABCMeta):
    def __init__(self, pipeline, postprocess):
        super(BaseDataset, self).__init__()
        self.pipeline  = Compose(pipeline['transforms'], 
                                 pipeline.get('bbox_params'), 
                                 pipeline.get('keypoint_params'),
                                 pipeline.get('additional_targets'))
        self.postprocess = None if postprocess is None else build_postprocess(postprocess)
        
        if 'Normalize' in [i.type for i in pipeline['transforms']]:
            idx = [i.type for i in pipeline['transforms']].index('Normalize')
            self.mean = self.pipeline.transforms[idx].mean
            self.std  = self.pipeline.transforms[idx].std
        else:
            self.mean = 0
            self.std  = 1

    @abstractmethod
    def load_data(self, idx):
        """
        Args:
            idx
        Return:
            results (dict): {'data1'    :np.ndarray, 
                             'data2'    :np.ndarray} 
        """
        pass

    def before_pipeline(self, results):
        """HOOK for before pipeline 
        """
        return results

    def after_pipeline(self, results):
        """HOOK for after pipeline 
        """
        return results

    @abstractmethod
    def gather(self, results):
        """Gather results elements to inputs and gt
        Return:
            results: {
                'image': value
                'gt'    : {item1: value1, item2: value2}
                'other' : {item1: value1, item2: value2}
            }
        """
        pass

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            results: {
                'image': value
                'gt'    : {item1: value1, item2: value2}
                'other' : {item1: value1, item2: value2}
            }
        """

        results  = self.load_data(idx)
        results  = self.before_pipeline(results)
        try:
            results  = self.pipeline(results)
            results  = self.after_pipeline(results)
            results  = self.gather(results)
        except:
            results = None

        if results is None:
            return self.__getitem__(np.random.randint(self.__len__()))
        return results