import collections
import albumentations as A

from .registry import build_from_cfg
from .pipelines import PIPELINES


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


