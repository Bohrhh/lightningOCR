import cv2
import math
import random
import numpy as np
import albumentations as A
from .registry import Registry

PIPELINES = Registry('pipeline')


PIPELINES.register(name='Normalize', obj=A.Normalize)


@PIPELINES.register()
class ClsRotate180(A.BasicTransform):

    @property
    def targets(self):
        return {"image": self.apply,
                "label": self.apply_to_label}
    
    def apply(self, image, **params):
        return cv2.rotate(image, cv2.ROTATE_180)

    def apply_to_label(self, label, **params):
        return 1


@PIPELINES.register()
class ClsResize(A.ImageOnlyTransform):
    def __init__(self, height, width, padding_value=0, always_apply=False, p=1.0):
        super(ClsResize, self).__init__(always_apply, p)
        self.height = height
        self.width = width
        self.padding_value = padding_value

    def apply(self, image, **params):
        h, w = image.shape[:2]
        ratio = w / float(h)
        if math.ceil(self.height * ratio) > self.width:
            resized_w = self.width
        else:
            resized_w = int(math.ceil(self.height * ratio))
        resized_image = cv2.resize(image, (resized_w, self.height))

        if len(image.shape) == 2:
            padding_im = np.zeros((self.height, self.width), dtype=image.dtype)
        else:
            padding_im = np.zeros((self.height, self.width, image.shape[2]), dtype=image.dtype)
        padding_im[:, 0:resized_w] = resized_image
        return padding_im