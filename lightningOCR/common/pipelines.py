import cv2
import math
import string
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
class TextLineResize(A.ImageOnlyTransform):
    def __init__(self, height, width, padding_value=0, always_apply=False, p=1.0):
        super(TextLineResize, self).__init__(always_apply, p)
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


class BaseRecLabelEncode(object):
    """ Convert between text-label and text-index """

    def __init__(self,
                 max_text_length,
                 character_dict_path=None,
                 character_type='ch',
                 use_space_char=False):
        support_character_type = [
            'ch', 'en', 'EN_symbol', 'french', 'german', 'japan', 'korean',
            'EN', 'it', 'xi', 'pu', 'ru', 'ar', 'ta', 'ug', 'fa', 'ur', 'rs',
            'oc', 'rsc', 'bg', 'uk', 'be', 'te', 'ka', 'chinese_cht', 'hi',
            'mr', 'ne', 'latin', 'arabic', 'cyrillic', 'devanagari'
        ]
        assert character_type in support_character_type, "Only {} are supported now but get {}".format(
            support_character_type, character_type)

        self.max_text_len = max_text_length
        self.beg_str = "sos"
        self.end_str = "eos"
        if character_type == "en":
            self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
            dict_character = list(self.character_str)
        elif character_type == "EN_symbol":
            # same with ASTER setting (use 94 char).
            self.character_str = string.printable[:-6]
            dict_character = list(self.character_str)
        elif character_type in support_character_type:
            self.character_str = ""
            assert character_dict_path is not None, "character_dict_path should not be None when character_type is {}".format(
                character_type)
            with open(character_dict_path, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode('utf-8').strip("\n").strip("\r\n")
                    self.character_str += line
            if use_space_char:
                self.character_str += " "
            dict_character = list(self.character_str)
        self.character_type = character_type
        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character

    def add_special_char(self, dict_character):
        return dict_character

    def encode(self, text):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]

        output:
            text: concatenated text index for CTCLoss.
                    [sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
            length: length of each text. [batch_size]
        """
        if len(text) == 0 or len(text) > self.max_text_len:
            return None
        if self.character_type == "en":
            text = text.lower()
        text_list = []
        for char in text:
            if char not in self.dict:
                # logger = get_logger()
                # logger.warning('{} is not in dict'.format(char))
                continue
            text_list.append(self.dict[char])
        if len(text_list) == 0:
            return None
        return text_list


@PIPELINES.register()
class CTCLabelEncode(BaseRecLabelEncode):
    """ Convert between text-label and text-index """

    def __init__(self,
                 max_text_length,
                 character_dict_path=None,
                 character_type='ch',
                 use_space_char=False,
                 **kwargs):
        super(CTCLabelEncode,
              self).__init__(max_text_length, character_dict_path,
                             character_type, use_space_char)

    def __call__(self, *args, force_apply=False, **kwargs):
        if args:
            raise KeyError("You have to pass data to augmentations as named arguments, for example: aug(image=image)")
        text = kwargs['target']
        text = self.encode(text)
        if text is None:
            return None
        kwargs['length'] = np.array(len(text))
        text = text + [0] * (self.max_text_len - len(text))
        kwargs['target'] = np.array(text)
        return kwargs

    def add_special_char(self, dict_character):
        dict_character = ['blank'] + dict_character
        return dict_character