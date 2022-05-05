import cv2
import math
import string
import random
import numpy as np
import albumentations as A
from .registry import Registry

PIPELINES = Registry('pipeline')


PIPELINES.register(name='Normalize', obj=A.Normalize)
PIPELINES.register(name='ColorJitter', obj=A.ColorJitter)
PIPELINES.register(name='GaussianBlur', obj=A.GaussianBlur)
PIPELINES.register(name='GaussNoise', obj=A.GaussNoise)


@PIPELINES.register()
class ClsRotate180(A.BasicTransform):

    @property
    def targets(self):
        return {"image": self.apply,
                "target": self.apply_to_target}
    
    def apply(self, image, **params):
        return cv2.rotate(image, cv2.ROTATE_180)

    def apply_to_target(self, target, **params):
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
                 min_text_length=1,
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
        self.min_text_len = min_text_length
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
        if len(text) < self.min_text_len or len(text) > self.max_text_len:
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
                 min_text_length=1,
                 character_dict_path=None,
                 character_type='ch',
                 use_space_char=False,
                 **kwargs):
        super(CTCLabelEncode,
              self).__init__(max_text_length, min_text_length, character_dict_path,
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

        # for loss ace
        label = [0] * len(self.character)
        for x in text:
            label[x] += 1
        kwargs['target_ace'] = np.array(label)

        return kwargs

    def add_special_char(self, dict_character):
        dict_character = ['blank'] + dict_character
        return dict_character


class WarpMLS:
    def __init__(self, src, src_pts, dst_pts, dst_w, dst_h, trans_ratio=1.):
        self.src = src
        self.src_pts = src_pts
        self.dst_pts = dst_pts
        self.pt_count = len(self.dst_pts)
        self.dst_w = dst_w
        self.dst_h = dst_h
        self.trans_ratio = trans_ratio
        self.grid_size = 100
        self.rdx = np.zeros((self.dst_h, self.dst_w))
        self.rdy = np.zeros((self.dst_h, self.dst_w))

    @staticmethod
    def __bilinear_interp(x, y, v11, v12, v21, v22):
        return (v11 * (1 - y) + v12 * y) * (1 - x) + (v21 *
                                                      (1 - y) + v22 * y) * x

    def generate(self):
        self.calc_delta()
        return self.gen_img()

    def calc_delta(self):
        w = np.zeros(self.pt_count, dtype=np.float32)

        if self.pt_count < 2:
            return

        i = 0
        while 1:
            if self.dst_w <= i < self.dst_w + self.grid_size - 1:
                i = self.dst_w - 1
            elif i >= self.dst_w:
                break

            j = 0
            while 1:
                if self.dst_h <= j < self.dst_h + self.grid_size - 1:
                    j = self.dst_h - 1
                elif j >= self.dst_h:
                    break

                sw = 0
                swp = np.zeros(2, dtype=np.float32)
                swq = np.zeros(2, dtype=np.float32)
                new_pt = np.zeros(2, dtype=np.float32)
                cur_pt = np.array([i, j], dtype=np.float32)

                k = 0
                for k in range(self.pt_count):
                    if i == self.dst_pts[k][0] and j == self.dst_pts[k][1]:
                        break

                    w[k] = 1. / (
                        (i - self.dst_pts[k][0]) * (i - self.dst_pts[k][0]) +
                        (j - self.dst_pts[k][1]) * (j - self.dst_pts[k][1]))

                    sw += w[k]
                    swp = swp + w[k] * np.array(self.dst_pts[k])
                    swq = swq + w[k] * np.array(self.src_pts[k])

                if k == self.pt_count - 1:
                    pstar = 1 / sw * swp
                    qstar = 1 / sw * swq

                    miu_s = 0
                    for k in range(self.pt_count):
                        if i == self.dst_pts[k][0] and j == self.dst_pts[k][1]:
                            continue
                        pt_i = self.dst_pts[k] - pstar
                        miu_s += w[k] * np.sum(pt_i * pt_i)

                    cur_pt -= pstar
                    cur_pt_j = np.array([-cur_pt[1], cur_pt[0]])

                    for k in range(self.pt_count):
                        if i == self.dst_pts[k][0] and j == self.dst_pts[k][1]:
                            continue

                        pt_i = self.dst_pts[k] - pstar
                        pt_j = np.array([-pt_i[1], pt_i[0]])

                        tmp_pt = np.zeros(2, dtype=np.float32)
                        tmp_pt[0] = np.sum(pt_i * cur_pt) * self.src_pts[k][0] - \
                                    np.sum(pt_j * cur_pt) * self.src_pts[k][1]
                        tmp_pt[1] = -np.sum(pt_i * cur_pt_j) * self.src_pts[k][0] + \
                                    np.sum(pt_j * cur_pt_j) * self.src_pts[k][1]
                        tmp_pt *= (w[k] / miu_s)
                        new_pt += tmp_pt

                    new_pt += qstar
                else:
                    new_pt = self.src_pts[k]

                self.rdx[j, i] = new_pt[0] - i
                self.rdy[j, i] = new_pt[1] - j

                j += self.grid_size
            i += self.grid_size

    def gen_img(self):
        src_h, src_w = self.src.shape[:2]
        dst = np.zeros_like(self.src, dtype=np.float32)

        for i in np.arange(0, self.dst_h, self.grid_size):
            for j in np.arange(0, self.dst_w, self.grid_size):
                ni = i + self.grid_size
                nj = j + self.grid_size
                w = h = self.grid_size
                if ni >= self.dst_h:
                    ni = self.dst_h - 1
                    h = ni - i + 1
                if nj >= self.dst_w:
                    nj = self.dst_w - 1
                    w = nj - j + 1

                di = np.reshape(np.arange(h), (-1, 1))
                dj = np.reshape(np.arange(w), (1, -1))
                delta_x = self.__bilinear_interp(
                    di / h, dj / w, self.rdx[i, j], self.rdx[i, nj],
                    self.rdx[ni, j], self.rdx[ni, nj])
                delta_y = self.__bilinear_interp(
                    di / h, dj / w, self.rdy[i, j], self.rdy[i, nj],
                    self.rdy[ni, j], self.rdy[ni, nj])
                nx = j + dj + delta_x * self.trans_ratio
                ny = i + di + delta_y * self.trans_ratio
                nx = np.clip(nx, 0, src_w - 1)
                ny = np.clip(ny, 0, src_h - 1)
                nxi = np.array(np.floor(nx), dtype=np.int32)
                nyi = np.array(np.floor(ny), dtype=np.int32)
                nxi1 = np.array(np.ceil(nx), dtype=np.int32)
                nyi1 = np.array(np.ceil(ny), dtype=np.int32)

                if len(self.src.shape) == 3:
                    x = np.tile(np.expand_dims(ny - nyi, axis=-1), (1, 1, 3))
                    y = np.tile(np.expand_dims(nx - nxi, axis=-1), (1, 1, 3))
                else:
                    x = ny - nyi
                    y = nx - nxi
                dst[i:i + h, j:j + w] = self.__bilinear_interp(
                    x, y, self.src[nyi, nxi], self.src[nyi, nxi1],
                    self.src[nyi1, nxi], self.src[nyi1, nxi1])

        dst = np.clip(dst, 0, 255)
        dst = np.array(dst, dtype=np.uint8)

        return dst


def tia_distort(src, segment=4):
    img_h, img_w = src.shape[:2]

    cut = img_w // segment
    thresh = cut // 3

    src_pts = list()
    dst_pts = list()

    src_pts.append([0, 0])
    src_pts.append([img_w, 0])
    src_pts.append([img_w, img_h])
    src_pts.append([0, img_h])

    dst_pts.append([np.random.randint(thresh), np.random.randint(thresh)])
    dst_pts.append(
        [img_w - np.random.randint(thresh), np.random.randint(thresh)])
    dst_pts.append(
        [img_w - np.random.randint(thresh), img_h - np.random.randint(thresh)])
    dst_pts.append(
        [np.random.randint(thresh), img_h - np.random.randint(thresh)])

    half_thresh = thresh * 0.5

    for cut_idx in np.arange(1, segment, 1):
        src_pts.append([cut * cut_idx, 0])
        src_pts.append([cut * cut_idx, img_h])
        dst_pts.append([
            cut * cut_idx + np.random.randint(thresh) - half_thresh,
            np.random.randint(thresh) - half_thresh
        ])
        dst_pts.append([
            cut * cut_idx + np.random.randint(thresh) - half_thresh,
            img_h + np.random.randint(thresh) - half_thresh
        ])

    trans = WarpMLS(src, src_pts, dst_pts, img_w, img_h)
    dst = trans.generate()

    return dst


def tia_stretch(src, segment=4):
    img_h, img_w = src.shape[:2]

    cut = img_w // segment
    thresh = cut * 4 // 5

    src_pts = list()
    dst_pts = list()

    src_pts.append([0, 0])
    src_pts.append([img_w, 0])
    src_pts.append([img_w, img_h])
    src_pts.append([0, img_h])

    dst_pts.append([0, 0])
    dst_pts.append([img_w, 0])
    dst_pts.append([img_w, img_h])
    dst_pts.append([0, img_h])

    half_thresh = thresh * 0.5

    for cut_idx in np.arange(1, segment, 1):
        move = np.random.randint(thresh) - half_thresh
        src_pts.append([cut * cut_idx, 0])
        src_pts.append([cut * cut_idx, img_h])
        dst_pts.append([cut * cut_idx + move, 0])
        dst_pts.append([cut * cut_idx + move, img_h])

    trans = WarpMLS(src, src_pts, dst_pts, img_w, img_h)
    dst = trans.generate()

    return dst


def tia_perspective(src):
    img_h, img_w = src.shape[:2]

    thresh = img_h // 2

    src_pts = list()
    dst_pts = list()

    src_pts.append([0, 0])
    src_pts.append([img_w, 0])
    src_pts.append([img_w, img_h])
    src_pts.append([0, img_h])

    dst_pts.append([0, np.random.randint(thresh)])
    dst_pts.append([img_w, np.random.randint(thresh)])
    dst_pts.append([img_w, img_h - np.random.randint(thresh)])
    dst_pts.append([0, img_h - np.random.randint(thresh)])

    trans = WarpMLS(src, src_pts, dst_pts, img_w, img_h)
    dst = trans.generate()

    return dst


def get_crop_pad(image):
    """
    random crop or pad
    """
    h, w, _ = image.shape
    pixel_min = 1
    pixel_max = 8
    pixel = int(random.randint(pixel_min, pixel_max))
    pixel = min(pixel, h - 1)

    img_copy = image.copy()
    ratio = random.randint(0, 3)
    if ratio == 0:
        img_result = img_copy[pixel:h, :, :]
    elif ratio == 1:
        img_result = img_copy[0:h - pixel, :, :]
    elif ratio == 3:
        img_result = np.pad(img_copy, ((0,pixel),(0,0),(0,0)), mode='edge')
    else:
        img_result = np.pad(img_copy, ((pixel,0),(0,0),(0,0)), mode='edge')
    return img_result


@PIPELINES.register()
class RecTIA(A.ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1.0):
        super(RecTIA, self).__init__(always_apply, p)

    def apply(self, image, **params):
        prob = 0.5
        img_height, img_width = image.shape[0:2]
        new_img = image
        if random.random() <= prob and img_height >= 20 and img_width >= 20:
            new_img = tia_distort(new_img, random.randint(3, 6))

        if random.random() <= prob and img_height >= 20 and img_width >= 20:
            new_img = tia_stretch(new_img, random.randint(3, 6))

        if random.random() <= prob:
            new_img = tia_perspective(new_img)

        if random.random() <= prob and img_height >= 20 and img_width >= 20:
            new_img = get_crop_pad(new_img)
        
        return new_img


@PIPELINES.register()
class Reverse(A.ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1.0):
        super(Reverse, self).__init__(always_apply, p)

    def apply(self, image, **params):
        return 255 - image


@PIPELINES.register()
class Flipud(A.ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1.0):
        super(Flipud, self).__init__(always_apply, p)

    def apply(self, image, **params):
        return image[::-1]