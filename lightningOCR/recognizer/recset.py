import cv2
import lmdb
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from torch.utils.data import Dataset
from abc import ABCMeta, abstractmethod

from lightningOCR.common import Compose, DATASETS, BaseDataset
from lightningOCR.common.utils import is_chinese

matplotlib.use('Agg')  # for writing to files only


@DATASETS.register()
class RecDataset(BaseDataset):
    """Dataset for recognition.

    An example of file structure is as followed.

    data folder organization
    ${data}
    ├── data.mdb

    Args:
        data_root (str or list of str): data root or list of data roots
        pipeline (list[dict]): Processing pipeline
        length (int or None): length of dataset
    """

    def __init__(self,
                 data_root,
                 pipeline,
                 length=None,
                 fontfile=None):
        super(RecDataset, self).__init__(pipeline)
        if isinstance(data_root, str):
            self.data_root = [data_root]
        else:
            self.data_root = data_root

        num_samples = 0
        for i, root in enumerate(self.data_root):
            with lmdb.open(root) as env:
                txn = env.begin(write=False)
                num_samples += int(txn.get('num-samples'.encode()))

        self.length = num_samples if length is None else length
        assert self.length <= num_samples, \
            "The data number is only {}, but set by {}".format(num_samples, self.length)

        self.fontfile = fontfile
    
    def open_lmdb(self):
        # load lmdb data
        self.lmdb_sets = {}
        for i, root in enumerate(self.data_root):
            env = lmdb.open(root,
                            max_readers=32,
                            readonly=True,
                            lock=False,
                            readahead=False,
                            meminit=False)
            txn = env.begin(write=False)
            num_samples = int(txn.get('num-samples'.encode()))
            self.lmdb_sets[i] = {"dirpath":root, "env":env, "txn":txn, "num_samples":num_samples}
        self.data_idx_order_list = self.dataset_traversal()

    def dataset_traversal(self):
        lmdb_num = len(self.lmdb_sets)
        total_sample_num = 0
        for lno in range(lmdb_num):
            total_sample_num += self.lmdb_sets[lno]['num_samples']
        data_idx_order_list = np.zeros((total_sample_num, 2))
        beg_idx = 0
        for lno in range(lmdb_num):
            tmp_sample_num = self.lmdb_sets[lno]['num_samples']
            end_idx = beg_idx + tmp_sample_num
            data_idx_order_list[beg_idx:end_idx, 0] = lno
            data_idx_order_list[beg_idx:end_idx, 1] \
                = list(range(tmp_sample_num))
            data_idx_order_list[beg_idx:end_idx, 1] += 1
            beg_idx = beg_idx + tmp_sample_num
        return data_idx_order_list

    def get_lmdb_sample_info(self, txn, index):
        label_key = 'label-%09d'.encode() % index
        label = txn.get(label_key)
        if label is None:
            return None
        label = label.decode('utf-8')
        img_key = 'image-%09d'.encode() % index
        imgbuf = txn.get(img_key)
        return imgbuf, label

    def load_data(self, idx):
        if not hasattr(self, 'lmdb_sets'):
            self.open_lmdb()
        lmdb_idx, file_idx = self.data_idx_order_list[idx]
        lmdb_idx = int(lmdb_idx)
        file_idx = int(file_idx)
        sample_info = self.get_lmdb_sample_info(self.lmdb_sets[lmdb_idx]['txn'], file_idx)
        if sample_info is None:
            return self.load_data(np.random.randint(self.__len__()))
        imgbuf, label = sample_info
        img = np.frombuffer(imgbuf, dtype='uint8')
        img = cv2.imdecode(img, 1)
        return {'image': img, 'label': label, 'target': label}

    def after_pipeline(self, results):
        image = results['image']
        if len(image.shape) == 2:
            image = image[..., None]
        results['image'] = image.transpose(2,0,1)
        return results

    def gather(self, results):
        results = {
            'image': results['image'],
            'gt':{
                'target': results['target'],
                'target_length': results['length'],
                'label': results['label']
            }
        }
        return results

    def __len__(self):
        return self.length

    def plot_idx(self, idx, out_file):
        results = self.__getitem__(idx)
        image = results['image']
        label = results['gt']['label']

        # reverse normalize
        image = image.transpose(1,2,0) * np.array(self.std) + np.array(self.mean)
        image = np.clip(image * 255, 0, 255).astype(np.uint8)

        plt.imshow(image[:,:,::-1])
        plt.title(label, fontproperties=font_manager.FontProperties(fname=self.fontfile))
        plt.savefig(out_file, dpi=200, bbox_inches='tight')
        plt.close()

    def plot_batch(self, batch, out_file):
        images = batch['image']
        labels = batch['gt']['label']

        # reverse normalize
        images = images.cpu().numpy()
        images = images.transpose(0,2,3,1) * np.array(self.std) + np.array(self.mean)
        images = np.clip(images * 255, 0, 255).astype(np.uint8)
        
        max_subplots = 16
        bs, h, w, c = images.shape
        bs = min(bs, max_subplots)  # limit plot images
        ns = int(np.ceil(bs ** 0.5))  # number of subplots (square)

        plt.subplots(ns, ns)
        for i, (img, label) in enumerate(zip(images, labels)):
            if i == bs:  # if last batch has fewer images than we expect
                break
            plt.subplot(ns, ns, i+1)
            plt.imshow(img[:,:,::-1])
            division = 10 if is_chinese(label) else 20
            length = len(label)
            label = [label[j*division:(j+1)*division] for j in range((length + division - 1 ) // division)]
            label = '\n'.join(label)
            plt.title(label, fontproperties=font_manager.FontProperties(fname=self.fontfile), fontsize=8)
            plt.xticks([])
            plt.yticks([])
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close()
