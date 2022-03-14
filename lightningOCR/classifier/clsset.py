import cv2
import lmdb
import numpy as np
from torch.utils.data import Dataset
from abc import ABCMeta, abstractmethod

from lightningOCR.common import Compose, DATASETS


@DATASETS.register()
class ClsDataset(Dataset, metaclass=ABCMeta):
    """Dataset for direction classification.

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
                 length=None):
        super(ClsDataset, self).__init__()
        if isinstance(data_root, str):
            self.data_root = [data_root]
        else:
            self.data_root = data_root

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

        self.pipeline  = Compose(pipeline['transforms'], 
                                 pipeline.get('bbox_params'), 
                                 pipeline.get('keypoint_params'),
                                 pipeline.get('additional_targets'))

        self.length = length
        
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

    def channel_first(self, results):
        image = results['image']
        if len(image.shape)==2:
            image = image[..., None]
        results['image'] = image.transpose(2,0,1)
        return results

    def __getitem__(self, idx):
        lmdb_idx, file_idx = self.data_idx_order_list[idx]
        lmdb_idx = int(lmdb_idx)
        file_idx = int(file_idx)
        sample_info = self.get_lmdb_sample_info(self.lmdb_sets[lmdb_idx]['txn'], file_idx)
        if sample_info is None:
            return self.__getitem__(np.random.randint(self.__len__()))
        imgbuf, label = sample_info
        img = np.frombuffer(imgbuf, dtype='uint8')
        img = cv2.imdecode(img, 1)

        data = {'image': img, 'label': 0}
        results = self.pipeline(data)
        results = self.channel_first(results)
        
        return results

    def __len__(self):
        if self.length is not None:
            return min(self.data_idx_order_list.shape[0], self.length)
        else:
            return self.data_idx_order_list.shape[0]