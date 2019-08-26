from torch.utils.data import Dataset
import numpy as np
from random import sample
from math import ceil

from utils import load_case, load_segmentation

# the max
TRAIN_DATAS_RANGE_MAX = 5


def hu_to_grayscale(volume, hu_min=-512, hu_max=512):
    mxval = np.max(volume)
    mnval = np.min(volume)
    im_volume = (volume - mnval)/max(mxval - mnval, 1e-3)
    im_volume = 255*im_volume
    return np.stack((im_volume, im_volume, im_volume), axis=-1)


class KidneyTumor(Dataset):
    def __init__(self, datas):
        super(KidneyTumor, self).__init__()
        self.datas = datas
        self.length = len(datas)
        self.cache_vol = None
        self.cache_seg = None
        self.cache_case_id = -1

    def __getitem__(self, idx):
        case_id, no = self.datas[idx]
        if self.cache_case_id != case_id:
            del self.cache_vol
            del self.cache_seg
            self.cache_vol, self.cache_seg = load_case(case_id)
            self.cache_vol, self.cache_seg = self.cache_vol.get_data(), self.cache_seg.get_data()
            self.cache_case_id = case_id
        return hu_to_grayscale(self.cache_vol[no]), self.cache_seg[no]

    def __len__(self):
        return self.length


# 数据预处理器
class Preprocessor(object):
    def __init__(self, sample_rate=.8, max_case_num=TRAIN_DATAS_RANGE_MAX):
        self.test_data_length = ceil(max_case_num*(1-sample_rate))
        self.train_data_length = max_case_num-self.test_data_length
        self.test_cases = sample(range(max_case_num), self.test_data_length)
        self.train_cases = list(
            filter(lambda x: x not in self.test_cases, range(max_case_num)))

    def get_train_datas(self):
        train_datas = []
        for case_id in self.train_cases:
            seg = load_segmentation(case_id).get_data()
            train_datas.extend((case_id, x) for x in range(len(seg)))
        return KidneyTumor(train_datas)

    def get_test_datas(self):
        test_datas = []
        for case_id in self.train_cases:
            seg = load_segmentation(case_id).get_data()
            test_datas.extend((case_id, x) for x in range(len(seg)))
        return KidneyTumor(test_datas)
