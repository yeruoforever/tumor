from torch.utils.data import Dataset
import numpy as np
from random import sample
from math import ceil

from utils import load_case, load_segmentation

# the max
TRAIN_DATAS_RANGE_MAX = 5


def hu_to_grayscale(volume, hu_min=-512, hu_max=512):
    "将原始CT图片数据处理为三通道灰度图,用hu_min和hu_max过滤过暗和过亮"
    mxval = np.max(volume)
    mnval = np.min(volume)
    im_volume = (volume - mnval)/max(mxval - mnval, 1e-3)
    im_volume = 255*im_volume
    return np.stack((im_volume, im_volume, im_volume), axis=-1)


class KidneyTumor(Dataset):
    "肾肿瘤数据集对象"

    def __init__(self, datas):
        "传入元素为（case_id，图片id）的数组来创建torch可用的Dataset"
        super(KidneyTumor, self).__init__()
        self.datas = datas
        self.length = len(datas)
        self.cache_vol = None
        self.cache_seg = None
        self.cache_case_id = -1

    def __getitem__(self, idx):
        "获得数据(batch_size,width,hight,chanel),和类别标签（batch_size,width,hight）"
        case_id, no = self.datas[idx]
        if self.cache_case_id != case_id:
            del self.cache_vol
            del self.cache_seg
            self.cache_vol, self.cache_seg = load_case(case_id)
            self.cache_vol, self.cache_seg = self.cache_vol.get_data(), self.cache_seg.get_data()
            self.cache_case_id = case_id
        return hu_to_grayscale(self.cache_vol[no]), self.cache_seg[no]

    def __len__(self):
        "返回数据集的长度"
        return self.length


# 数据预处理器
class Preprocessor(object):
    "用于生成训练数据和测试数据"

    def __init__(self, sample_rate=.8, max_case_num=TRAIN_DATAS_RANGE_MAX):
        '默认取80%数据作为训练集，以病人为单位'
        self.test_data_length = ceil(max_case_num*(1-sample_rate))
        self.train_data_length = max_case_num-self.test_data_length
        self.test_cases = sample(range(max_case_num), self.test_data_length)
        self.train_cases = list(
            filter(lambda x: x not in self.test_cases, range(max_case_num)))

    def get_train_datas(self):
        "获得训练集"
        train_datas = []
        for case_id in self.train_cases:
            seg = load_segmentation(case_id).get_data()
            train_datas.extend((case_id, x) for x in range(len(seg)))
        return KidneyTumor(train_datas)

    def get_test_datas(self):
        "获得测试集"
        test_datas = []
        for case_id in self.train_cases:
            seg = load_segmentation(case_id).get_data()
            test_datas.extend((case_id, x) for x in range(len(seg)))
        return KidneyTumor(test_datas)
