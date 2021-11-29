from torch.utils.data import Dataset
from tqdm import tqdm
from os.path import join

import pandas as pd
import numpy as np


default_cities = {
    'train': ['trondheim', 'london', 'boston', 'melbourne', 'amsterdam', 'helsinki',
              'tokyo', 'toronto', 'saopaulo', 'moscow', 'zurich', 'paris', 'bangkok',
              'budapest', 'austin', 'berlin', 'ottawa', 'phoenix', 'goa', 'amman', 'nairobi', 'manila'],
    'val': ['cph', 'sf'],
    'test': ['miami', 'athens', 'buenosaires', 'stockholm', 'bengaluru', 'kampala']
}


class MSLS(Dataset):
    def __init__(self, root_dir, mode='train', cities_list=None, img_transform=None, negative_num=5,
                 positive_distance_threshold=10, negative_distance_threshold=25, batch_size=24, task='im2im',
                 seq_length=1, exclude_panos=True):
        """
        Mapillary Street-level Sequences数据集的读取

        :param root_dir: 数据集的路径
        :param mode: 数据集的模式
        :param cities_list: 城市列表
        :param img_transform: 图像转换函数
        :param negative_num: 每个正例对应的反例个数
        :param positive_distance_threshold: 正例的距离阈值
        :param negative_distance_threshold: 反例的距离阈值
        :param batch_size: 每批数据的大小
        :param task: 任务类型
        :param seq_length: 不同任务的序列长度
        :param exclude_panos: 是否排除全景图像
        """
        super().__init__()

        if cities_list is None:
            self.cities_list = default_cities[mode]
        else:
            self.cities_list = cities_list.split(',')

        self.mode = mode

        # 根据任务类型得到序列长度
        if task == 'im2im': # 图像到图像
            seq_length_q, seq_length_db = 1, 1
        elif task == 'seq2seq': # 图像序列到图像序列
            seq_length_q, seq_length_db = seq_length, seq_length
        elif task == 'seq2im': # 图像序列到图像
            seq_length_q, seq_length_db = seq_length, 1
        else:  # im2seq 图像到图像序列
            seq_length_q, seq_length_db = 1, seq_length

        # 载入数据
        load_data_bar = tqdm(self.cities_list)
        for city in load_data_bar:
            load_data_bar.set_description('=====> Load {} data'.format(city))

            # 根据城市获得数据文件夹名称
            subdir = 'test' if city in default_cities['test'] else 'train_val'

            # 读取训练集或验证集数据集
            if self.mode in ['train', 'val']:
                # 载入Query数据
                q_data = pd.read_csv(join(root_dir, subdir, city, 'query', 'postprocessed.csv'), index_col=0)
                q_data_raw = pd.read_csv(join(root_dir, subdir, city, 'query', 'raw.csv'), index_col=0)

                # 读取数据集数据
                db_data = pd.read_csv(join(root_dir, subdir, city, 'database', 'postprocessed.csv'), index_col=0)
                db_data_raw = pd.read_csv(join(root_dir, subdir, city, 'database', 'raw.csv'), index_col=0)

                # 根据任务把数据变成序列
                q_seq_keys, q_seq_idxs = self.rang_to_sequence(q_data, join(root_dir, subdir, city, 'query'),
                                                               seq_length_q)
                db_seq_keys, db_seq_idxs = self.rang_to_sequence(db_data, join(root_dir, subdir, city, 'database'),
                                                                 seq_length_db)


        print('xxx')

    def rang_to_sequence(self, data, path, seq_length):
        """
        把数组变为序列

        :param data: 表型数据
        :param path: 数据地址
        :param seq_length: 序列长度
        """
        # 去读序列信息
        seq_info = pd.read_csv(join(path, 'seq_info.csv'), index_col=0)

        # 图像序列的名称和图像序列的索引
        seq_keys, seq_idxs = [], []

        for idx in tqdm(data.index):
            # 边界的情况
            if idx < (seq_length // 2) or idx >= (len(seq_info) - seq_length // 2):
                continue

            # 计算当前帧的周边帧
            seq_idx = np.arange(-seq_length // 2, seq_length // 2) + 1 + idx
            # 获取一个序列帧
            seq = seq_info.iloc[seq_idx]

            # the sequence must have the same sequence key and must have consecutive frames
            if len(np.unique(seq['sequence_key'])) == 1 and (seq['frame_number'].diff()[1:] == 1).all():
                seq_key = ','.join([join(path, 'images', key + '.jpg') for key in seq['key']])

                # 保存图像序列的名称
                seq_keys.append(seq_key)
                # 保存图像序列的索引
                seq_idxs.append(seq_idx)

        return seq_keys, np.asarray(seq_idxs)

