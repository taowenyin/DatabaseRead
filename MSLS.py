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
                 sub_task='all', seq_length=1, exclude_panos=True):
        """
        Mapillary Street-level Sequences数据集的读取

        task（任务）：im2im（图像到图像）, seq2seq（图像序列到图像序列）, seq2im（图像序列到图像）, im2seq（图像到图像序列）

        sub_task（子任务）：all，s2w（summer2winter），w2s（winter2summer），o2n（old2new），n2o（new2old），d2n（day2night），n2d（night2day）

        :param root_dir: 数据集的路径
        :param mode: 数据集的模式[train, val, test]
        :param cities_list: 城市列表
        :param img_transform: 图像转换函数
        :param negative_num: 每个正例对应的反例个数
        :param positive_distance_threshold: 正例的距离阈值
        :param negative_distance_threshold: 反例的距离阈值
        :param batch_size: 每批数据的大小
        :param task: 任务类型 [im2im, seq2seq, seq2im, im2seq]
        :param sub_task: 任务类型 [all, s2w, w2s, o2n, n2o, d2n, n2d]
        :param seq_length: 不同任务的序列长度
        :param exclude_panos: 是否排除全景图像
        """
        super().__init__()

        if cities_list is None:
            self.cities_list = default_cities[mode]
        else:
            self.cities_list = cities_list.split(',')

        # 筛选后的Query图像
        self.q_images_key = []
        # 筛选后的Database图像
        self.db_images_key = []

        self.mode = mode
        self.sub_task = sub_task
        self.exclude_panos = exclude_panos

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

                # 如果验证集，那么需要确定子任务的类型
                if self.mode in ['val']:
                    q_idx = pd.read_csv(join(root_dir, subdir, city, 'query', 'subtask_index.csv'), index_col=0)
                    db_idx = pd.read_csv(join(root_dir, subdir, city, 'database', 'subtask_index.csv'), index_col=0)

                    # 从所有序列数据中根据符合子任务的中心索引找到序列数据帧
                    val_frames = np.where(q_idx[self.sub_task])[0]
                    q_seq_keys, q_seq_idxs = self.filter(q_seq_keys, q_seq_idxs, val_frames)

                    val_frames = np.where(db_idx[self.sub_task])[0]
                    db_seq_keys, db_seq_idxs = self.filter(db_seq_keys, db_seq_idxs, val_frames)

                # 筛选出不同全景的数据
                if self.exclude_panos:
                    panos_frames = np.where((q_data_raw['pano'] == False).values)[0]
                    # 从Query数据中筛选出不是全景的数据
                    q_seq_keys, q_seq_idxs = self.filter(q_seq_keys, q_seq_idxs, panos_frames)

                    panos_frames = np.where((db_data_raw['pano'] == False).values)[0]
                    # 从Query数据中筛选出不是全景的数据
                    db_seq_keys, db_seq_idxs = self.filter(db_seq_keys, db_seq_idxs, panos_frames)

                # 删除重复的idx
                unique_q_seq_idxs = np.unique(q_seq_idxs)
                unique_db_seq_idxs = np.unique(db_seq_idxs)

                # 如果排除重复后没有数据，那么就下一个城市
                if len(unique_q_seq_idxs) == 0 or len(unique_db_seq_idxs) == 0:
                    continue

                # 保存筛选后的图像
                self.q_images_key.extend(q_seq_keys)
                self.db_images_key.extend(db_seq_keys)

                # 从原数据中筛选后数据
                q_data = q_data.loc[unique_q_seq_idxs]
                db_data = db_data.loc[unique_db_seq_idxs]


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

        for idx in data.index:
            # 边界的情况
            if idx < (seq_length // 2) or idx >= (len(seq_info) - seq_length // 2):
                continue

            # 计算当前序列数据帧的周边帧
            seq_idx = np.arange(-seq_length // 2, seq_length // 2) + 1 + idx
            # 获取一个序列帧
            seq = seq_info.iloc[seq_idx]

            # 一个序列必须是具有相同的序列键值（即sequence_key相同），以及连续的帧（即frame_number之间的差值为1）
            if len(np.unique(seq['sequence_key'])) == 1 and (seq['frame_number'].diff()[1:] == 1).all():
                seq_key = ','.join([join(path, 'images', key + '.jpg') for key in seq['key']])

                # 保存图像序列的名称
                seq_keys.append(seq_key)
                # 保存图像序列的索引
                seq_idxs.append(seq_idx)

        return seq_keys, np.asarray(seq_idxs)

    def filter(self, seq_keys, seq_idxs, center_frame_condition):
        """
        根据序列中心点索引筛选序列

        :param seq_keys: 序列Key值
        :param seq_idxs: 序列索引
        :param center_frame_condition: 条件筛选的中心帧
        :return: 返回筛选后的Key和Idx
        """
        keys, idxs = [], []
        for key, idx in zip(seq_keys, seq_idxs):
            # 如果序列的中间索引在中心帧中，那么就把Key和Idx放入数组中
            if idx[len(idx) // 2] in center_frame_condition:
                keys.append(key)
                idxs.append(idx)
        return keys, np.asarray(idxs)

