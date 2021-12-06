from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from tqdm import tqdm
from os.path import join
from sklearn.neighbors import NearestNeighbors

import pandas as pd
import numpy as np
import sys


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
                 sub_task='all', seq_length=1, exclude_panos=True, positive_sampling=True):
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
        :param positive_sampling: 是否进行正采样
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
        # 所有Query对应的正例索引
        self.all_positive_indices = []
        # Query的序列索引
        self.q_seq_idx = []
        # positive的序列索引
        self.p_seq_idx = []
        # 不是负例的索引
        self.non_negative_indices = []
        # 路边的数据
        self.sideways = []
        # 晚上的数据
        self.night = []

        self.mode = mode
        self.sub_task = sub_task
        self.exclude_panos = exclude_panos
        self.positive_distance_threshold = positive_distance_threshold
        self.negative_distance_threshold = negative_distance_threshold

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

            # 保存没有正例的图像数
            non_positive_q_seq_keys_count = 0
            # 保存有正例的图像数
            has_positive_q_seq_keys_count = 0
            # 保存数据集的个数
            q_seq_keys_count = 0

            # 获取到目前为止用于索引的城市图像的长度
            _lenQ = len(self.q_images_key)
            _lenDb = len(self.db_images_key)

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
                q_seq_keys_count = len(q_seq_keys)

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

                # 获取图像的UTM坐标
                utm_q = q_data[['easting', 'northing']].values.reshape(-1, 2)
                utm_db = db_data[['easting', 'northing']].values.reshape(-1, 2)

                # 获取Query图像的Night状态、否是Sideways，以及图像索引
                night, sideways, index = q_data['night'].values, \
                                         (q_data['view_direction'] == 'Sideways').values, \
                                         q_data.index

                # 创建最近邻算法，使用暴力搜索法
                neigh = NearestNeighbors(algorithm='brute')
                # 对数据集进行拟合
                neigh.fit(utm_db)
                # 在Database中找到符合positive_distance_threshold要求的Query数据的最近邻数据的索引
                positive_distance, positive_indices = neigh.radius_neighbors(utm_q, self.positive_distance_threshold)
                # 保存所有正例索引
                self.all_positive_indices.extend(positive_indices)

                # 训练模式下，获取负例索引
                if self.mode == 'train':
                    negative_distance, negative_indices = neigh.radius_neighbors(
                        utm_q, self.negative_distance_threshold)

                # 查看每个Seq的正例
                for q_seq_key_idx in range(len(q_seq_keys)):
                    # 返回每个序列的帧集合
                    q_frame_idxs = self.seq_idx_2_frame_idx(q_seq_key_idx, q_seq_idxs)
                    # 返回q_frame_idxs在unique_q_seq_idxs中的索引集合
                    q_uniq_frame_idx = self.frame_idx_2_uniq_frame_idx(q_frame_idxs, unique_q_seq_idxs)
                    # 返回序列Query中序列对应的正例索引
                    positive_uniq_frame_idxs = np.unique([p for pos in positive_indices[q_uniq_frame_idx] for p in pos])

                    # 查询的序列Query至少要有一个正例
                    if len(positive_uniq_frame_idxs) > 0:
                        # 获取正例所在的序列索引，并去除重复的索引
                        positive_seq_idx = np.unique(self.uniq_frame_idx_2_seq_idx(
                            unique_db_seq_idxs[positive_uniq_frame_idxs], db_seq_idxs))

                        # todo 不知道是什么意思
                        self.p_seq_idx.append(positive_seq_idx + _lenDb)
                        self.q_seq_idx.append(q_seq_key_idx + _lenQ)

                        # 在训练的时候需要根据两个阈值找到正例和负例
                        if self.mode == 'train':
                            # 找到不是负例的数据
                            n_uniq_frame_idxs = np.unique(
                                [n for nonNeg in negative_indices[q_uniq_frame_idx] for n in nonNeg])
                            # 找到不是负例所在的序列索引，并去除重复的索引
                            n_seq_idx = np.unique(
                                self.uniq_frame_idx_2_seq_idx(unique_db_seq_idxs[n_uniq_frame_idxs], db_seq_idxs))

                            # 保存数据
                            self.non_negative_indices.append(n_seq_idx + _lenDb)

                            # todo 不知道是什么意思
                            if sum(night[np.in1d(index, q_frame_idxs)]) > 0:
                                self.night.append(len(self.q_seq_idx) - 1)
                            if sum(sideways[np.in1d(index, q_frame_idxs)]) > 0:
                                self.sideways.append(len(self.q_seq_idx) - 1)

                        has_positive_q_seq_keys_count += 1
                    else:
                        non_positive_q_seq_keys_count += 1

            # 读取测试集数据集，GPS/UTM/Pano都不可用
            elif self.mode in ['test']:
                # 载入对应子任务的图像索引
                q_idx = pd.read_csv(join(root_dir, subdir, city, 'query', 'subtask_index.csv'), index_col=0)
                db_idx = pd.read_csv(join(root_dir, subdir, city, 'database', 'subtask_index.csv'), index_col=0)

                # 根据任务把数据变成序列
                q_seq_keys, q_seq_idxs = self.rang_to_sequence(q_idx, join(root_dir, subdir, city, 'query'),
                                                               seq_length_q)
                db_seq_keys, db_seq_idxs = self.rang_to_sequence(db_idx, join(root_dir, subdir, city, 'database'),
                                                                 seq_length_db)

                # 从所有序列数据中根据符合子任务的中心索引找到序列数据帧
                val_frames = np.where(q_idx[self.sub_task])[0]
                q_seq_keys, q_seq_idxs = self.filter(q_seq_keys, q_seq_idxs, val_frames)

                val_frames = np.where(db_idx[self.sub_task])[0]
                db_seq_keys, db_seq_idxs = self.filter(db_seq_keys, db_seq_idxs, val_frames)

                # 保存筛选后的图像
                self.q_images_key.extend(q_seq_keys)
                self.db_images_key.extend(db_seq_keys)

                # 添加Query索引
                self.q_seq_idx.extend(list(range(_lenQ, len(q_seq_keys) + _lenQ)))

            tqdm.write('{}城市的数据，有正例的图像有[{}/{}]个，没有正例的图像有[{}/{}]个'.format(
                city,
                has_positive_q_seq_keys_count,
                q_seq_keys_count,
                non_positive_q_seq_keys_count,
                q_seq_keys_count))

        # 如果选择了城市、任务和子任务的组合，其中没有Query和Database图像，则退出。
        if len(self.q_images_key) == 0 or len(self.db_images_key) == 0:
            tqdm.write("退出...")
            tqdm.write("如果选择了城市、任务和子任务的组合，其中没有Query和Database图像，则退出")
            tqdm.write("尝试选择不同的子任务或其他城市")
            sys.exit()

        self.q_seq_idx = np.asarray(self.q_seq_idx)
        self.q_images_key = np.asarray(self.q_images_key)
        self.p_seq_idx = np.asarray(self.p_seq_idx, dtype=object)
        self.non_negative_indices = np.asarray(self.non_negative_indices, dtype=object)
        self.db_images_key= np.asarray(self.db_images_key)
        self.sideways = np.asarray(self.sideways)
        self.night = np.asarray(self.night)

        if self.mode in ['train']:
            # 计算正例采样的权重
            if positive_sampling:
                self.__calcSamplingWeights__()
            else:
                self.weights = np.ones(len(self.q_seq_idx)) / float(len(self.q_seq_idx))

    def __calcSamplingWeights__(self):
        """
        计算数据权重
        """
        # 计算Query大小
        N = len(self.q_seq_idx)

        # 初始化权重都为1
        self.weights = np.ones(N)

        # 夜间或侧面时权重更高
        if len(self.night) != 0:
            self.weights[self.night] += N / len(self.night)
        if len(self.sideways) != 0:
            self.weights[self.sideways] += N / len(self.sideways)

        # 打印权重信息
        tqdm.write("#侧面 [{}/{}]; #夜间; [{}/{}]".format(len(self.sideways), N, len(self.night), N))
        tqdm.write("正面和白天的权重为{:.4f}".format(1))
        if len(self.night) != 0:
            tqdm.write("正面且夜间的权重为{:.4f}".format(1 + N / len(self.night)))
        if len(self.sideways) != 0:
            tqdm.write("侧面且白天的权重为{:.4f}".format(1 + N / len(self.sideways)))
        if len(self.sideways) != 0 and len(self.night) != 0:
            tqdm.write("侧面且夜间的权重为{:.4f}".format(1 + N / len(self.night) + N / len(self.sideways)))


    def seq_idx_2_frame_idx(self, q_seq_key, q_seq_keys):
        """
        把序列索引转化为帧索引

        :param q_seq_key: 序列索引
        :param q_seq_keys: 序列集合
        :return: 索引对应的序列集合
        """
        return q_seq_keys[q_seq_key]

    def frame_idx_2_uniq_frame_idx(self, frame_idx, uniq_frame_idx):
        """
        获取frame_idx在uniq_frame_idx中的索引列表

        :param frame_idx: 一个序列的帧ID
        :param uniq_frame_idx: 所有帧ID
        :return: 获取frame_idx在uniq_frame_idx中的索引列表
        """

        # 在不重复的数据帧列表uniq_frame_idx中找到要找的数据帧frame_idx，并产生对应的Mask
        frame_mask = np.in1d(uniq_frame_idx, frame_idx)

        # 返回frame_idx在uniq_frame_idx中的索引
        return np.where(frame_mask)[0]

    def uniq_frame_idx_2_seq_idx(self, frame_idxs, seq_idxs):
        """
        返回图像帧对应的序列索引

        :param frame_idxs: 图像帧
        :param seq_idxs: 序列索引
        :return: 图像正所在的序列索引
        """

        # 在序列索引列表seq_idxs中找到要找的数据帧frame_idxs，并产生对应的Mask
        mask = np.in1d(seq_idxs, frame_idxs)
        # 把Mask重新组织成seq_idxs的形状
        mask = mask.reshape(seq_idxs.shape)

        # 得到序列的索引
        return np.where(mask)[0]

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

    def __len__(self):
        return 10

    def __getitem__(self, index):

        return None, None, None, None,

    @staticmethod
    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))


        return None

