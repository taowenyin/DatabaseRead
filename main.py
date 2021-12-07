import argparse

from MSLS import MSLS
from torch.utils.data import DataLoader
from tqdm import trange
from tqdm import tqdm
from time import sleep


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Database Read')

    parser.add_argument('--dataset_root_dir', type=str, default='/mnt/Dataset/Mapillary_Street_Level_Sequences',
                        help='Root directory of dataset')

    opt = parser.parse_args()

    dataset = MSLS(root_dir=opt.dataset_root_dir, cities_list='trondheim,london',
                   mode='train', task='im2im', seq_length=1)

    # 更新数据集的匹数
    dataset.new_epoch()

    # 载入数据集
    load_cached_subset = trange(dataset.cached_subset_size, desc='读取第1批数据')
    for sub_iter in load_cached_subset:
        load_cached_subset.set_description('读取第{}批数据'.format(sub_iter))

        # 获取新一批数据
        dataset.refresh_data()

        load_data_loader = tqdm(DataLoader(dataset=dataset, batch_size=2, shuffle=False, collate_fn=MSLS.collate_fn),
                                leave=True, desc='读取Batch 0')
        for iteration, (query, positives, negatives, neg_counts, indices) in enumerate(load_data_loader, start=sub_iter):
            load_data_loader.set_description('读取Batch {}'.format(iteration))

            sleep(1)

    pass
