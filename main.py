import argparse

from MSLS import MSLS
from torch.utils.data import DataLoader


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Database Read')

    parser.add_argument('--dataset_root_dir', type=str, default='/mnt/Dataset/Mapillary_Street_Level_Sequences',
                        help='Root directory of dataset')

    opt = parser.parse_args()

    dataset = MSLS(root_dir=opt.dataset_root_dir, cities_list='trondheim,london',
                   mode='train', task='im2im', seq_length=1)

    data_loader = DataLoader(dataset=dataset, batch_size=10, shuffle=False, collate_fn=MSLS.collate_fn)

    for iteration, (query, positives, negatives, negCounts, indices) in enumerate(data_loader):
        print('xxx')

    pass
