import argparse

from MSLS import MSLS


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Database Read')

    parser.add_argument('--dataset_root_dir', type=str, default='/mnt/Dataset/Mapillary_Street_Level_Sequences',
                        help='Root directory of dataset')

    opt = parser.parse_args()

    dataset = MSLS(root_dir=opt.dataset_root_dir,
                   mode='train', task='seq2seq', seq_length=1)

    pass
