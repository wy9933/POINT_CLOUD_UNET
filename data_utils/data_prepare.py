import os
import argparse

import data_utils.s3dis as s3dis

def data_prepare(dataset_name, dataset_root, dataset_target, file_type='npy'):
    # check dataset support
    dataset = None
    if dataset_name == 's3dis':
        dataset = s3dis.PrepareDataset(dataset_root, dataset_target, file_type)

    if dataset == None:
        raise Exception('Dataset %s not supported yet or a WRONG dataset name' % dataset_name)
    elif not dataset.is_available():
        raise Exception('Dataset %s not supported yet' % dataset_name)
    
    # check dataset path exist
    if not os.path.exists(dataset_root) or not os.path.exists(dataset_target):
        raise Exception('Dataset path not exist, please check out your set path')


    dataset.prepare()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare Dataset')
    parser.add_argument('--dataset_name', type=str, default='', help='The name of dataset')
    parser.add_argument('--dataset_root', type=str, default='', help='The root dir of dataset')
    parser.add_argument('--dataset_target', type=str, default='', help='The target dir of dataset')
    parser.add_argument('--file_type', type=str, default='', help='The output file type of dataset')
    args = parser.parse_args()

    data_prepare(args.dataset_name, args.dataset_root, args.dataset_target, args.file_type)
