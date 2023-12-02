import os
import argparse
from importlib import import_module

import data_utils.s3dis as s3dis

def data_prepare(dataset_name, dataset_source, dataset_root, file_type='npy'):
    # check dataset support
    dataset = None
    if dataset_name == 's3dis':
        dataset = s3dis.PrepareDataset(dataset_source, dataset_root, file_type)

    if dataset == None:
        raise Exception('Dataset %s not supported yet or a WRONG dataset name' % dataset_name)
    elif not dataset.is_available():
        raise Exception('Dataset %s not supported yet' % dataset_name)
    
    # check dataset path exist
    if not os.path.exists(dataset_source) or not os.path.exists(dataset_root):
        raise Exception('Dataset path not exist, please check out your set path')

    dataset.prepare()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare Dataset')
    parser.add_argument('--dataset_name', type=str, default='', help='The name of dataset')
    parser.add_argument('--dataset_source', type=str, default='', help='The source dir of dataset initial data')
    parser.add_argument('--dataset_root', type=str, default='', help='The root dir of prepared dataset')
    parser.add_argument('--file_type', type=str, default='', help='The output file type of dataset')
    args = parser.parse_args()

    data_prepare(args.dataset_name, args.dataset_source, args.dataset_root, args.file_type)
