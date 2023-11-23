import os
import numpy as np
import torch

def load_data(data_path):
    file_type = os.path.splitext(data_path)[1][1:]
    if file_type == 'txt':
        data = np.loadtxt(data_path)
    elif file_type == 'npy':
        data = np.load(data_path)
    else:
        raise Exception('File type \'{}\' not supported.'.format(file_type))
    return data
    
def collate_fn(batch):
    coord, feat, label = list(zip(*batch))
    offset, count = [], 0
    for item in coord:
        count += item.shape[0]
        offset.append(count)
    return torch.cat(coord), torch.cat(feat), torch.cat(label), torch.IntTensor(offset)