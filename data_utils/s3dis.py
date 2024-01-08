import os
import glob
import tqdm
import numpy as np

from torch.utils.data import Dataset

from data_utils.common_util import load_data

class S3DIS():
    def __init__(self) -> None:
        """Predefined s3dis related properties
        """
        self.names = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter']
        self.name2label = {name: i for i, name in enumerate(self.names)}
        self.label2name = {i: name for i, name in enumerate(self.names)}
        self.name2color = {'ceiling':	[0,   255, 0  ], 
                         'floor':	    [0,   0,   255],
                         'wall':	    [0,   255, 255],
                         'beam':        [255, 255, 0  ],
                         'column':      [255, 0,   255],
                         'window':      [100, 100, 255],
                         'door':        [200, 200, 100],
                         'table':       [170, 120, 200],
                         'chair':       [255, 0,   0  ],
                         'sofa':        [200, 100, 100],
                         'bookcase':    [10,  200, 100],
                         'board':       [200, 200, 200],
                         'clutter':     [50,  50,  50 ]} 
    

class PrepareDataset():
    def __init__(self, original_dir, dataset_dir) -> None:
        self.original_dir = original_dir
        self.dataset_dir = dataset_dir
        self.s3dis = S3DIS()

    def prepare(self) -> None:
        """prepare s3dis dataset raw data

        Raises:
            Exception: The number of scenes in S3DIS dataset is incorrect
            Exception: Output file type not supported
        """
        # check original data dir path
        scene_num = 272
        scene_list = glob.glob(os.path.join(self.original_dir, '*/*'))
        if len(scene_list) != scene_num:
            raise Exception('The number of scenes in S3DIS dataset is incorrect (have {} scenes, should have {} scenes), please check the dataset integrity'.format(len(scene_list), scene_num))
        
        # prepare dataset
        for scene in tqdm.tqdm(glob.glob(os.path.join(self.original_dir, '*/*'))):
            area_name = os.path.basename(os.path.dirname(scene))
            scene_name = os.path.basename(scene)

            points_list = []
            for anno in glob.glob(os.path.join(scene, '*/*.txt')):
                class_name = os.path.basename(anno).split('_')[0]
                if class_name not in self.s3dis.names:
                    class_name = 'clutter'

                points = np.loadtxt(anno)
                labels = np.ones((points.shape[0], 1)) * self.s3dis.name2label[class_name]
                points_list.append(np.concatenate([points, labels], axis=1))  # Nx7
            
            data_label = np.concatenate(points_list, 0)
            coord_min = np.amin(data_label[:, 0:3], axis=0)
            data_label[:, 0:3] -= coord_min
            
            # save data
            os.makedirs(self.dataset_dir, exist_ok=True)
            output_filename = area_name + "_" + scene_name
            np.save(os.path.join(self.dataset_dir, output_filename + '.npy'), data_label)
        
    def is_available(self):
        return True


class S3DISDataset(Dataset):
    def __init__(self, data_root, split='train', test_area=5, loop=1, load_mode='voxel', cfg=None):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.loop = loop
        self.load_mode = load_mode
        self.cfg = cfg
        data_list = sorted(os.listdir(data_root))
        data_list = [os.path.splitext(item)[0] for item in data_list if 'Area_' in item]
        if split == 'train':
            self.data_list = [item for item in data_list if not 'Area_{}'.format(test_area) in item]
        else:
            self.data_list = [item for item in data_list if 'Area_{}'.format(test_area) in item]
        print("Totally {} samples in {} set.".format(len(self.data_list), split))

    def __getitem__(self, idx):
        idx = idx % len(self.data_list)
        # TODO 各种方式的数据加载
        if self.load_mode == 'voxel':
            from data_utils.load_voxel_mode import data_prepare
            data_path = os.path.join(self.data_root, self.data_list[idx] + '.' + self.cfg.file_type)
            raw_data = np.load(data_path)
            coord, feat, label = raw_data[:, 0:3], raw_data[:, 3:6], raw_data[:, 6]
            coord, feat, label = data_prepare(coord, feat, label, self.split, self.cfg.voxel_size, self.cfg.voxel_max, self.cfg.transform, self.cfg.shuffle_index)
        else:
            raise Exception('Dataset load mode \'{}\' not supported.'.format(self.load_mode))
        return coord, feat, label

    def __len__(self):
        return len(self.data_list) * self.loop

