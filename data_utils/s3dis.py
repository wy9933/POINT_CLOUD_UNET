import os
import glob
import tqdm
import numpy as np

class S3DIS():
    def __init__(self) -> None:
        """Predefined s3dis related properties
        """
        self.names = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter']
        self.name2label = {name: i for i, name in enumerate(self.names)}
        self.label2name = {i: name for i, name in enumerate(self.names)}
        self.name2color = {'ceiling':	    [0,   255, 0  ], 
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
    def __init__(self, original_dir, dataset_dir, file_type) -> None:
        self.original_dir = original_dir
        self.dataset_dir = dataset_dir
        self.file_type = file_type
        self.s3dis = S3DIS()

    def prepare(self) -> None:
        # check original data dir path
        scene_num = 272
        scene_list = glob.glob(os.path.join(self.original_dir, '*/*'))
        if len(scene_list) != scene_num:
            raise Exception('The number of scenes in S3DIS dataset is incorrect (have %d scenes, should have %d scenes), please check the dataset integrity' % (len(scene_list), scene_num))

        # check data file type
        legal_types = ['npy', 'txt']
        if self.file_type not in legal_types:
            raise Exception('Output file type \'%s\' not supported, please choose a type in: %s.' % (self.file_type, ', '.join(legal_types)))
        
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
            if self.file_type == 'npy':
                np.save(os.path.join(self.dataset_dir, output_filename + '.npy'), data_label)
            elif self.file_type == 'txt':
                np.savetxt(os.path.join(self.dataset_dir, output_filename + '.txt'), data_label, fmt='%f %f %f %d %d %d %d')
        
    def is_available(self):
        return True


if __name__ == "__main__":
    prepare = PrepareDataset("E:/dataset/s3dis/Stanford3dDataset_v1.2_Aligned_Version", "E:/dataset/s3dis/raw_data", 'npy')
    prepare.prepare()