import yaml
import argparse
import os
import fnmatch
import pickle
from utils.project_points import project_points
from utils.gen_offline_data import gen_data

class PandasetDataset():
    def __init__(self, cfg, split='train'):
        self.cfg = cfg
        self.split = split
        self.cams = cfg['cams']

    def set_split(self, split):
        self.sequences = self.cfg['sequences'][split]
        self.split = split

    def get_infos(self):
        infos = []
        for seq in self.sequences:
            print('%s Sequence: %s' % (self.split, seq))

            lidar_path = os.path.join(self.cfg['paths']['source'], seq, 'lidar')
            num_samples = len(fnmatch.filter(os.listdir(lidar_path), '*.pkl.gz'))

            info = [{'sequence': seq,
                     'frame_idx': {'cloud': str(idx), 'sem': str(idx)},
                     'calib':[{'sequence': seq,
                              'camera': c,
                              'idx': idx} for c in self.cams],
                     'cloud': os.path.join(self.cfg['paths']['source'], seq, 'lidar', ("{:02d}.pkl.gz".format(idx))),
                     'sem_image': [os.path.join(self.cfg['paths']['sem_images'],seq, 'camera', c, ("{:02d}.png".format(idx))) for c in self.cams],
                     'sem2d': os.path.join(self.cfg['paths']['sem2d'], ("{}_{}.bin".format(seq, idx))),
                     'sem2d_scores': os.path.join(self.cfg['paths']['sem2d_scores'], ("{}_{}.bin".format(seq, idx))),
                     'sem3d': os.path.join(self.cfg['paths']['sem3d'], ("{}_{:02d}.bin".format(seq, idx))),
                     'pickle': os.path.join(self.cfg['paths']['pickle'], ("{}_{:02d}.pkl.gz".format(seq, idx))),
                     'gt': os.path.join(self.cfg['paths']['source'], seq, 'annotations/semseg', ("{:02d}.pkl.gz".format(idx)))
                    } for idx in range(num_samples)]
            infos.extend(info)

        return infos

class CarlaDataset():
    def __init__(self, cfg, split='train'):

        self.cfg = cfg
        self.split = split
        self.cams = cfg['cams']
        self.frames = {}

    def set_split(self, split):
        self.sequences = self.cfg['sequences'][split]
        self.split = split

        for seq in self.sequences:
            aux = []
            file = os.path.join(self.cfg['paths']['source'], f'frames_{seq}.txt')
            with open(file, 'r') as f:
                lines = f.readlines()
                for l in lines:
                    aux.append(l.strip().split(' '))
            self.frames[seq] = aux

    def get_infos(self):
        infos = []
        for seq in self.sequences:
            print('%s Sequence: %s' % (self.split, seq))

            num_samples = len(self.frames[seq])

            info = [{'sequence': seq,
                     'frame_idx': {'cloud': self.frames[seq][idx][0], 'sem': self.frames[seq][idx][1]},
                     'calib': [os.path.join(self.cfg['paths']['source'], 'calibs') for c in self.cams],
                     'cloud': os.path.join(self.cfg['paths']['source'], 'cloud', seq, f'{self.frames[seq][idx][0]}.bin'),
                     'sem_image': [os.path.join(self.cfg['paths']['sem_images'], seq, c, f'{self.frames[seq][idx][1]}.png') for c in self.cams],
                     'sem2d': os.path.join(self.cfg['paths']['sem2d'], ("{}_{}.bin".format(seq, idx))),
                     'sem2d_scores': os.path.join(self.cfg['paths']['sem2d_scores'], ("{}_{}.bin".format(seq, idx))),
                     'sem3d': os.path.join(self.cfg['paths']['sem3d'], seq, f'{idx}.bin'),
                     'pickle': os.path.join(self.cfg['paths']['pickle'], ("{}_{}.pkl.gz".format(seq, idx))),
                    } for idx in range(num_samples)]
            infos.extend(info)

        return infos

class KittiDataset():
    def __init__(self, cfg, split='train'):
        self.cfg = cfg
        self.split = split
        self.cams = cfg['cams']

    def set_split(self, split):
        self.sequences = self.cfg['sequences'][split]
        self.split = split

    def get_infos(self):
        infos = []
        for seq in self.sequences:
            print('%s Sequence: %s' % (self.split, seq))

            lidar_path = os.path.join(self.cfg['paths']['source'], seq, 'velodyne')
            num_samples = len(fnmatch.filter(os.listdir(lidar_path), '*.bin'))

            info = [{'sequence': seq,
                     'frame_idx': {'cloud': str(idx), 'sem': str(idx)},
                     'calib':[{'sequence': seq,
                              'camera': c,
                              'idx': idx} for c in self.cams],
                     'cloud': os.path.join(self.cfg['paths']['source'], seq, 'velodyne', ("{:06d}.bin".format(idx))),
                     'sem_image': [os.path.join(self.cfg['paths']['sem_images'],seq, 'image_2', ("{:06d}.png".format(idx))) for c in self.cams],
                     'sem2d': os.path.join(self.cfg['paths']['sem2d'], ("{}_{:06d}.bin".format(seq, idx))),
                     'sem2d_scores': os.path.join(self.cfg['paths']['sem2d_scores'], ("{}_{:06d}.bin".format(seq, idx))),
                     'sem3d': os.path.join(self.cfg['paths']['sem3d'], ("{}_{:06d}.bin".format(seq, idx))),
                     'pickle': os.path.join(self.cfg['paths']['pickle'], ("{}_{:06d}.pkl.gz".format(seq, idx))),
                     'gt': os.path.join(self.cfg['paths']['source'], seq, 'labels', ("{:06d}.label".format(idx)))
                    } for idx in range(num_samples)]
            infos.extend(info)

        return infos

# Dataloaders
datasets = {
    'pandaset': PandasetDataset,
    'carla': CarlaDataset,
    'kitti': KittiDataset
}

def create_infos(dataset_cfg, save_path):
    name = cfg['training_params']['name']
    dataset = datasets[name](cfg=dataset_cfg)
    for split in ['train', 'val']:
        print("---------------- Start to generate {} data infos ---------------".format(split))
        dataset.set_split(split)
        infos = dataset.get_infos()
        file_path = os.path.join(save_path, '{}_infos_{}.pkl'.format(name, split))
        with open(file_path, 'wb') as f:
            pickle.dump(infos, f)
        print('{} info {} file is saved to {}'.format(name, split, file_path))

        if args.build_sem2d:
            print('Building {} pointclouds with 2d labels'.format(split))
            project_points(infos, cfg['color_map'], name)

    if args.offline_data:
        print('Saving voxel input data')
        gen_data(cfg)

    print('---------------Data preparation Done---------------')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/pandaset.yaml', help='Configs path')
    parser.add_argument('--save_path', type=str, default='datasets/pandaset/out', help='Output path')
    parser.add_argument('--build_sem2d', action='store_true', help='Whether to build pointclouds with labels from 2d semantic images')
    parser.add_argument('--offline_data', action='store_true', help='Whether save voxels to pickle files')

    args = parser.parse_args()

    with open(args.config_path, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    create_infos(
        dataset_cfg=cfg,
        save_path=args.save_path
    )
