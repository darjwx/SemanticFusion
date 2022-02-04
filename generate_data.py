class PandasetDataset():
    def __init__(self, cfg, split='train', root_path=None):
        self.cfg = cfg
        self.dataset = ps.DataSet(os.path.join(root_path, 'data'))
        self.split = split

    def set_split(self, split):
        self.sequences = self.cfg['sequences'][split]
        self.split = split

    def get_infos(self):
        infos = []
        for seq in self.sequences:
            print('%s Sequence: %s' % (self.split, seq))
            s = self.dataset[seq]
            s.load_lidar()

            info = [{'sequence': seq,
                     'frame_idx': idx,
                     'cloud': os.path.join(self.cfg['paths']['source'], 'data', seq, 'lidar', ("{:02d}.pkl.gz".format(idx))),
                     'sem2d': os.path.join(self.cfg['paths']['sem2d'], ("{}_{}.bin".format(seq, idx))),
                     'sem3d': os.path.join(self.cfg['paths']['sem3d'], ("{}_{:02d}.bin".format(seq, idx))),
                     'gt': os.path.join(self.cfg['paths']['source'], 'data', seq, 'annotations/semseg', ("{:02d}.pkl.gz".format(idx)))
                    } for idx in range(len(s.lidar.data))]
            infos.extend(info)

        return infos

def create_pandaset_infos(dataset_cfg, data_path, save_path):
    dataset = PandasetDataset(cfg=dataset_cfg, root_path=data_path)
    for split in ['train', 'val']:
        print("---------------- Start to generate {} data infos ---------------".format(split))
        dataset.set_split(split)
        infos = dataset.get_infos()
        file_path = os.path.join(save_path, 'pandaset_infos_{}.pkl'.format(split))
        with open(file_path, 'wb') as f:
            pickle.dump(infos, f)
        print("Pandaset info {} file is saved to {}".format(split, file_path))

    print('---------------Data preparation Done---------------')



if __name__ == '__main__':
    import yaml
    import argparse
    import os
    import sys
    import pickle

    sys.path.append('/home/darjimen/pandaset-devkit/python')
    import pandaset as ps

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/pandaset.yaml', help='Configs path')
    parser.add_argument('--data_path', type=str, default='datasets/pandaset/data', help='Data path')
    parser.add_argument('--save_path', type=str, default='datasets/pandaset/out', help='Output path')
    args = parser.parse_args()

    with open(args.config_path, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    create_pandaset_infos(
        dataset_cfg=cfg,
        data_path=args.data_path,
        save_path=args.save_path
    )
