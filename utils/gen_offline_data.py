from dataloader.point_loader import PointLoader
from torch.utils.data import DataLoader
import torch
import pickle
import gzip
from tqdm import tqdm
import os

def gen_data(cfg):
    # Load configs
    name = cfg['training_params']['name']
    batch_size = 1 # batch size set to 1 to save individual files per frame
    voxel_size = cfg['training_params']['voxel_size']
    max_num_points = cfg['training_params']['max_num_points']
    max_voxels = cfg['training_params']['max_voxels']
    input_size = cfg['training_params']['input_size']
    num_classes = cfg['training_params']['num_classes']
    pc_range = cfg['training_params']['pc_range']
    data_train = cfg['paths']['data_train']
    data_val = cfg['paths']['data_val']
    gt_map = cfg['gt_map']
    sem_map = cfg['sem_map']
    out = cfg['paths']['pickle']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_train = PointLoader(name, data_train, voxel_size, max_num_points, max_voxels, input_size, num_classes, pc_range, gt_map, sem_map, device)
    dataset_val = PointLoader(name, data_val, voxel_size, max_num_points, max_voxels, input_size, num_classes, pc_range, gt_map, sem_map, device)

    trainloader = DataLoader(dataset_train, batch_size, shuffle=False, num_workers=0)
    valloader = DataLoader(dataset_val, batch_size, shuffle=False, num_workers=0)

    print('Train voxels')
    pbar = tqdm(total=len(trainloader))
    for _, d in enumerate(trainloader):
        seq = d['misc']['seq'][0]
        idx = d['misc']['frame']['cloud'][0]
        if name == 'pandaset':
            idx = idx.zfill(2)
        elif name == 'kitti':
            idx = idx.zfill(6)

        filename = f'{seq}_{idx}.pkl.gz'
        for k in d.keys():
            if k != 'misc':
                # Remove batch dim
                d[k] = torch.squeeze(d[k], 0)

        with gzip.open(os.path.join(out, filename), 'wb') as pkl:
            pickle.dump(d, pkl, protocol=pickle.HIGHEST_PROTOCOL)

        pbar.update(1)
    pbar.close()

    print('Val voxels')
    pbar = tqdm(total=len(valloader))
    for _, d in enumerate(valloader):
        seq = d['misc']['seq'][0]
        idx = d['misc']['frame']['cloud'][0]
        if name == 'pandaset':
            idx = idx.zfill(2)
        elif name == 'kitti':
            idx = idx.zfill(6)

        filename = f'{seq}_{idx}.pkl.gz'
        for k in d.keys():
            if k != 'misc':
                # Remove batch dim
                d[k] = torch.squeeze(d[k], 0)

        with gzip.open(os.path.join(out, filename), 'wb') as pkl:
            pickle.dump(d, pkl, protocol=pickle.HIGHEST_PROTOCOL)

        pbar.update(1)
    pbar.close()