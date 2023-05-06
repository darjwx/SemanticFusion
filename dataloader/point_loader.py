import numpy as np
np.random.seed(1)
import pickle
import pandas as pd
from numba import jit

# Pytorch
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# Pandaset calibration
from utils import pandaset_util as ps_util

from utils import video_utils as vu

# Dataset's data functions

# Pandaset: load gt and raw_cloud data
def pd_data(idx, infos):
    pd_cloud = pd.read_pickle(infos[idx]['cloud'])

    # Calib hardcoded to back camera. We do not care about the sensor
    # in the ego2lidar transformation
    calib = ps_util.PandasetCalibration('datasets/pandaset/data/data', infos[idx]['calib'][0])

    # Use 360 lidar
    pd_cloud = pd_cloud[pd_cloud.d == 0]
    raw_cloud = pd_cloud.to_numpy()
    # xyz values only
    raw_cloud = raw_cloud[:, :3]
    # Pandaset default coord system is 'ego'.
    # We need 'lidar' to filter the points with
    # max range values.
    raw_cloud = calib.project_ego_to_lidar(raw_cloud)

    # Rotate velo
    raw_cloud = ps_util.rotate_velo(raw_cloud, np.pi/2)

    # Ground truth
    gt = pd.read_pickle(infos[idx]['gt']).to_numpy()
    gt = gt[pd_cloud.index.to_numpy()].astype(np.uint8).squeeze(1)

    return raw_cloud, gt

# Carla: load gt and raw_cloud data
def carla_data(idx, infos):
    cr_data = np.fromfile(infos[idx]['cloud'], dtype=np.float32).reshape(-1, 4)

    raw_cloud = cr_data[:, :3]
    gt = cr_data[:, 3]

    return raw_cloud, gt

def kitti_data(idx, infos):
    # Load lidar
    raw_cloud = np.fromfile(infos[idx]['cloud'], dtype=np.float32).reshape(-1, 4)[:, :3]

    # Load gt
    gt = np.fromfile(infos[idx]['gt'], dtype=np.uint32).reshape((-1))
    gt = gt & 0xFFFF  # Lower half have semantic labels

    # Ignore points outside of img fov
    import os
    from utils.kitti_utils import KittiCalibration
    calib = KittiCalibration(os.path.join('datasets/kitti/odometry/dataset/sequences/', infos[idx]['calib'][0]['sequence'], 'calib.txt'))

    pc_cam = calib.project_lidar_to_camera(raw_cloud)
    pc_img = calib.project_lidar_to_image(raw_cloud)
    _, fov_idx = vu.pc_in_image_fov(pc_img, pc_cam, [376,1241])

    raw_cloud = raw_cloud[fov_idx]
    gt = gt[fov_idx]

    return raw_cloud, gt, fov_idx

# Supported datasets
datasets = {
    'pandaset': pd_data,
    'carla': carla_data,
    'kitti': kitti_data
}

class PointLoader(Dataset):
    def __init__(self, dataset, data, voxel_size, max_num_points, max_voxels, input_size,\
                 num_classes, pc_range, gt_map, sem_map, device, offline=False):

        infos = []
        with open(data, 'rb') as f:
            self.infos = pickle.load(f)

        self.offline = offline
        if not self.offline:
            self.device = device
            self.voxel_size = np.array(voxel_size)
            self.max_num_points = max_num_points
            self.max_voxels = max_voxels
            self.input_size = input_size
            self.num_classes = num_classes
            self.pc_range = pc_range
            self.gt_map = gt_map
            self.sem_map = sem_map
            self.dataset = dataset

    def __len__(self):
        return np.shape(self.infos)[0]

    def __getitem__(self, idx):

        if not self.offline:
            from spconv.pytorch.utils import PointToVoxel, gather_features_by_pc_voxel_id
            raw_cloud, gt, fov = datasets[self.dataset](idx, self.infos)

            sem2d = np.fromfile(self.infos[idx]['sem2d'], dtype=np.float32).reshape(-1,2)
            sem2d = sem2d[fov]
            scores2d = sem2d[:,1]
            classes2d = sem2d[:,0].astype(np.uint8)

            sem3d = np.fromfile(self.infos[idx]['sem3d'], dtype=np.float32).reshape(-1, 2)
            sem3d = sem3d[fov]
            scores3d = sem3d[:,1]
            classes3d = sem3d[:,0]
            classes3d = np.vectorize(self.sem_map.__getitem__)(classes3d)

            # Onehot with scores
            sem2d_onehot = np.zeros((classes2d.shape[0], self.num_classes))
            sem2d_onehot[np.arange(classes2d.shape[0]),classes2d] = scores2d
            sem3d_onehot = np.zeros((classes3d.shape[0], self.num_classes))
            sem3d_onehot[np.arange(classes3d.shape[0]),classes3d] = scores3d

            gt = np.vectorize(self.gt_map.__getitem__)(gt)

            gt_onehot = np.zeros((gt.shape[0], self.num_classes))
            gt_onehot[np.arange(gt.shape[0]),gt] = 1

            # Concat vectors
            aux = np.concatenate((raw_cloud, sem2d_onehot), 1)
            data = np.concatenate((aux, sem3d_onehot), 1)
            data = np.concatenate((data, gt_onehot), 1)
            data = np.concatenate((data, classes3d[:, np.newaxis]), 1)

            gen = PointToVoxel(
                vsize_xyz=self.voxel_size,
                coors_range_xyz=self.pc_range,
                num_point_features=self.input_size+self.num_classes+1,
                max_num_voxels=self.max_voxels,
                max_num_points_per_voxel=self.max_num_points,
                device=self.device)
            voxels, coords, num_points, pc_voxel_id = gen.generate_voxel_with_id(torch.from_numpy(data).float().to(self.device), empty_mean=True)
            coords = F.pad(coords, ((1,0)), mode='constant', value=0)

            misc = {
                'seq': self.infos[idx]['sequence'],
                'frame': self.infos[idx]['frame_idx']
            }

            train = {'raw_cloud': raw_cloud, 'pc_voxel_id': pc_voxel_id, 'c3d': voxels[:,:,-1], 'input_data': voxels[:,:,:self.input_size], 'gt': voxels[:,:,self.input_size:self.input_size+self.num_classes], 'coors': coords, 'misc': misc}
        else:
            import pickle
            import gzip
            with gzip.open(self.infos[idx]['pickle'], 'rb') as pkl:
                train = pickle.load(pkl)

        return train