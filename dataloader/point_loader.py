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

# Supported datasets
datasets = {
    'pandaset': pd_data,
    'carla': carla_data,
}

# Use jit compiler
# Using jit: 15m per epoch
# Plain python: 1h 10m per epoch
@jit(nopython=True)
def generate_voxels_numba(voxels_c3d, c3d, data, voxelgrid, voxels, voxels_gt, num_points, gt,\
                          min_grid_size, max_grid_size, minxyz, max_voxels, max_num_points,\
                          voxel_size, coors):
    # data: [x,y,z,sem2d,sem3d]
    # gt: [gt sem]
    # Obtain number of voxels in xyz

    for i in range(data.shape[0]):
        # Transform coords to voxel space
        cv = np.floor((data[i,:3] - minxyz.astype(np.float32)) / voxel_size.astype(np.float32)).astype(np.int32)

        # Ignore points outside of range
        if np.any(cv < min_grid_size) or np.any(cv >= max_grid_size):
            continue

        voxelid = voxelgrid[cv[2], cv[0], cv[1]]

        if num_points[voxelid] < max_num_points:
            voxels[voxelid, num_points[voxelid]] = data[i]
            voxels_gt[voxelid, num_points[voxelid]] = gt[i]
            voxels_c3d[voxelid, num_points[voxelid]] = c3d[i]

        num_points[voxelid] += 1

    # If the number of non-empty voxels exceeds max allowed, use random sampling.
    if num_points.nonzero()[0].shape[0] > max_voxels:
        # Prioratize voxels with more points
        idx = np.argsort(num_points)[::-1] # Sort ids: prioritize voxels with more points
        voxels = voxels[idx[0:max_voxels]]
        voxels_gt = voxels_gt[idx[0:max_voxels]]
        voxels_c3d = voxels_c3d[idx[0:max_voxels]]
        coors = coors[idx[0:max_voxels]]

        # Update point numbers after resampling
        num_points = num_points[idx[0:max_voxels]]
    else:
        idx = num_points.nonzero()[0]
        inv_idx = np.nonzero(num_points == 0)[0]

        # Zero padding needed
        pd = max_voxels - idx.shape[0]
        idx = np.append(idx, inv_idx[:pd])

        voxels = voxels[np.sort(idx)]
        voxels_gt = voxels_gt[np.sort(idx)]
        voxels_c3d = voxels_c3d[np.sort(idx)]
        coors = coors[np.sort(idx)]

        # Update point numbers after resampling
        num_points = num_points[np.sort(idx)]

    # Duplicate points instead of zero-padding
    mask = (num_points < max_num_points) & (num_points != 0)
    mask = mask.nonzero()[0]
    for i in mask:
        d = max_num_points - num_points[i]
        voxels[i,num_points[i]:max_num_points] = np.repeat(np.expand_dims(voxels[i,num_points[i]-1], axis=0), d).reshape(-1,d).T

    # Final shape: Voxel x [xyz, sem2d, sem3d],
    #                      [xyz, sem2d, sem3d],
    #                      [xyz, sem2d, sem3d]

    return voxels, voxels_gt, coors, voxels_c3d

class PointLoader(Dataset):
    def __init__(self, dataset, data, voxel_size, max_num_points, max_voxels, input_size,\
                 num_classes, pc_range, target_grid_size, gt_map, sem_map):

        infos = []
        with open(data, 'rb') as f:
            self.infos = pickle.load(f)

        self.voxel_size = np.array(voxel_size)
        self.max_num_points = max_num_points
        self.max_voxels = max_voxels
        self.input_size = input_size
        self.num_classes = num_classes
        self.pc_range = pc_range
        self.gt_map = gt_map
        self.sem_map = sem_map
        self.dataset = dataset

        # Voxels variables
        self.minxyz = np.array([self.pc_range[0], self.pc_range[2], self.pc_range[4]]).astype(int)
        self.grid_size_x = np.floor((self.pc_range[1] - self.pc_range[0]) / self.voxel_size[0]).astype(int)
        self.grid_size_y = np.floor((self.pc_range[3] - self.pc_range[2]) / self.voxel_size[1]).astype(int)
        self.grid_size_z = np.floor((self.pc_range[5] - self.pc_range[4]) / self.voxel_size[2]).astype(int)
        self.pc_grid_size = np.array([self.grid_size_x, self.grid_size_y, self.grid_size_z]).astype(int)

        self.target_grid_size = np.array(target_grid_size).astype(int)
        self.min_grid_size = ((self.pc_grid_size - self.target_grid_size)/2).astype(int)
        self.max_grid_size = (self.min_grid_size + self.target_grid_size).astype(int)
        self.grid_size = self.pc_grid_size[0]*self.pc_grid_size[1]*self.pc_grid_size[2]

    def __len__(self):
        return np.shape(self.infos)[0]

    def __getitem__(self, idx):

        raw_cloud, gt = datasets[self.dataset](idx, self.infos)

        sem2d = np.fromfile(self.infos[idx]['sem2d'], dtype=np.float32).reshape(-1,2)
        scores2d = sem2d[:,1]
        classes2d = sem2d[:,0].astype(np.uint8)

        sem3d = np.fromfile(self.infos[idx]['sem3d'], dtype=np.float32).reshape(-1, 2)
        scores3d = sem3d[:,1]
        classes3d = sem3d[:,0]
        classes3d = np.vectorize(self.sem_map.__getitem__)(classes3d)

        # Onehot with scores
        sem2d_onehot = np.zeros((classes2d.shape[0], self.num_classes))
        sem2d_onehot[np.arange(classes2d.shape[0]),classes2d] = scores2d
        sem3d_onehot = np.zeros((classes3d.shape[0], self.num_classes))
        sem3d_onehot[np.arange(classes3d.shape[0]),classes3d] = scores3d

        # Concat vectors
        aux = np.concatenate((raw_cloud, sem2d_onehot), 1)
        data = np.concatenate((aux, sem3d_onehot), 1)

        gt = np.vectorize(self.gt_map.__getitem__)(gt)

        gt_onehot = np.zeros((gt.shape[0], self.num_classes))
        gt_onehot[np.arange(gt.shape[0]),gt] = 1

        # Filter data with voxels
        input_data, input_gt, coors, voxels_c3d = self.generate_voxels(data, gt_onehot, classes3d)

        # Tensors
        input_data = torch.from_numpy(input_data)
        input_gt = torch.from_numpy(input_gt)
        voxels_c3d = torch.from_numpy(voxels_c3d)
        misc = {
            'seq': self.infos[idx]['sequence'],
            'frame': self.infos[idx]['frame_idx']
        }

        train = {'c3d': voxels_c3d.float(), 'input_data': input_data.float(), 'gt': input_gt.float(), 'coors': coors, 'misc': misc}

        return train

    def generate_voxels(self, data, gt, c3d):

        # Create an ordered voxelgrid.
        # Each position holds the voxel ID.
        ids = np.arange(start=0, stop=self.grid_size, step=1, dtype=np.int32)
        voxelgrid = ids.reshape((self.pc_grid_size[2], self.pc_grid_size[0], self.pc_grid_size[1]))

        # Get coords of each voxel
        coors = np.argwhere(voxelgrid != -1)
        # Add zero padding for batch dim
        coors = np.pad(coors, ((0,0),(1,0)), mode='constant', constant_values=0).astype(np.int32)

        voxels = np.zeros((self.grid_size, self.max_num_points, self.input_size), dtype=np.float32)
        voxels_c3d = np.zeros((self.grid_size, self.max_num_points), dtype=np.float32)
        voxels_gt = np.zeros((self.grid_size, self.max_num_points, self.num_classes), dtype=np.float32)
        num_points = np.zeros(self.grid_size, dtype=np.int32)

        # Call voxel generation with numba
        input_data, input_gt, coors, c3d_results = generate_voxels_numba(voxels_c3d, c3d, data, voxelgrid, voxels, voxels_gt,\
                                                     num_points, gt, self.min_grid_size, self.max_grid_size, self.minxyz,\
                                                     self.max_voxels, self.max_num_points, self.voxel_size, coors)

        return input_data, input_gt, coors, c3d_results
