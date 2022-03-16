import numpy as np
np.random.seed(1)
import pickle
import pandas as pd

# Pytorch
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class PointLoader(Dataset):
    def __init__(self, data, voxel_size, max_num_points, max_voxels, input_size, num_classes, pc_range, gt_map, sem_map):

        infos = []
        with open(data, 'rb') as f:
            self.infos = pickle.load(f)

        self.voxel_size = voxel_size
        self.max_num_points = max_num_points
        self.max_voxels = max_voxels
        self.input_size = input_size
        self.num_classes = num_classes
        self.pc_range = pc_range
        self.gt_map = gt_map
        self.sem_map = sem_map

    def __len__(self):
        return np.shape(self.infos)[0]

    def __getitem__(self, idx):
        pd_cloud = pd.read_pickle(self.infos[idx]['cloud'])

        # Use 360 lidar
        pd_cloud = pd_cloud[pd_cloud.d == 0]
        raw_cloud = pd_cloud.to_numpy()
        # xyz values only
        raw_cloud = raw_cloud[:, :3]

        sem2d = np.fromfile(self.infos[idx]['sem2d'], dtype=np.uint8)
        sem2d = np.vectorize(self.sem_map.__getitem__)(sem2d)

        sem3d = np.fromfile(self.infos[idx]['sem3d'], dtype=np.uint8).reshape(-1, 2)
        # Files contain scores too, we only want classes
        sem3d = sem3d[:,0]
        sem3d = np.vectorize(self.sem_map.__getitem__)(sem3d)

        sem2d_onehot = np.zeros((sem2d.shape[0], self.num_classes))
        sem2d_onehot[np.arange(sem2d.shape[0]),sem2d] = 1
        sem3d_onehot = np.zeros((sem3d.shape[0], self.num_classes))
        sem3d_onehot[np.arange(sem3d.shape[0]),sem3d] = 1

        # Concat vectors
        aux = np.concatenate((raw_cloud, sem2d_onehot), 1)
        data = np.concatenate((aux, sem3d_onehot), 1)

        # Ground truth
        gt = pd.read_pickle(self.infos[idx]['gt']).to_numpy()
        gt = gt[pd_cloud.index.to_numpy()].astype(np.uint8).squeeze(1)

        gt = np.vectorize(self.gt_map.__getitem__)(gt)

        gt_onehot = np.zeros((gt.shape[0], self.num_classes))
        gt_onehot[np.arange(gt.shape[0]),gt] = 1

        # Filter data with voxels
        input_data, input_gt = self.generate_voxels(data, gt_onehot)

        # Tensors
        input_data = torch.from_numpy(input_data)
        input_gt = torch.from_numpy(input_gt)

        train = {'input_data': input_data.float(), 'gt': input_gt.float()}

        return train

    def generate_voxels(self, data, gt):
        # data: [x,y,z,sem2d,sem3d]
        # gt: [gt sem]
        # Obtain number of voxels in xy
        minxyz = [self.pc_range[0], self.pc_range[2], self.pc_range[4]]

        grid_size_x = np.floor((self.pc_range[1] - self.pc_range[0]) / self.voxel_size[0]).astype(int)
        grid_size_y = np.floor((self.pc_range[3] - self.pc_range[2]) / self.voxel_size[1]).astype(int)
        grid_size_z = np.floor((self.pc_range[5] - self.pc_range[4]) / self.voxel_size[2]).astype(int)

        grid_size = [grid_size_x, grid_size_y, grid_size_z]

        # Compare the number of voxels with max allowed.
        # If it is higher, use it.
        if grid_size[0]*grid_size[1]*grid_size[2] > self.max_voxels:
            n_voxels = grid_size[0]*grid_size[1]*grid_size[2]
        # If it is lower use max number.
        else:
            n_voxels = self.max_voxels

        voxelgrid = -np.ones(grid_size, dtype=np.int32)
        voxels = np.zeros((n_voxels, self.max_num_points, self.input_size))
        voxels_gt = np.zeros((n_voxels, self.max_num_points, self.num_classes))
        num_points = np.zeros(n_voxels, dtype=np.int32)

        num_voxels = 0
        for i in range(data.shape[0]):
            # Transform coords to voxel space
            cv = np.floor((data[i,:3] - minxyz) / self.voxel_size).astype(int)

            # Ignore points outside of range
            if np.any(cv < minxyz) or np.any(cv >= grid_size):
                continue

            voxelid = voxelgrid[cv[0], cv[1], cv[2]]
            # Case 1: new voxel
            if voxelid == -1:
                voxelid = num_voxels
                voxelgrid[cv[0], cv[1], cv[2]] = num_voxels
                num_voxels += 1

            # Case 2: existing voxel, check current number of points
            if num_points[voxelid] < self.max_num_points:
                voxels[voxelid, num_points[voxelid]] = data[i]
                voxels_gt[voxelid, num_points[voxelid]] = gt[i]
                num_points[voxelid] += 1

        # If the number of voxels exceeds max allowed, use random sampling.
        if voxels.shape[0] > self.max_voxels:
            rs = np.random.choice(voxels.shape[0], size=self.max_voxels, replace=False)
            voxels = voxels[rs]
            voxels_gt = voxels_gt[rs]

        # Final shape: Voxel x [xyz, sem2d, sem3d],
        #                      [xyz, sem2d, sem3d],
        #                      [xyz, sem2d, sem3d]

        return voxels, voxels_gt
