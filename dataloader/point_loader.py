import numpy as np
import pickle
import pandas as pd

# Pytorch
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class PointLoader(Dataset):
    def __init__(self, data, voxel_size, max_num_points, max_voxels, input_size, num_classes):

        infos = []
        with open(data, 'rb') as f:
            self.infos = pickle.load(f)

        self.voxel_size = voxel_size
        self.max_num_points = max_num_points
        self.max_voxels = max_voxels
        self.input_size = input_size
        self.num_classes = num_classes

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
        sem3d = np.fromfile(self.infos[idx]['sem3d']).astype(np.uint8)
        sem2d_onehot = np.zeros((sem2d.shape[0], self.num_classes+1))
        sem2d_onehot[np.arange(sem2d.shape[0]),sem2d] = 1
        sem3d_onehot = np.zeros((sem3d.shape[0], self.num_classes+1))
        sem3d_onehot[np.arange(sem3d.shape[0]),sem3d] = 1

        # Concat vectors
        aux = np.concatenate((raw_cloud, sem2d_onehot), 1)
        data = np.concatenate((aux, sem3d_onehot), 1)

        # Ground truth
        gt = pd.read_pickle(self.infos[idx]['gt']).to_numpy()
        gt = gt[pd_cloud.index].astype(np.uint8).reshape(-1)
        gt_onehot = np.zeros((gt.shape[0], self.num_classes+1))
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
        minx = np.min(data[:,0])
        miny = np.min(data[:,1])
        minxy = [minx, miny]

        grid_size_x = np.floor((np.max(data[:,0]) - np.min(data[:,0])) / self.voxel_size).astype(int)
        grid_size_y = np.floor((np.max(data[:,1]) - np.min(data[:,1])) / self.voxel_size).astype(int)
        grid_size = [grid_size_x+1, grid_size_y+1]

        voxelgrid = -np.ones(grid_size, dtype=np.int32)
        voxels = np.zeros((grid_size_x*grid_size_y, self.max_num_points, self.input_size))
        voxels_gt = np.zeros((grid_size_x*grid_size_y, self.max_num_points, 43))
        num_points = np.zeros(grid_size_x*grid_size_y, dtype=np.int32)
        num_voxels = 0
        for i in range(data.shape[0]):
            # Transform coords to voxel space
            cv = np.floor((data[i,:2] - minxy) / self.voxel_size).astype(int)
            voxelid = voxelgrid[cv[0], cv[1]]
            if voxelid == -1:
                voxelid = num_voxels
                num_voxels += 1

            if num_points[voxelid] < self.max_num_points:
                voxels[voxelid, num_points[voxelid]] = data[i]
                voxels_gt[voxelid, num_points[voxelid]] = gt[i]
                num_points[voxelid] += 1

        # Choose voxels randomly according to max number
        rs = np.random.choice(voxels.shape[0], size=self.max_voxels, replace=False)
        voxels = voxels[rs]
        voxels_gt = voxels_gt[rs]

        # Final shape: Voxel x [xyz, sem2d, sem3d],
        #                      [xyz, sem2d, sem3d],
        #                      [xyz, sem2d, sem3d]

        return voxels, voxels_gt
