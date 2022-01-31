import numpy as np
import pickle
import pandas as pd

# Pytorch
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class PointLoader(Dataset):
    def __init__(self, data, voxel_size, max_num_points):

        infos = []
        with open(data, 'rb') as f:
            self.infos = pickle.load(f)

        self.voxel_size = voxel_size
        self.max_num_points = max_num_points

    def __len__(self):
        return np.shape(self.infos)[0]

    def __getitem__(self, idx):
        raw_cloud = np.fromfile(self.infos[idx]['cloud']).reshape(-1, 3)
        sem2d = np.fromfile(self.infos[idx]['sem2d'])
        sem3d = np.fromfile(self.infos[idx]['sem3d'])

        # Tensors
        sem2d_tensor = torch.from_numpy(sem2d)
        sem3d_tensor = torch.from_numpy(sem3d)
        raw_cloud_tensor = torch.from_numpy(raw_cloud)

        # One-hot vectors
        sem2d_onehot = F.one_hot(sem2d_tensor, num_classes=13)
        sem3d_onehot = F.one_hot(sem3d_tensor, num_classes=13)

        # Concat vectors
        aux = torch.cat(raw_cloud, sem2d_onehot, 1)
        data = torch.cat((aux, sem3d_onehot), 1)

        # Filter data with voxels
        ids = self.generate_voxels(raw_cloud)
        input_data = []
        for i in range(ids.shape[0]):
            input_data.append(data[ids[i]])
        input_data = torch.from_numpy(np.asarray(input_data))

        train = {'input_data': input_data, 'raw_cloud': raw_cloud_tensor, 'sem2d': sem2d_onehot, 'sem3d': sem3d_onehot}

        return train

    def generate_voxels(self, pc):
        # Obtain number of voxels in xy
        grid_size_x = np.floor(np.max(pc[:,0]) - np.min(pc[:,0]) / self.voxel_size).astype(int)
        grid_size_y = np.floor(np.max(pc[:,1]) - np.min(pc[:,1]) / self.voxel_size).astype(int)

        # Grid mask
        min_x = np.min(pc[:,0])
        min_y = np.min(pc[:,1])
        voxels = []

        for i in range(grid_size_x):
            for j in range(grid_size_y):
                amin_x = min_x + i*self.voxel_size
                amax_x = amin_x + self.voxel_size
                amin_y = min_y + j*self.voxel_size
                amax_y = amin_y + self.voxel_size

                mask = np.ones(pc.shape[0], dtype=bool)
                mask = np.logical_and(mask, pc[:, 0] > amin_x)
                mask = np.logical_and(mask, pc[:, 0] < amax_x)
                mask = np.logical_and(mask, pc[:, 1] > amin_y)
                mask = np.logical_and(mask, pc[:, 1] < amax_y)

                voxels.append(np.nonzero(mask)[0])

        # Filter voxels
        voxels = np.asarray(voxels, dtype=object)
        aux_del = []
        for i in range(voxels.shape[0]):
            n = len(voxels[i])
            # Check if there are points in each voxel.
            if n == 0:
                aux_del.append(i)
            # Filter with max points, if less append the last point until false
            elif n < self.max_num_points:
                while(len(voxels[i]) < self.max_num_points):
                    voxels[i] = np.append(voxels[i], voxels[i][-1])
            # If voxel exceeds max points, ignore those extra ones
            elif n > self.max_num_points:
                voxels[i] = voxels[i][0:self.max_num_points]

        # Delete extra points
        voxels = np.delete(voxels, aux_del, 0)

        # With returned ids, final shape can be built.
        # Final shape: Voxel x [xyz, sem2d, sem3d],
        #                      [xyz, sem2d, sem3d],
        #                      [xyz, sem2d, sem3d]

        return voxels
