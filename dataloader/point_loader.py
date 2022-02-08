import numpy as np
import pickle
import pandas as pd

# Pytorch
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class PointLoader(Dataset):
    def __init__(self, data, voxel_size, max_num_points, max_voxels):

        infos = []
        with open(data, 'rb') as f:
            self.infos = pickle.load(f)

        self.voxel_size = voxel_size
        self.max_num_points = max_num_points
        self.max_voxels = max_voxels

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
        sem3d = np.fromfile(self.infos[idx]['sem3d'])

        # Ground truth
        gt = pd.read_pickle(self.infos[idx]['gt']).to_numpy()
        gt = gt[pd_cloud.index].astype(np.uint8)
        gt_onehot = F.one_hot(torch.from_numpy(gt).long(), num_classes=43).squeeze(1)


        # Tensors
        sem2d_tensor = torch.from_numpy(sem2d)
        sem3d_tensor = torch.from_numpy(sem3d)
        raw_cloud_tensor = torch.from_numpy(raw_cloud)

        # One-hot vectors
        sem2d_onehot = F.one_hot(sem2d_tensor.long(), num_classes=34).double()
        sem3d_onehot = F.one_hot(sem3d_tensor.long(), num_classes=34).double()

        # Concat vectors
        aux = torch.cat((raw_cloud_tensor, sem2d_onehot), 1)
        data = torch.cat((aux, sem3d_onehot), 1)

        # Filter data with voxels
        input_data, input_gt = self.generate_voxels(raw_cloud, data, gt_onehot)

        train = {'input_data': input_data, 'gt': input_gt}

        return train

    def generate_voxels(self, pc, data, gt):
        # Obtain number of voxels in xy
        grid_size_x = np.floor(np.max(pc[:,0]) - np.min(pc[:,0]) / self.voxel_size).astype(int)
        grid_size_y = np.floor(np.max(pc[:,1]) - np.min(pc[:,1]) / self.voxel_size).astype(int)

        # Grid mask
        min_x = np.min(pc[:,0])
        min_y = np.min(pc[:,1])
        ids = []

        # Tensor to numpy
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

                ids.append(np.nonzero(mask)[0])

        aux_data = []
        aux_gt = []
        data = data.numpy()
        gt = gt.numpy()
        for i in range(np.shape(ids)[0]):
            aux_data.append(data[ids[i]])
            aux_gt.append(gt[ids[i]])

        voxels = np.asarray(aux_data, dtype=object)
        voxels_gt = np.asarray(aux_gt, dtype=object)

        empty = np.zeros((1,71))
        empty_gt = np.zeros((1, 43))
        for i in range(voxels.shape[0]):
            n = voxels[i].shape[0]
            # Check if there are points in each voxel.
            if n == 0 or n < self.max_num_points:
                while voxels[i].shape[0] < self.max_num_points:
                    voxels[i] = np.concatenate((voxels[i], empty))
                    voxels_gt[i] = np.concatenate((voxels_gt[i], empty_gt))
            # If voxel exceeds max points, ignore those extra ones
            elif n > self.max_num_points:
                rs = np.random.choice(voxels[i].shape[0], size=self.max_num_points, replace=False)
                voxels[i] = voxels[i][rs,:]
                voxels_gt[i] = voxels_gt[i][rs,:]

        # Filter voxels
        # Max number of voxels
        if voxels.shape[0] > self.max_voxels:
            # Random sampling
            rs = np.random.choice(voxels.shape[0], size=self.max_voxels, replace=False)
            voxels = voxels[rs]
            voxels_gt = voxels_gt[rs]
        elif voxels.shape[0] < self.max_voxels:
            # Add empty voxels
            empty = np.zeros(voxels[0].shape)
            empty_gt = np.zeros(voxels_gt[0].shape)
            while voxels.shape[0] < max_voxels:
                voxels = np.concatenate((voxels, empty))
                voxels_gt = np.concatenate((voxels_gt, empty_gt))

        # Final shape: Voxel x [xyz, sem2d, sem3d],
        #                      [xyz, sem2d, sem3d],
        #                      [xyz, sem2d, sem3d]

        # [n,] to [n,p,c]
        # object to float32
        voxels = np.rollaxis(np.dstack(voxels), -1).astype(np.float32)
        voxels_gt = np.rollaxis(np.dstack(voxels_gt), -1).astype(np.float32)

        return torch.from_numpy(voxels), torch.from_numpy(voxels_gt)
