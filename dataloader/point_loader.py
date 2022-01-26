import numpy as np
import pickle

#Pytorch
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class PointLoader(Dataset):
    def __init__(self, data):

        infos = []
        with open(data, 'rb') as f:
            self.infos = pickle.load(f)

    def __len__(self):
        return np.shape(self.infos)[0]

    def __getitem__(self, idx):
        raw_cloud = np.fromfile(self.infos[idx]['cloud']).reshape(-1, 3)
        sem2d = np.fromfile(self.infos[idx]['sem2d'])
        sem3d = np.fromfile(self.infos[idx]['sem3d'])

        #Tensors
        sem2d_tensor = torch.from_numpy(sem2d)
        sem3d_tensor = torch.from_numpy(sem3d)
        raw_cloud_tensor = torch.from_numpy(raw_cloud)

        #One-hot vectors
        sem2d_onehot = F.one_hot(sem2d_tensor, num_classes=13)
        sem3d_onehot = F.one_hot(sem3d_tensor, num_classes=13)

        #Concat vectors
        aux = torch.cat(raw_cloud, sem2d_onehot, 1)
        self.input_data = torch.cat((aux, sem3d_onehot), 1)

        train = {'input_data': input_data, 'raw_cloud': raw_cloud_tensor, 'sem2d': sem2d_onehot, 'sem3d': sem3d_onehot}

        return train
