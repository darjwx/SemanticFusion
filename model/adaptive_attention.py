import torch
from torch import nn
import torch.nn.functional as F

class PointNet(nn.Module):
    def __init__(self, m, n, k_size):
        super(PointNet, self).__init__()
        self.fc1 = nn.Linear(m, n)
        self.bn = nn.BatchNorm1d(n)
        self.max = nn.MaxPool1d(k_size)

    def forward(self, x):
        x = self.fc1(x)
        # BatchNorm expects batch x ch x voxel
        x = x.transpose(1,2)
        x = self.max(F.relu(self.bn(x)))
        # Go back to normal shape
        x = x.transpose(1,2)

        return x

class Model(nn.Module):
    def __init__(self, m, num_points, num_voxels):
        super(Model, self).__init__()
        # PointNets
        self.pn1 = PointNet(m, 64, num_points)
        self.pn2 = PointNet(64, 128, num_voxels)

        # Mlp
        # 192: 64+128 - local features + global features
        self.fc1 = nn.Linear(192, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 1)
        self.bn2 = nn.BatchNorm1d(1)

    def forward(self, x):
        # x: (batch) x voxel x points x ch
        x1 = []
        # Mlp at voxel level
        for i in range(x.size(1)):
            x1.append(self.pn1(x[:,i]).squeeze(1))
        # List -> tensor
        x1 = torch.stack(x1)
        # voxel x batch -> batch x voxel
        x1 = x1.transpose(0,1)

        # (batch) x voxel x ch
        x2 = self.pn2(x1).squeeze(1)

        # Expand pn2 output (batch x 128) to match pn1
        # and concat both tensors
        x2 = x2.unsqueeze(1).expand(-1, x1.size(1), 128)

        y = torch.cat((x1,x2), dim=2)

        # Mlp
        # (batch) x voxel x ch
        # ch = 64 (pn1) + 128 (pn2)
        y = self.fc1(y)
        # Transpose dims for batch norm
        y = y.transpose(1,2)
        y = self.bn1(y)
        # Go back to normal shape
        y = y.transpose(1,2)

        # Mlp with Sigmoid activation
        # (batch) x voxel x ch(64)
        y = self.fc2(y)
        # Transpose dims for batch norm
        y = y.transpose(1,2)
        y = torch.sigmoid(self.bn2(y))
        # Go back to normal shape
        y = y.transpose(1,2)
        att_mask = y.squeeze(2)

        # (batch) x voxel (ch = 1)
        return att_mask
