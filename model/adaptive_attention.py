import torch
from torch import nn
import torch.nn.functional as F

class PointNet(nn.Module):
    def __init__(self, m, n, k_size):
        super(PointNet, self).__init__()
        self.fc1 = nn.Linear(m, n)
        self.bn = nn.BatchNorm1d(n)

        self.k_size = k_size
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
    def __init__(self, m, num_points, max_voxels):
        super(Model, self).__init__()
        # PointNets
        self.pn1 = PointNet(m, 64, num_points)
        self.pn2 = PointNet(64, 128, max_voxels)

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
        for i in range(x.size(0)):
            x1.append(self.pn1(x[i]).squeeze(1))
        # List -> tensor
        x1 = torch.stack(x1)

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

def fusion_voxels(raw_cloud, sem2d, sem3d, att_mask):
    # a = 2d semantics * attention_mask
    # b = 3d semantics * (1 - attention mask)
    # c = concat -> raw cloud, b+a

    att_mask = att_mask.unsqueeze(2).expand(-1, -1, sem2d.size(3))

    # This approach assumes points inside voxels have the same class,
    # which is not always true.
    a = sem2d[:,:,0,:]
    a = a*att_mask

    b = sem3d[:,:,0,:]
    b = b*(1 - att_mask)

    aux = (b+a).unsqueeze(2).expand(-1,-1, raw_cloud.size(2), sem2d.size(3))
    c = torch.cat((raw_cloud, aux), dim=3)

    return c
