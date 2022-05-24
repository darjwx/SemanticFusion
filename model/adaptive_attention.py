import torch
from torch import nn
import torch.nn.functional as F

class PointNetConv(nn.Module):
    def __init__(self, m, n, k_size=None):
        super(PointNetConv, self).__init__()
        self.conv1 = nn.Conv2d(m, n, 1, bias=False)
        self.bn = nn.BatchNorm2d(n)

        self.k_size = k_size
        if self.k_size != None:
            self.max = nn.MaxPool2d((1, k_size))

    def forward(self, x):
        # Conv: (batch) x voxel x ch x points
        x = self.conv1(x)
        x = F.relu(self.bn(x))

        if self.k_size != None:
            x = self.max(x)

        return x

class PointNet(nn.Module):
    def __init__(self, m, n, k_size):
        super(PointNet, self).__init__()
        self.fc1 = nn.Linear(m, n, bias=False)
        self.bn = nn.BatchNorm1d(n)

        self.k_size = k_size
        self.max = nn.MaxPool1d(k_size)

    def forward(self, x):
        # (batch) x voxel x ch
        x = self.fc1(x)
        # BatchNorm expects batch x ch x voxel
        x = x.transpose(1,2)
        x = self.max(F.relu(self.bn(x)))

        return x

class Model(nn.Module):
    def __init__(self, m, num_points, max_voxels):
        super(Model, self).__init__()
        self.num_points = num_points
        self.max_voxels = max_voxels

        # PointNets
        # PN1: Point features
        # PN2: Voxels features
        # PN3: Global features
        self.pn1 = PointNetConv(m, 64)
        self.pn2 = PointNetConv(64, 128, self.num_points)
        self.pn3 = PointNet(128, 256, self.max_voxels)

        # Conv layers
        # 192: 64+128+256 - point + voxel + global feeatures
        self.conv1 = nn.Conv2d(448, 64, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(1)

    def forward(self, x):
        # x: (batch) x voxel x points x ch
        x = x.permute(0,3,1,2)
        # PointNetConv: (batch) x ch x voxels x points
        x1 = self.pn1(x)
        # PointNetConv with max pool -> (batch) x ch x voxels
        x2 = self.pn2(x1)
        # PointNet for global features -> (batch) x ch
        x3 = self.pn3(x2.transpose(1,2).squeeze(-1))

        # Expand pn2 (batch x 128 x voxels) and pn3 (batch x 256) to match pn1
        x2 = x2.expand(-1, 128, self.max_voxels, self.num_points)
        x3 = x3.unsqueeze(-1).expand(-1, 256, self.max_voxels, self.num_points)


        # Concat pn1, pn2 and pn3 outputs
        y = torch.cat((x1,x2,x3), dim=1)

        # Conv + BatchNorm + ReLU
        # (batch) x ch x voxel x points
        # ch = 64 (pn1) + 128 (pn2) + 256 (pn3)
        y = self.conv1(y)
        y = F.relu(self.bn1(y))

        # Conv layer with Sigmoid activation
        # (batch) x ch(64) x voxel x points
        y =self.bn2(self.conv2(y))
        # (batch) x voxel x points (ch = 1)
        y = y.permute(0,2,3,1)
        att_mask = torch.sigmoid(y)

        return att_mask

def fusion_voxels(raw_cloud, sem2d, sem3d, att_mask):
    # a = 2d semantics * attention_mask
    # b = 3d semantics * (1 - attention mask)
    # c = concat -> raw cloud, b+a

    # att_mask -> (batch) x voxels x points x num_classes
    att_mask = att_mask.expand(-1, -1, -1, sem2d.size(3))

    a = sem2d*att_mask
    b = sem3d*(1 - att_mask)
    aux = (b+a)

    c = torch.cat((raw_cloud, aux), dim=3)

    return c
