import torch
from torch import nn
import spconv.pytorch as spconv
import numpy as np

class PointNetConv(nn.Module):
    def __init__(self, m, n, k_size=None):
        super(PointNetConv, self).__init__()
        self.conv1 = nn.Conv2d(m, n, 1, bias=False)
        self.bn = nn.BatchNorm2d(n)
        self.lr = nn.LeakyReLU()

        self.k_size = k_size
        if self.k_size != None:
            self.max = nn.MaxPool2d((1, k_size))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Conv: (batch) x voxel x ch x points
        x = self.conv1(x)
        x = self.lr(self.bn(x))

        if self.k_size != None:
            x = self.max(x)

        return x

class PointNet(nn.Module):
    def __init__(self, m, n, k_size):
        super(PointNet, self).__init__()
        self.fc1 = nn.Linear(m, n, bias=False)
        self.bn = nn.BatchNorm1d(n)
        self.lr = nn.LeakyReLU()

        self.k_size = k_size
        self.max = nn.MaxPool1d(k_size)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(mean=0.0, std=1.0)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # (batch) x voxel x ch
        x = self.fc1(x)
        # BatchNorm expects batch x ch x voxel
        x = x.transpose(1,2)
        x = self.max(self.lr(self.bn(x)))

        return x

class Model(nn.Module):
    def __init__(self, m, num_points, max_voxels, sparse_shape, batch_size,in_filters=[16, 32, 64], out_filters=[32, 64, 128]):
        super(Model, self).__init__()
        self.num_points = num_points
        self.max_voxels = max_voxels
        self.sparse_shape = sparse_shape
        self.batch_size = batch_size

        # PointNets
        # PN1: Point features
        # PN2: Voxels features
        # PN3: Global features
        self.pn1 = PointNetConv(m, 16)
        # 3D conv layers
        self.conv3d = []
        for i in range(len(in_filters)):
            self.conv3d.append(spconv.SparseSequential(
                spconv.SubMConv3d(in_filters[i], out_filters[i], 3, bias=False, algo=spconv.ConvAlgo.Native),
                nn.BatchNorm1d(out_filters[i]),
                nn.LeakyReLU(),
            ))
        self.conv3d = nn.ModuleList(self.conv3d)
        self.max = nn.MaxPool2d((1,self.num_points))
        self.pn3 = PointNet(128, 256, self.max_voxels)

        # Conv layers
        # 192: 64+128+256 - point + voxel + global features
        self.conv1 = nn.Conv2d(400, 64, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.lr = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(64, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(mean=0.0, std=1.0)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


    def forward(self, x, coors):
        # x: (batch) x voxel x points x ch
        x = x.permute(0,3,1,2)
        # PointNetConv: (batch) x ch x voxels x points
        x1 = self.pn1(x)
        # PointNetConv with max pool -> (batch) x ch x voxels

        # Add batch idx, needed for SparseConveTensor
        bs = np.arange(self.batch_size)
        batch_idx = np.repeat(bs, x1.size(2))
        coors[:,:,0] = torch.from_numpy(batch_idx).reshape(self.batch_size,x1.size(2))

        # Sparse tensor and conv3d
        exp_voxels = spconv.SparseConvTensor(x1.reshape(-1,16), coors.reshape(-1,4), self.sparse_shape, self.batch_size)
        for conv in self.conv3d:
            exp_voxels = conv(exp_voxels)

        f = exp_voxels.features.view(self.batch_size,self.max_voxels,self.num_points,128).transpose(3,2)
        f = self.max(f)

        # PointNet for global features -> (batch) x ch
        x3 = self.pn3(f.squeeze(-1))

        # Expand pn2 (batch x 128 x voxels) and pn3 (batch x 256) to match pn1
        f = f.transpose(1,2)
        f = f.expand(-1, 128, self.max_voxels, self.num_points)

        x3 = x3.unsqueeze(-1).expand(-1, 256, self.max_voxels, self.num_points)

        # Concat pn1, pn2 and pn3 outputs
        y = torch.cat((x1,f,x3), dim=1)

        # Conv + BatchNorm + ReLU
        # (batch) x ch x voxel x points
        # ch = 64 (pn1) + 128 (pn2) + 256 (pn3)
        y = self.conv1(y)
        y = self.lr(self.bn1(y))

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
