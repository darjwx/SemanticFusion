# Copyright 2023 Darío Jiménez Juiz
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import nn
import spconv.pytorch as spconv
import numpy as np
from spconv.pytorch.utils import gather_features_by_pc_voxel_id

class PointNetConv(nn.Module):
    def __init__(self, m, n, k_size=None):
        super(PointNetConv, self).__init__()

        self.mlp = []
        for f1, f2 in zip(m,n):
            self.mlp.append(nn.Sequential(
                nn.Conv2d(f1, f2, 1, bias=False),
                nn.LeakyReLU()
            ))
        self.mlp = nn.ModuleList(self.mlp)

        self.k_size = k_size
        if self.k_size != None:
            self.max = nn.MaxPool2d((1, k_size))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")

    def forward(self, x):
        # Conv: (batch) x voxel x ch x points
        for m in self.mlp:
            x = m(x)

        if self.k_size != None:
            x = self.max(x)

        return x

class Model(nn.Module):
    def __init__(self, m, num_points, max_voxels, sparse_shape, unified_mask=False, in_filters=[128, 256, 128], out_filters=[256, 128, 128]):
        super(Model, self).__init__()
        self.num_points = num_points
        self.max_voxels = max_voxels
        self.sparse_shape = sparse_shape
        self.unified_mask = unified_mask

        # PointNets
        # PN1: Point features
        # PN2: Voxels features
        self.pn1 = PointNetConv([m,32,64], [32,64,128])
        # 3D conv layers
        self.max = nn.MaxPool2d((1,self.num_points))

        self.subm1 = spconv.SubMConv3d(128, 256, 3, bias=False, indice_key='sm1')
        self.act1 = nn.LeakyReLU()
        self.subm2 = spconv.SubMConv3d(256, 128, 3, bias=False, indice_key='sm2')
        self.act2 = nn.LeakyReLU()
        self.subm3 = spconv.SubMConv3d(128, 256, 3, bias=False, indice_key='sm3')
        self.act3 = nn.LeakyReLU()
        self.subm4 = spconv.SubMConv3d(256, 128, 3, bias=False, indice_key='sm4')
        self.act4 = nn.LeakyReLU()

        # Conv layers
        # 192: 64+128+256 - point + voxel + global features
        self.conv1 = nn.Linear(256, 256, bias=False)
        self.lr = nn.LeakyReLU()
        self.conv2 = nn.Linear(256, 128, bias=False)
        self.lr2 = nn.LeakyReLU()
        if self.unified_mask:
            self.fc1 = nn.Linear(128, 1, bias=False)
            self.sigmoid = nn.Sigmoid()
        else:
            self.fc1 = nn.Linear(128, 4, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear)):
            m.weight.data.normal_(mean=0.0, std=1.0)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, (nn.Conv2d)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")


    def forward(self, x, coors, pc_voxel_id):
        # x: (batch) x voxel x points x ch
        x = x.permute(0,3,1,2)
        # PointNetConv: (batch) x ch x voxels x points
        x1 = self.pn1(x)
        # PointNetConv with max pool -> (batch) x ch x voxels

        # Add batch idx, needed for SparseConveTensor
        batch_size = coors.size(0)
        bs = np.arange(batch_size)
        batch_idx = np.repeat(bs, x1.size(2))
        coors[:,:,0] = torch.from_numpy(batch_idx).reshape(batch_size, x1.size(2))

        # Sparse tensor and conv3d

        aux_x1 = self.max(x1).squeeze(-1)
        exp_voxels = spconv.SparseConvTensor(aux_x1.permute(0,2,1).reshape(-1,128), coors.reshape(-1,4), self.sparse_shape[::-1], batch_size)

        shortcut = self.subm1(exp_voxels)
        shortcut = shortcut.replace_feature(self.act1(shortcut.features))
        shortcut = self.subm2(shortcut)
        shortcut = shortcut.replace_feature(self.act2(shortcut.features))
        res = self.subm3(shortcut)
        res = res.replace_feature(self.act3(res.features))
        res = self.subm4(res)
        res = res.replace_feature(self.act4(res.features))
        res = res.replace_feature(res.features + shortcut.features)

        # # Expand pn2 (batch x 128 x voxels) and pn3 (batch x 256) to match pn1
        x1 = gather_features_by_pc_voxel_id(aux_x1.squeeze().permute(1,0), pc_voxel_id.view(-1))
        x2 = gather_features_by_pc_voxel_id(res.features, pc_voxel_id.view(-1))

        # Concat pn1, pn2 and pn3 outputs
        y = torch.cat((x1,x2), dim=1)

        # Conv + BatchNorm + LeakyReLU
        # (batch) x ch x voxel x points
        # ch = 64 (pn1) + 128 (pn2) + 256 (pn3)
        y = self.conv1(y)
        y = self.lr(y)

        # Conv layer with Sigmoid activation
        # (batch) x ch(64) x voxel x points
        y = self.conv2(y)
        y = self.lr2(y)

        att_mask = self.fc1(y)

        if self.unified_mask:
            att_mask = self.sigmoid(att_mask)

        return att_mask
