from model.adaptive_attention import *
from dataloader.point_loader import PointLoader

import torch
torch.manual_seed(1)
from torch.utils.data import DataLoader
import torch.optim as optim

import yaml
import argparse
import os
from tqdm import tqdm

def build_loss(fusion, gt, num_classes, weights=None):

    labels_train = fusion[:, :, 0, 3:num_classes+3]

    gt = gt.squeeze(3)
    gt = gt[:, :, 0, 0:num_classes]
    _, labels_gt = torch.max(gt, dim=2)

    if weights is not None:
        loss = nn.CrossEntropyLoss(weight=weights, ignore_index=0)
    else:
        loss = nn.CrossEntropyLoss(ignore_index=0)

    l = 0
    for i in range(labels_train.size(0)):
        l += loss(labels_train.view(-1, num_classes).float(), labels_gt.view(-1))
    return l


parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, default='configs/pandaset.yaml', help='Configs path')
args = parser.parse_args()

with open(args.config_path, 'r') as ymlfile:
    cfg = yaml.safe_load(ymlfile)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load configs
lr = cfg['training_params']['lr']
epochs = cfg['training_params']['epochs']
batch_size = cfg['training_params']['batch_size']
voxel_size = cfg['training_params']['voxel_size']
max_num_points = cfg['training_params']['max_num_points']
input_size = cfg['training_params']['input_size']
num_classes = cfg['training_params']['num_classes']
data = cfg['paths']['data_train']

model = Model(input_size, max_num_points)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

dataset_train = PointLoader(data, voxel_size, max_num_points)
trainloader = DataLoader(dataset_train, batch_size, shuffle=False, num_workers=0)

print('Training')
for epoch in tqdm(range(epochs)):
    for i, d in enumerate(trainloader):
        input = d['input_data'].to(device)
        raw_cloud = d['raw_cloud']
        sem2d = d['sem2d']
        sem3d = d['sem3d']
        gt = d['gt']
        att_mask = model(input)

        f = fusion_voxels(raw_cloud, sem2d, sem3d, att_mask)

        #Loss
        train_loss = build_loss(f, gt, num_classes)
        print('Train loss: {}'.format(train_loss))

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

    print('Training, epoch {}, loss {}'.format(epoch, train_loss))
    if epoch % 4 == 0:
        with torch.no_grad():
            for data in valloader:
                input = d['input_data'].to(device)
                raw_cloud = d['raw_cloud']
                sem2d = d['sem2d']
                sem3d = d['sem3d']
                att_mask = model(input_data)

                f = fusion_voxels(raw_cloud, sem2d, sem3d, att_mask)

                #Loss
                val_loss = build_loss(f, gt)
                print('Validation loss: {}'.format(val_loss))

            print('Validation loss in epoch {}: {}'.format(epoch, val_loss))
