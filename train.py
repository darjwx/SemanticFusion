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

def main():
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
    max_voxels = cfg['training_params']['max_voxels']
    input_size = cfg['training_params']['input_size']
    num_classes = cfg['training_params']['num_classes']
    data_train = cfg['paths']['data_train']
    data_val = cfg['paths']['data_val']

    model = Model(input_size, max_num_points, max_voxels)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    dataset_train = PointLoader(data_train, voxel_size, max_num_points, max_voxels, input_size, num_classes)
    dataset_val = PointLoader(data_val, voxel_size, max_num_points, max_voxels, input_size, num_classes)

    trainloader = DataLoader(dataset_train, batch_size, shuffle=False, num_workers=4)
    valloader = DataLoader(dataset_val, batch_size, shuffle=False, num_workers=4)

    print('Training')
    for epoch in range(epochs):
        print('-------- Epoch {} --------'.format(epoch))
        pbar = tqdm(total=len(trainloader))

        model.train()
        rloss = 0.0
        for i, d in enumerate(trainloader):
            input = d['input_data'].to(device)
            gt = d['gt'].to(device)
            raw_cloud = input[:,:,:,:3]
            sem2d = input[:,:,:,3:num_classes+3]
            sem3d = input[:,:,:,num_classes+2:-1]

            att_mask = model(input)

            f = fusion_voxels(raw_cloud, sem2d, sem3d, att_mask)

            #Loss
            train_loss = build_loss(f, gt, num_classes)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            rloss += train_loss.item()

            if i %50 == 0 and i != 0:
                print('Training, epoch {}, loss {}'.format(epoch, rloss/50))
                rloss = 0.0

            pbar.update(1)
        pbar.close()

        if epoch % 10 == 0:
            print('validating epoch {}'.format(epoch))
            model.eval()
            rloss_val = 0.0
            with torch.no_grad():
                for data in valloader:
                    input = d['input_data'].to(device)
                    gt = d['gt'].to(device)
                    raw_cloud = input[:,:,:,:3]
                    sem2d = input[:,:,:,3:num_classes+3]
                    sem3d = input[:,:,:,num_classes+2:-1]
                    att_mask = model(input)

                    f = fusion_voxels(raw_cloud, sem2d, sem3d, att_mask)

                    #Loss
                    val_loss = build_loss(f, gt, num_classes)
                    rloss_val += val_loss.item()

                print('Validation loss in epoch {}: {}'.format(epoch, rloss_val/len(valloader)))
                rloss_val = 0.0

if __name__ == '__main__':
    main()
