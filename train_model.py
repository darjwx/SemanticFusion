from model.adaptive_attention import *
from dataloader.point_loader import PointLoader

import torch
torch.manual_seed(1)
from torch.utils.data import DataLoader
import torch.optim as optim

from utils.build_loss import build_loss
from utils.lovasz_losses import iou

import yaml
import argparse
import os
from tqdm import tqdm
import numpy as np

# TensorBoard
from torch.utils.tensorboard import SummaryWriter

def main():
    train_writer = SummaryWriter('logs/tb/test/train')
    val_writer = SummaryWriter('logs/tb/test/val')

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/pandaset.yaml', help='Configs path')
    args = parser.parse_args()

    with open(args.config_path, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load configs
    name = cfg['training_params']['name']
    lr = cfg['training_params']['lr']
    epochs = cfg['training_params']['epochs']
    batch_size = cfg['training_params']['batch_size']
    voxel_size = cfg['training_params']['voxel_size']
    sparse_shape = cfg['training_params']['sparse_shape']
    max_num_points = cfg['training_params']['max_num_points']
    max_voxels = cfg['training_params']['max_voxels']
    input_size = cfg['training_params']['input_size']
    num_classes = cfg['training_params']['num_classes']
    pc_range = cfg['training_params']['pc_range']
    data_train = cfg['paths']['data_train']
    data_val = cfg['paths']['data_val']
    gt_map = cfg['gt_map']
    sem_map = cfg['sem_map']
    classes = cfg['classes']
    model_path = cfg['paths']['model_path']
    val_rate = cfg['training_params']['val_rate']
    offline_loader = cfg['training_params']['offline_loader']

    model = Model(input_size, max_num_points, max_voxels, sparse_shape, batch_size)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    dataset_train = PointLoader(name, data_train, voxel_size, max_num_points, max_voxels, input_size, num_classes, pc_range, sparse_shape, gt_map, sem_map, offline_loader)
    dataset_val = PointLoader(name, data_val, voxel_size, max_num_points, max_voxels, input_size, num_classes, pc_range, sparse_shape, gt_map, sem_map, offline_loader)

    trainloader = DataLoader(dataset_train, batch_size, shuffle=True, num_workers=4)
    valloader = DataLoader(dataset_val, batch_size, shuffle=False, num_workers=4)

    print('Training')
    last_mAiou = 0
    for epoch in range(epochs):
        print('-------- Epoch {} --------'.format(epoch))
        pbar = tqdm(total=len(trainloader))

        model.train()
        rloss = 0.0
        for i, d in enumerate(trainloader):
            num_voxels = d['voxel_stats']
            input = d['input_data'].to(device)
            gt = d['gt'].to(device)
            coors = d['coors'].to(device)
            raw_cloud = input[:,:,:,:3]
            sem2d = input[:,:,:,3:num_classes+3]
            sem3d = input[:,:,:,num_classes+3:input_size]

            # Ignore preds that will never match gt
            _, labels_gt = torch.max(gt, dim=3)
            _, labels2d = torch.max(sem2d, dim=3)
            _, labels3d = torch.max(sem3d, dim=3)
            idx = torch.logical_or((labels2d == labels_gt), (labels3d == labels_gt))

            # Build gt for binary loss
            bin_gt = -torch.ones(idx.size()).to(device)
            bin_gt[(labels2d == labels_gt)] = 1
            bin_gt[(labels3d == labels_gt)] = 0

            bin_gt = bin_gt[bin_gt != -1]

            att_mask = model(input, coors)

            # Onehot with scores -> onehot with 1s
            aux = torch.ones(sem2d.shape, dtype=torch.float32).to(device)
            sem2d_onehot = torch.where(sem2d != 0, aux, sem2d)
            sem3d_onehot = torch.where(sem3d != 0, aux, sem3d)
            f = fusion_voxels(raw_cloud, sem2d_onehot, sem3d_onehot, att_mask)

            # Loss
            train_loss = build_loss(att_mask, bin_gt, idx)

            train_loss.backward()
            optimizer.step()

            rloss += train_loss.item()

            optimizer.zero_grad()
            pbar.update(1)
        pbar.close()

        print('Training, epoch {}, loss {}'.format(epoch, rloss/len(trainloader)))

        train_writer.add_scalar('loss', rloss/len(trainloader), epoch)
        train_writer.close()

        if epoch % val_rate == 0:
            print('validating epoch {}'.format(epoch))
            model.eval()
            rloss_val = 0.0
            ious = np.zeros((len(valloader), num_classes-1))
            with torch.no_grad():
                for v, data in enumerate(valloader):
                    num_voxels = data['voxel_stats']
                    input = data['input_data'].to(device)
                    gt = data['gt'].to(device)
                    coors = data['coors'].to(device)
                    raw_cloud = input[:,:,:,:3]
                    sem2d = input[:,:,:,3:num_classes+3]
                    sem3d = input[:,:,:,num_classes+3:input_size]
                    att_mask = model(input, coors)

                    # Ignore preds that will never match gt
                    _, labels_gt = torch.max(gt, dim=3)
                    _, labels2d = torch.max(sem2d, dim=3)
                    _, labels3d = torch.max(sem3d, dim=3)
                    idx = torch.logical_or((labels2d == labels_gt), (labels3d == labels_gt))

                    # Build gt for binary loss
                    bin_gt = -torch.ones(idx.size()).to(device)
                    bin_gt[(labels2d == labels_gt)] = 1
                    bin_gt[(labels3d == labels_gt)] = 0

                    bin_gt = bin_gt[bin_gt != -1]

                    # Onehot with scores -> onehot with 1s
                    aux = torch.ones(sem2d.shape, dtype=torch.float32).to(device)
                    sem2d_onehot = torch.where(sem2d != 0, aux, sem2d)
                    sem3d_onehot = torch.where(sem3d != 0, aux, sem3d)
                    f = fusion_voxels(raw_cloud, sem2d_onehot, sem3d_onehot, att_mask)

                    # Loss
                    val_loss = build_loss(att_mask, bin_gt, idx)
                    rloss_val += val_loss.item()

                    # IoU
                    ious[v] = iou(f, gt, num_classes, ignore=0)

                aiou = 0
                for o in range(ious.shape[1]):
                    aiou += np.mean(ious[:,o])
                    print('mIOU - {}: {}'.format(classes[o], np.mean(ious[:,o])))
                print('Average mIOU - {}'.format(aiou/(num_classes-1)))
                print('Validation loss in epoch {}: {}'.format(epoch, rloss_val/len(valloader)))

                mAiou = aiou/(num_classes-1)

                # Save model
                if mAiou > last_mAiou:
                    print(f'Saving model with a IoU of {mAiou}%')
                    torch.save(model.state_dict(), model_path)
                    last_mAiou = mAiou

                val_writer.add_scalar('iou', mAiou, epoch)
                val_writer.add_scalar('loss', rloss_val/len(valloader), epoch)
                val_writer.close()

                rloss_val = 0.0

if __name__ == '__main__':
    main()
