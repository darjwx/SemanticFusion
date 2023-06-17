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


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load configs
    name = cfg['training_params']['name']
    lr = cfg['training_params']['lr']
    epochs = cfg['training_params']['epochs']
    batch_size = cfg['training_params']['batch_size']
    voxel_size = cfg['training_params']['voxel_size']
    max_num_points = cfg['training_params']['max_num_points']
    max_voxels = cfg['training_params']['max_voxels']
    input_size = cfg['training_params']['input_size']
    num_classes = cfg['training_params']['num_classes']
    unified_mask = cfg['training_params']['unified_mask']
    pc_range = cfg['training_params']['pc_range']
    data_train = cfg['paths']['data_train']
    data_val = cfg['paths']['data_val']
    gt_map = cfg['gt_map']
    sem_map = cfg['sem_map']
    class_names = cfg['classes']
    model_path = cfg['paths']['model_path']
    val_rate = cfg['training_params']['val_rate']
    offline_loader = cfg['training_params']['offline_loader']
    train_metrics = cfg['training_params']['train_metrics']

    # 2D score filters
    scores2d_order = cfg['scores2d_filters']['scores2d_order']
    scores2d_ignore = cfg['scores2d_filters']['scores2d_ignore']
    merged_classes = cfg['scores2d_filters']['merged_classes']

    sparse_x = np.floor((pc_range[3] - pc_range[0]) / voxel_size[0]).astype(int)
    sparse_y = np.floor((pc_range[4] - pc_range[1]) / voxel_size[1]).astype(int)
    sparse_z = np.floor((pc_range[5] - pc_range[2]) / voxel_size[2]).astype(int)
    sparse_shape = [sparse_x, sparse_y, sparse_z]

    model = Model(input_size, max_num_points, max_voxels, sparse_shape, unified_mask)
    model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=lr)

    dataset_train = PointLoader(name, data_train, voxel_size, max_num_points, max_voxels, input_size, num_classes, pc_range, gt_map, sem_map, scores2d_order, scores2d_ignore, merged_classes, device, offline_loader)
    dataset_val = PointLoader(name, data_val, voxel_size, max_num_points, max_voxels, input_size, num_classes, pc_range, gt_map, sem_map, scores2d_order, scores2d_ignore, merged_classes, device, offline_loader)

    trainloader = DataLoader(dataset_train, batch_size, shuffle=True, num_workers=0)
    valloader = DataLoader(dataset_val, batch_size, shuffle=False, num_workers=0)

    print('Training')
    last_mAiou = 0
    for epoch in range(epochs):
        print('-------- Epoch {} --------'.format(epoch))
        pbar = tqdm(total=len(trainloader))

        model.train()
        rloss = 0.0

        if train_metrics:
            ious_train = np.zeros((len(trainloader), num_classes-1))

        for i, d in enumerate(trainloader):
            pc_voxel_id = d['pc_voxel_id'].to(device)
            input = d['input_data'].to(device)
            gt = d['gt'].to(device)
            coors = d['coors'].to(device)
            sem2d = d['sem2d'].to(device)
            sem3d = d['sem3d'].to(device)

            # Ignore preds that will never match gt
            _, labels_gt = torch.max(gt, dim=2)
            _, labels2d = torch.max(sem2d, dim=2)
            _, labels3d = torch.max(sem3d, dim=2)

            # Build gt for binary loss
            if unified_mask:
                idx_2d = torch.logical_and((labels2d == labels_gt), (labels3d != labels_gt))
                idx_3d = (labels3d == labels_gt)
                idx = torch.logical_or(idx_2d, idx_3d)
                att_gt = torch.zeros(idx.size(), dtype=torch.float).to(device)

                att_gt[idx_2d] = 1
                att_gt = att_gt.view(-1)
            else:
                idx_2d = (labels2d == labels_gt)
                idx_3d = (labels3d == labels_gt)
                idx_both = torch.logical_and(idx_2d, idx_3d)
                idx_neither = torch.logical_not(torch.logical_or(idx_2d, idx_3d))
                idx = torch.logical_or(idx_2d, idx_3d)
                att_gt = torch.ones(idx.size(), dtype=torch.float).to(device)

                att_gt[idx_2d] = 0
                att_gt[idx_3d] = 1
                att_gt[idx_both] = 2
                att_gt[idx_neither] = 3
                att_gt = att_gt.view(-1)

            att_mask = model(input, coors, pc_voxel_id)

            _, classes2d = torch.max(sem2d, dim=2)
            _, classes3d = torch.max(sem3d, dim=2)

            # Loss
            train_loss = build_loss(att_mask.squeeze(-1), att_gt, unified_mask)
            train_loss.backward()
            optimizer.step()

            if train_metrics:

                if unified_mask:
                    classes = torch.cat((classes3d.permute(1,0), classes2d.permute(1,0)), dim=-1)

                    preds = att_mask.squeeze().clone().long()
                    preds[preds >= 0.5] = 1
                    preds[preds < 0.5] = 0
                else:
                    classes = torch.cat((classes2d.permute(1,0), classes3d.permute(1,0)), dim=-1)

                    _, preds = torch.max(att_mask, dim=-1)
                    preds = torch.clip(preds, min=0, max=1)

                labels = classes[torch.arange(classes.size(0)), preds]
                _, labels_gt = torch.max(gt.squeeze(), dim=1)

                ious_train[i] = iou(labels, labels_gt, num_classes, ignore=0, adapt_arrays=False)

            rloss += train_loss.item()

            optimizer.zero_grad()
            pbar.update(1)
        pbar.close()

        if train_metrics:
            aiou = 0
            for o in range(ious_train.shape[1]):
                aiou += np.mean(ious_train[:,o])
            mAiou = aiou/(num_classes-1)

            print('Training, epoch {}, loss {}'.format(epoch, rloss/len(trainloader)))

            train_writer.add_scalar('iou', mAiou, epoch)
            train_writer.add_scalar('loss/total', rloss/len(trainloader), epoch)
            train_writer.close()

        if epoch % val_rate == 0:
            print('validating epoch {}'.format(epoch))
            model.eval()
            rloss_val = 0.0

            ious = np.zeros((len(valloader), num_classes-1))
            pbar = tqdm(total=len(valloader))
            with torch.no_grad():
                for v, data in enumerate(valloader):
                    pc_voxel_id = data['pc_voxel_id'].to(device)
                    input = data['input_data'].to(device)
                    gt = data['gt'].to(device)
                    coors = data['coors'].to(device)
                    sem2d = data['sem2d'].to(device)
                    sem3d = data['sem3d'].to(device)

                    # Ignore preds that will never match gt
                    _, labels_gt = torch.max(gt, dim=2)
                    _, labels2d = torch.max(sem2d, dim=2)
                    _, labels3d = torch.max(sem3d, dim=2)

                    if unified_mask:
                        idx_2d = torch.logical_and((labels2d == labels_gt), (labels3d != labels_gt))
                        idx_3d = (labels3d == labels_gt)
                        idx = torch.logical_or(idx_2d, idx_3d)
                        att_gt = torch.zeros(idx.size(), dtype=torch.float).to(device)

                        att_gt[idx_2d] = 1
                        att_gt = att_gt.view(-1)
                    else:
                        idx_2d = (labels2d == labels_gt)
                        idx_3d = (labels3d == labels_gt)
                        idx_both = torch.logical_and(idx_2d, idx_3d)
                        idx_neither = torch.logical_not(torch.logical_or(idx_2d, idx_3d))
                        idx = torch.logical_or(idx_2d, idx_3d)
                        att_gt = torch.ones(idx.size(), dtype=torch.float).to(device)

                        att_gt[idx_2d] = 0
                        att_gt[idx_3d] = 1
                        att_gt[idx_both] = 2
                        att_gt[idx_neither] = 3
                        att_gt = att_gt.view(-1)

                    att_mask = model(input, coors, pc_voxel_id)

                    _, classes2d = torch.max(sem2d, dim=2)
                    _, classes3d = torch.max(sem3d, dim=2)

                    # Loss
                    val_loss = build_loss(att_mask.squeeze(-1), att_gt, unified_mask)
                    rloss_val += val_loss.item()

                    if unified_mask:
                        classes = torch.cat((classes3d.permute(1,0), classes2d.permute(1,0)), dim=-1)

                        preds = att_mask.squeeze().clone().long()
                        preds[preds >= 0.5] = 1
                        preds[preds < 0.5] = 0
                    else:
                        classes = torch.cat((classes2d.permute(1,0), classes3d.permute(1,0)), dim=-1)

                        _, preds = torch.max(att_mask, dim=-1)
                        preds = torch.clip(preds, min=0, max=1)

                    labels = classes[torch.arange(classes.size(0)), preds]
                    _, labels_gt = torch.max(gt.squeeze(), dim=1)

                    ious[v] = iou(labels, labels_gt, num_classes, ignore=0, adapt_arrays=False)
                    pbar.update(1)
                pbar.close()

                aiou = 0
                for o in range(ious.shape[1]):
                    aiou += np.mean(ious[:,o])
                    print('mIOU - {}: {}'.format(class_names[o], np.mean(ious[:,o])))
                print('Average mIOU - {}'.format(aiou/(num_classes-1)))
                print('Validation loss in epoch {}: {}'.format(epoch, rloss_val/len(valloader)))

                mAiou = aiou/(num_classes-1)

                # Save model
                if mAiou > last_mAiou:
                    print(f'Saving model with a IoU of {mAiou}%')
                    torch.save(model.state_dict(), model_path)
                    last_mAiou = mAiou

                val_writer.add_scalar('iou', mAiou, epoch)
                val_writer.add_scalar('loss/total', rloss_val/len(valloader), epoch)
                val_writer.close()

if __name__ == '__main__':
    main()
