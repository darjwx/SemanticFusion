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
from utils.lovasz_losses import iou

import math
import yaml
import argparse
import os
from tqdm import tqdm
import numpy as np
import pickle
import gzip

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/pandaset.yaml', help='Configs path')
    parser.add_argument('--model', type=str, default='model-save.pt', help='Saved model')
    parser.add_argument('--out_path', type=str, default='out/pandaset/labels', help='Final predictions')
    args = parser.parse_args()

    with open(args.config_path, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load configs
    name = cfg['training_params']['name']
    batch_size = 1 # One file per frame
    voxel_size = cfg['training_params']['voxel_size']
    max_num_points = cfg['training_params']['max_num_points']
    max_voxels = cfg['training_params']['max_voxels']
    input_size = cfg['training_params']['input_size']
    num_classes = cfg['training_params']['num_classes']
    unified_mask = cfg['training_params']['unified_mask']
    pc_range = cfg['training_params']['pc_range']
    data_val = cfg['paths']['data_val']
    gt_map = cfg['gt_map']
    sem_map = cfg['sem_map']
    class_names = cfg['classes']
    offline_loader = cfg['training_params']['offline_loader']

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

    # Load model
    model.load_state_dict(torch.load(args.model))

    dataset_val = PointLoader(name, data_val, voxel_size, max_num_points, max_voxels, input_size, num_classes, pc_range, gt_map, sem_map, scores2d_order, scores2d_ignore, merged_classes, device, offline_loader)
    valloader = DataLoader(dataset_val, batch_size, shuffle=False, num_workers=0)

    digits = int(math.log10(len(valloader)))+1
    print('Validating model')
    model.eval()
    pbar = tqdm(total=len(valloader))
    ious = np.zeros((len(valloader), num_classes-1))
    ious_c3d = np.zeros((len(valloader), num_classes-1))
    with torch.no_grad():
        for d, data in enumerate(valloader):
            pc_voxel_id = data['pc_voxel_id'].to(device)
            input = data['input_data'].to(device)
            gt = data['gt'].to(device)
            coors = data['coors'].to(device)
            raw_cloud = data['raw_cloud'].to(device)
            sem2d = data['sem2d'].to(device)
            sem3d = data['sem3d'].to(device)
            att_mask = model(input, coors, pc_voxel_id)

            misc = data['misc']
            seq = misc['seq'][0]
            frame = misc['frame']['cloud'][0]
            if name == 'pandaset':
                frame = frame.zfill(2)
            elif name == 'kitti':
                frame = frame[0].zfill(6)

            _, labels_gt = torch.max(gt, dim=2)
            _, classes2d = torch.max(sem2d, dim=2)
            _, classes3d = torch.max(sem3d, dim=2)

            if unified_mask:
                classes = torch.cat((classes3d.permute(1,0), classes2d.permute(1,0)), dim=-1)

                preds = att_mask.squeeze().clone()
                preds[preds >= 0.5] = 1
                preds[preds < 0.5] = 0
                preds = preds.long()
            else:
                classes = torch.cat((classes2d.permute(1,0), classes3d.permute(1,0)), dim=-1)

                _, preds = torch.max(att_mask, dim=-1)
                preds = torch.clip(preds, min=0, max=1)

            labels = classes[torch.arange(classes.size(0)), preds]
            _, labels_gt = torch.max(gt.squeeze(), dim=1)

            # IoU
            ious[d] = iou(labels, labels_gt, num_classes, ignore=0, adapt_arrays=False)

            # Source 3D IoU
            ious_c3d[d] = iou(classes3d, labels_gt, num_classes, ignore=0, adapt_arrays=False)

            labels = labels.cpu().numpy()
            labels_gt = labels_gt.cpu().numpy()

            val_results = {}
            val_results['labels'] = labels
            val_results['points'] = raw_cloud.squeeze().cpu().numpy()
            val_results['gt'] = labels_gt
            val_results['att_mask'] = att_mask.cpu().numpy()
            val_results['frame'] = frame
            val_results['seq'] = seq[0]
            val_results['c3d'] = classes3d.squeeze().cpu().numpy()

            file_path = os.path.join(args.out_path, '{}.gz'.format(str(d).zfill(digits)))
            with gzip.open(file_path, 'wb') as f:
                pickle.dump(val_results, f)

            pbar.update(1)
        pbar.close()

        aiou = 0
        aiou_c3d = 0
        print('mIOU - 3D baseline -- Fusion results')
        for o in range(ious.shape[1]):
            aiou += np.mean(ious[:,o])
            aiou_c3d += np.mean(ious_c3d[:,o])
            print('mIOU - {}: {} -> {}'.format(class_names[o], np.mean(ious_c3d[:,o]), np.mean(ious[:,o]), 0))
        print('Average mIOU {} -> {}'.format(aiou_c3d/(num_classes-1), aiou/(num_classes-1), 0))

if __name__ == '__main__':
    main()
