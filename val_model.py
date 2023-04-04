from model.adaptive_attention import *
from dataloader.point_loader import PointLoader

import torch
torch.manual_seed(1)
from torch.utils.data import DataLoader
import torch.optim as optim
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
    sparse_shape = cfg['training_params']['sparse_shape']
    max_num_points = cfg['training_params']['max_num_points']
    max_voxels = cfg['training_params']['max_voxels']
    input_size = cfg['training_params']['input_size']
    num_classes = cfg['training_params']['num_classes']
    pc_range = cfg['training_params']['pc_range']
    data_val = cfg['paths']['data_val']
    gt_map = cfg['gt_map']
    sem_map = cfg['sem_map']
    classes = cfg['classes']
    offline_loader = cfg['training_params']['offline_loader']

    model = Model(input_size, max_num_points, max_voxels, sparse_shape, batch_size)
    model = model.to(device)

    # Load model
    model.load_state_dict(torch.load(args.model))

    dataset_val = PointLoader(name, data_val, voxel_size, max_num_points, max_voxels, input_size, num_classes, pc_range, sparse_shape, gt_map, sem_map, offline_loader)
    valloader = DataLoader(dataset_val, batch_size, shuffle=False, num_workers=4)
    digits = int(math.log10(len(valloader)))+1

    print('Validating model')
    model.eval()
    pbar = tqdm(total=len(valloader))
    ious = np.zeros((len(valloader), num_classes-1))
    ious_c3d = np.zeros((len(valloader), num_classes-1))
    with torch.no_grad():
        for d, data in enumerate(valloader):
            num_voxels = data['voxel_stats']
            input = data['input_data'].to(device)
            gt = data['gt'].to(device)
            coors = data['coors'].to(device)
            raw_cloud = input[:,:,:,:3]
            sem2d = input[:,:,:,3:num_classes+3]
            sem3d = input[:,:,:,num_classes+3:input_size]
            att_mask = model(input, coors)

            misc = data['misc']
            seq = misc['seq'][0]
            frame = misc['frame']['cloud'][0]
            if name == 'pandaset':
                frame = frame.zfill(2)
            elif name == 'kitti':
                frame = frame.zfill(6)

            f = fusion_voxels(raw_cloud, sem2d, sem3d, att_mask)

            # IoU
            ious[d] = iou(f, gt, num_classes, ignore=0)

            # Save labels
            f = f.view(-1, f.size(-1))
            labels = f[:,3:input_size]
            _, labels = torch.max(labels, dim=1)
            gt = gt.view(-1, gt.size(-1))
            _, labels_gt = torch.max(gt, dim=1)

            labels = labels.cpu().numpy()
            points = f[:,:3].cpu().numpy()
            labels_gt = labels_gt.cpu().numpy()

            #C3D IoU
            c3d = data['c3d'].view(-1).cpu().numpy().astype(np.uint8)
            ious_c3d[d] = iou(c3d, labels_gt, num_classes, ignore=0, adapt_arrays=False)

            val_results = {}
            val_results['labels'] = labels
            val_results['points'] = points
            val_results['gt'] = labels_gt
            val_results['att_mask'] = att_mask.view(-1).cpu().numpy()
            val_results['frame'] = frame
            val_results['seq'] = seq
            val_results['c3d'] = c3d


            file_path = os.path.join(args.out_path, '{}.gz'.format(str(d).zfill(digits)))
            with gzip.open(file_path, 'wb') as f:
                pickle.dump(val_results, f)

            pbar.update(1)
        pbar.close()

        aiou = 0
        aiou_c3d = 0
        for o in range(ious.shape[1]):
            aiou += np.mean(ious[:,o])
            aiou_c3d += np.mean(ious_c3d[:,o])
            print('mIOU - {}: {} -> {}'.format(classes[o], np.mean(ious_c3d[:,o]), np.mean(ious[:,o])))
        print('Average mIOU {} -> {}'.format(aiou_c3d/(num_classes-1), aiou/(num_classes-1)))

if __name__ == '__main__':
    main()
