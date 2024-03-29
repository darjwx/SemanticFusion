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

import numpy as np
import os
import pandas as pd
from PIL import Image
from tqdm import tqdm

def pc_in_image_fov(img_points, cam_points, dims):
    fov_idx = np.ones(img_points.shape[0], dtype=bool)

    # Discard points with negative z (points from the opposite cam)
    # in camera coordinates.
    fov_idx = np.logical_and(fov_idx, cam_points[:,2] > 0)

    # Discard points outside of image xy range.
    fov_idx = np.logical_and(fov_idx, img_points[:, 0] > 0)
    fov_idx = np.logical_and(fov_idx, img_points[:, 0] < dims[1])
    fov_idx = np.logical_and(fov_idx, img_points[:, 1] > 0)
    fov_idx = np.logical_and(fov_idx, img_points[:, 1] < dims[0])
    img_points = img_points[fov_idx]

    return img_points, fov_idx

def project_points(infos, color_map, dataset):

    # Transform color value arrays to strings and use them as keys
    # in the dataset specific color_map
    def get_value_with_key(color):
        str_code = '[{},{},{}]'.format(color[0], color[1], color[2])

        value = color_map.get(str_code, 0)
        return value

    for i in tqdm(range(len(infos))):
        info = infos[i]
        calib_info = info['calib']

        # Load LiDAR cloud
        if dataset == 'pandaset':
            pc = pd.read_pickle(info['cloud'])
            pc = pc[pc.d == 0].to_numpy()[:, :3]
        elif dataset == 'carla':
            pc = np.fromfile(info['cloud'], dtype=np.float32).reshape(-1, 4)[:, :3]
        elif dataset == 'kitti':
            pc = np.fromfile(info['cloud'], dtype=np.float32).reshape(-1, 4)[:, :3]

        color_labels = np.zeros((pc.shape[0], 3), dtype=np.uint8)
        id_labels = np.zeros((pc.shape[0]), dtype=np.uint8)
        score_labels = np.zeros((pc.shape[0], 19), dtype=np.float32)
        # Get semantic images from each camera
        for id, im in enumerate(info['sem_image']):
            # 1. Project pc into image
            if dataset == 'pandaset':
                from .pandaset_util import PandasetCalibration
                calib = PandasetCalibration('datasets/pandaset/data/data', calib_info[id])

                pc_lidar = calib.project_ego_to_lidar(pc)
                pc_cam = calib.project_lidar_to_camera(pc_lidar)
                pc_img = calib.project_lidar_to_image(pc_lidar)
            elif dataset == 'carla':
                from .carla_utils import CarlaCalibration
                calib = CarlaCalibration(info['calib'][id], 'cam0') #TODO do not harcode camera

                pc_cam = calib.project_lidar_to_rect(pc)
                pc_img = calib.project_lidar_to_image(pc)
            elif dataset == 'kitti':
                from .kitti_utils import KittiCalibration
                calib = KittiCalibration(os.path.join('datasets/kitti/odometry/dataset/sequences/', calib_info[id]['sequence'], 'calib.txt'))

                pc_cam = calib.project_lidar_to_camera(pc)
                pc_img = calib.project_lidar_to_image(pc)


            # Extra A channel contains scores
            img = np.array(Image.open(im).convert('RGB'))
            scores = np.fromfile(info['sem2d_scores'], dtype=np.float32).reshape(img.shape[0], img.shape[1], 19)

            # 2. Filter cloud with image boundaries
            pc_fov, fov_idx = pc_in_image_fov(pc_img, pc_cam, img.shape)
            # 3. Get colors for each cloud point
            color_labels[fov_idx] = img[pc_fov[:,1].astype(int), pc_fov[:,0].astype(int)]
            score_labels[fov_idx] = scores[pc_fov[:,1].astype(int), pc_fov[:,0].astype(int)] / 100

        # Transform color values into class ids and save them in bin files for each seq_frame
        id_labels = np.vectorize(get_value_with_key, signature='(n)->()')(color_labels)
        id_labels = np.expand_dims(id_labels, axis=1)
        sem2d = np.concatenate((id_labels, score_labels), axis=1)
        sem2d.astype(np.float32).tofile(info['sem2d'])
