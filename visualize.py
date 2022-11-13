import argparse
import os
import pickle
import yaml
import cv2 as cv
import numpy as np
from tqdm import tqdm
from pathlib import Path
import gzip
import utils.video_utils as video_utils

def main(args):
    # Load color map
    color_map = video_utils.Colours().get_color_map('labels')

    with open(args.config_path, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    classes = cfg['classes']
    cams = cfg['cams']
    dataset = cfg['training_params']['name']
    pc_range = cfg['training_params']['pc_range']
    img_dir = args.data
    out_path = args.video
    print('\n' + str(out_path))

    # Video variables
    fourcc = cv.VideoWriter_fourcc(*'MPEG')
    out = cv.VideoWriter(out_path, fourcc, 10, (1920,1080))

    labels_path = Path(args.labels_path)

    for file in tqdm(sorted(os.listdir(labels_path))):
        labels_file = os.path.join(args.labels_path, file)
        with gzip.open(labels_file, 'rb') as file:
            data = pickle.load(file)

        seq_idx = data['seq']
        f = data['frame']
        cloud_lidar = data['points']
        labels = data['labels']

        dict_img = {}
        for i, c in enumerate(cams):
            if dataset == 'pandaset':
                dir = img_dir + '/' + str(seq_idx) + '/camera/' + c + '/' + f + '.jpg'
                img = cv.imread(dir)
            else:
                img = np.zeros((720, 1280, 3), np.uint8)

            if dataset == 'pandaset':
                from utils import pandaset_util as ps_util
                # Calibration
                calib_info = {'sequence': seq_idx,
                            'camera': c,
                            'idx': int(f)}
                calib = ps_util.PandasetCalibration(args.data, calib_info)

                # Revert back to Pandaset's original rotation.
                cloud_norot = ps_util.rotate_velo(cloud_lidar.copy(), -np.pi/2)
                # Project points to camera and image frames.
                cloud_cam = calib.project_lidar_to_camera(cloud_norot.copy())
                points = calib.project_lidar_to_image(cloud_norot.copy())
                points, fov_flag = video_utils.pc_in_image_fov(points, cloud_cam, img.shape)

            elif dataset == 'carla':
                from utils.carla_utils import CarlaCalibration
                calib = CarlaCalibration(os.path.join(cfg['paths']['source'], 'calibs'), 'cam0') #TODO do not harcode camera

                cam_points = calib.project_lidar_to_rect(cloud_lidar)
                points = calib.project_lidar_to_image(cloud_lidar)
                points, fov_flag = video_utils.pc_in_image_fov(points, cam_points, img.shape)

            if i == 0:
                bev, points_id = video_utils.birds_eye_point_cloud(cloud_lidar,side_range=(pc_range[2], pc_range[3]),fwd_range=(pc_range[0], pc_range[1]), res=0.1, min_height=pc_range[4], max_height=pc_range[5])
                bev = cv.bitwise_not(bev)
                # TODO: bev still thinks its BGR
                bev = cv.cvtColor(bev, cv.COLOR_GRAY2RGB)
                ind = points_id.nonzero()
                bev[ind[0],ind[1]] = color_map[labels[points_id[ind[0],ind[1]]].astype(int)]
                bev = bev[:,:, ::-1] # TODO: bev still thinks its BGR. Manually change to RGB

            fov_labels = labels[fov_flag]

            for p in range(points.shape[0]):
                clr = color_map[int(fov_labels[p])]
                color = (int(clr[2]), int(clr[1]), int(clr[0]))
                img = cv.circle(img, (int(points[p,0]), int(points[p,1])), 2, color, -1)

            # stack images
            dict_img[c] = img

        # Stack images
        if dataset == 'pandaset':
            aux1 = np.hstack((dict_img['front_left_camera'], dict_img['front_camera']))
            front_side = np.hstack((aux1, dict_img['front_right_camera']))
            aux1 = np.hstack((dict_img['left_camera'], dict_img['back_camera']))
            back_side = np.hstack((aux1, dict_img['right_camera']))
            final_img = np.vstack((front_side, back_side))
        else:
            final_img = img.copy()

        # Draw legend in bev
        bev = cv.resize(bev, (pc_range[3]*20, final_img.shape[0]))

        for cl in range(0, np.shape(classes)[0]):
            clr = color_map[cl+1]
            clr = (int(clr[2]), int(clr[1]), int(clr[0]))
            cv.rectangle(bev, (50, 50+60*cl), (100, 100+60*cl), clr,-1)
            cv.putText(bev, classes[cl], (110, 75+60*cl), cv.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv.LINE_AA)

        final_img = np.hstack((final_img, bev))
        final_img = cv.resize(final_img, (1920,1080))
        out.write(final_img)

    out.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--labels_path', default='out/pandaset/labels/test/', type=str, help='Labels path')
    parser.add_argument('--video', default='fusion_painting_pandaset.avi', type=str, help='Video name and path')
    parser.add_argument('--data', default='datasets/pandaset/data/data', type=str, help='Dataset path')
    parser.add_argument('--config_path', type=str, default='configs/pandaset.yaml', help='Configs path')

    args = parser.parse_args()

    main(args)
