import argparse
import pickle
import yaml
import cv2 as cv
import numpy as np
from tqdm import tqdm
from utils import pandaset_util as ps_util
from pathlib import Path
import gzip
import utils.color_utils as color_utils

def main(args):
    # Load color map
    color_map = color_utils.Colours().get_color_map('labels')

    with open(args.config_path, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    classes = cfg['classes']
    val = cfg['sequences']['val']
    cams = ['front_camera', 'front_left_camera', 'front_right_camera', 'back_camera', 'left_camera', 'right_camera']
    img_dir = args.data
    out_path = args.video
    print('\n' + str(out_path))

    # Video variables
    fourcc = cv.VideoWriter_fourcc(*'MPEG')
    out = cv.VideoWriter(out_path, fourcc, 10, (1920,1080))

    labels_path = Path(args.labels_path)

    p_idx = ps_util.PandaIdx()
    p_idx.add_split(val)
    m_height = 70
    m_width =  50
    cont = 0
    for i in tqdm(sorted(labels_path.iterdir())):
        seq_idx, f = p_idx.get_idx()
        labels_file = args.labels_path + str(cont) + '.gz'
        with gzip.open(labels_file, 'rb') as file:
            data = pickle.load(file)

        cloud_lidar = data['points']
        labels = data['labels']

        dict_img = {}
        for c in cams:
            dir = img_dir + '/' + str(seq_idx) + '/camera/' + c + '/' + f + '.jpg'
            img = cv.imread(dir)

            # Calibration
            calib_info = {'sequence': seq_idx,
                          'camera': c,
                          'idx': int(f)}
            calib = ps_util.PandasetCalibration(args.data, calib_info)

            if c == 'front_camera':
                cloud_ego = calib.project_lidar_to_ego(cloud_lidar)
                cloud_lidar = ps_util.rotate_velo(cloud_lidar, np.pi/2)
                bev, points_id = ps_util.birds_eye_point_cloud(cloud_lidar,side_range=(-m_width, m_width),fwd_range=(-m_height, m_height), res=0.1)
                bev = cv.bitwise_not(bev)
                # TODO: bev still thinks its BGR
                bev = cv.cvtColor(bev, cv.COLOR_GRAY2RGB)
                ind = points_id.nonzero()
                bev[ind[0],ind[1]] = color_map[labels[points_id[ind[0],ind[1]]].astype(int)]

            points, points3d, fov_flag = ps_util.projection(cloud_ego, calib)
            fov_labels = labels[fov_flag]

            for p in range(points.shape[0]):
                clr = color_map[int(fov_labels[p])]
                color = (int(clr[2]), int(clr[1]), int(clr[0]))
                img = cv.circle(img, (int(points[p,0]), int(points[p,1])), 2, color, -1)

            # stack images
            dict_img[c] = img

        # Stack images
        aux1 = np.hstack((dict_img['front_left_camera'], dict_img['front_camera']))
        front_side = np.hstack((aux1, dict_img['front_right_camera']))
        aux1 = np.hstack((dict_img['left_camera'], dict_img['back_camera']))
        back_side = np.hstack((aux1, dict_img['right_camera']))
        final_img = np.vstack((front_side, back_side))

        # Draw legend in bev
        bev = cv.resize(bev, (1400, 2160))
        for cl in range(0, np.shape(classes)[0]):
            clr = color_map[cl+1]
            clr = (int(clr[2]), int(clr[1]), int(clr[0]))
            cv.rectangle(bev, (50, 50+60*cl), (100, 100+60*cl), clr,-1)
            cv.putText(bev, classes[cl], (110, 75+60*cl), cv.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv.LINE_AA)

        final_img = np.hstack((final_img, bev))
        final_img = cv.resize(final_img, (1920,1080))
        out.write(final_img)

        p_idx.update()
        cont += 1

    out.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--labels_path', default='out/pandaset/labels/test/', type=str, help='Labels path')
    parser.add_argument('--video', default='fusion_painting_pandaset.avi', type=str, help='Video name and path')
    parser.add_argument('--data', default='datasets/pandaset/data/data', type=str, help='Dataset path')
    parser.add_argument('--config_path', type=str, default='configs/pandaset.yaml', help='Configs path')

    args = parser.parse_args()

    main(args)
