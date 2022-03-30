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
    color_map = color_utils.Colours().get_color_map('att_mask')

    with open(args.config_path, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    val = cfg['sequences']['val']
    out_path = args.video
    print('\n' + str(out_path))

    # Video variables
    fourcc = cv.VideoWriter_fourcc(*'MPEG')
    out = cv.VideoWriter(out_path, fourcc, 10, (1400,2800))

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
        att_mask = data['att_mask']

        cloud_lidar = ps_util.rotate_velo(cloud_lidar, np.pi/2)
        bev, points_id = ps_util.birds_eye_point_cloud(cloud_lidar,side_range=(-m_width, m_width),fwd_range=(-m_height, m_height), res=0.1)
        bev = cv.bitwise_not(bev)
        # TODO: bev still thinks its BGR
        bev = cv.cvtColor(bev, cv.COLOR_GRAY2RGB)
        ind = points_id.nonzero()

        ns = np.floor(att_mask*39) # Transform from scores to a color level
        bev[ind[0],ind[1]] = color_map[ns[points_id[ind[0],ind[1]]].astype(int)]
        bev = bev[:,:,::-1] # TODO: bev still thinks its BGR. Manually change to RGB

        bev = cv.resize(bev, (1400, 2800))
        out.write(bev)

        p_idx.update()
        cont += 1

    out.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--labels_path', default='out/pandaset/labels/test/', type=str, help='Labels path')
    parser.add_argument('--video', default='att_mask_cloud.avi', type=str, help='Video name and path')
    parser.add_argument('--data', default='datasets/pandaset/data/data', type=str, help='Dataset path')
    parser.add_argument('--config_path', type=str, default='configs/pandaset.yaml', help='Configs path')

    args = parser.parse_args()

    main(args)
