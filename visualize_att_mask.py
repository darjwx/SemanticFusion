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

class PandaIdx():
    def __init__(self):
        self.seq = 1
        self.idx = 0
        self.split = None

    def add_split(self, split):
        self.split = split
        self.seq = split[0]
        self.index = 0

    def update(self):
        if self.split == None:
            self.idx += 1

            if self.idx == 80:
                self.idx = 0
                self.seq += 1
        else:
            self.idx += 1

            if self.idx == 80:
                self.idx = 0
                self.index += 1
                # Check index overflow
                try:
                    self.seq = self.split[self.index]
                except IndexError:
                    print('End of split')
                    self.index = 0
                    self.seq = self.split[0]

    def get_idx(self):
        if self.split == None:
            return str(self.seq).zfill(3), str(self.idx).zfill(2)
        else:
            return self.seq, str(self.idx).zfill(2)

def birds_eye_point_cloud(points,
                          side_range=(-10, 10),
                          fwd_range=(-10,10),
                          res=0.1,
                          min_height = -2.73,
                          max_height = 1.27,
                          saveto=None):
    """
    Creates an 2D birds eye view representation of the point cloud data.
        You can optionally save the image to specified filename.
    Args:
        points:     (numpy array)
                    N rows of points data
                    Each point should be specified by at least 3 elements x,y,z
        side_range: (tuple of two floats)
                    (-left, right) in metres
                    left and right limits of rectangle to look at.
        fwd_range:  (tuple of two floats)
                    (-behind, front) in metres
                    back and front limits of rectangle to look at.
        res:        (float) desired resolution in metres to use
                    Each output pixel will represent an square region res x res
                    in size.
        min_height:  (float)(default=-2.73)
                    Used to truncate height values to this minumum height
                    relative to the sensor (in metres).
                    The default is set to -2.73, which is 1 metre below a flat
                    road surface given the configuration in the kitti dataset.
        max_height: (float)(default=1.27)
                    Used to truncate height values to this maximum height
                    relative to the sensor (in metres).
                    The default is set to 1.27, which is 3m above a flat road
                    surface given the configuration in the kitti dataset.
        saveto:     (str or None)(default=None)
                    Filename to save the image as.
                    If None, then it just displays the image.
    """
    x_lidar = points[:, 0]
    y_lidar = points[:, 1]
    z_lidar = points[:, 2]

    # INDICES FILTER - of values within the desired rectangle
    # Note left side is positive y axis in LIDAR coordinates
    ff = np.logical_and((x_lidar > fwd_range[0]), (x_lidar < fwd_range[1]))
    ss = np.logical_and((y_lidar > -side_range[1]), (y_lidar < -side_range[0]))
    indices = np.argwhere(np.logical_and(ff,ss)).flatten()

    # CONVERT TO PIXEL POSITION VALUES - Based on resolution
    x_img = (-y_lidar[indices]/res).astype(np.int32) # x axis is -y in LIDAR
    y_img = (x_lidar[indices]/res).astype(np.int32)  # y axis is -x in LIDAR
                                                     # will be inverted later

    # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
    # floor used to prevent issues with -ve vals rounding upwards
    x_img -= int(np.floor(side_range[0]/res))
    y_img -= int(np.floor(fwd_range[0]/res))

    # CLIP HEIGHT VALUES - to between min and max heights
    pixel_values = np.clip(a = z_lidar[indices],
                           a_min=min_height,
                           a_max=max_height)

    # RESCALE THE HEIGHT VALUES - to be between the range 0-255
    pixel_values  = 255.

    # FILL PIXEL VALUES IN IMAGE ARRAY
    x_max = int((side_range[1] - side_range[0])/res)
    y_max = int((fwd_range[1] - fwd_range[0])/res)
    im = np.zeros([y_max, x_max], dtype=np.uint8)
    im[-y_img, x_img] = pixel_values # -y because images start from top left

    # Save on points_id the points already converted to image.
    points_id = np.zeros([y_max, x_max], dtype=np.uint32)
    points_id[-y_img, x_img] = indices

    return im, points_id

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

    p_idx = PandaIdx()
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
        bev, points_id = birds_eye_point_cloud(cloud_lidar,side_range=(-m_width, m_width),fwd_range=(-m_height, m_height), res=0.1)
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
