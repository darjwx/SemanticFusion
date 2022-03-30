""" Helper methods for loading and parsing KITTI data.

Author: Charles R. Qi
Date: September 2017
"""
from __future__ import print_function

import numpy as np
import os
import json, yaml
import transforms3d as t3d

from .calibration import *

class PandasetCalibration(Calibration):

    def __init__(self, root_dir, calib_info, from_video=False):
        with open(os.path.join(root_dir, 'extrinsic_calibration.yaml')) as f:
            self.extrinsics = yaml.safe_load(f)

        calibs = self.read_calib_files(root_dir, calib_info)

    def _heading_position_to_mat(self, heading, position):
        quat = np.array([heading["w"], heading["x"], heading["y"], heading["z"]])
        pos = np.array([position["x"], position["y"], position["z"]])
        transform_matrix = t3d.affines.compose(np.array(pos),
                                               t3d.quaternions.quat2mat(quat),
                                               [1.0, 1.0, 1.0])
        return transform_matrix

    def read_calib_files(self, root_dir, calib_info):
        ''' Read in a calibration file and parse into a dictionary.
        Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        '''
        camera_intrinsics = os.path.join(root_dir, calib_info['sequence'], 'camera', calib_info['camera'], 'intrinsics.json')
        camera_poses = os.path.join(root_dir, calib_info['sequence'], 'camera', calib_info['camera'], 'poses.json')

        ego_poses = os.path.join(root_dir, calib_info['sequence'], 'lidar', 'poses.json')
        gps_poses = os.path.join(root_dir, calib_info['sequence'], 'meta', 'gps.json')

        if os.path.isfile(camera_intrinsics):
            with open(camera_intrinsics, 'r') as f:
                file_data = json.load(f)
                self.f_u = file_data['fx']
                self.f_v = file_data['fy']
                self.c_u = file_data['cx']
                self.c_v = file_data['cy']

        if os.path.isfile(camera_poses):
            with open(camera_poses, 'r') as f:
                file_data = json.load(f)
                self.camera_pose = file_data[calib_info['idx']]

        if os.path.isfile(ego_poses):
            with open(ego_poses, 'r') as f:
                file_data = json.load(f)
                self.ego_pose = file_data[calib_info['idx']]

        if os.path.isfile(gps_poses):
            with open(gps_poses, 'r') as f:
                file_data = json.load(f)
                self.gps_pose = file_data[calib_info['idx']]

        lidar_pose_mat  = self._heading_position_to_mat(self.extrinsics[calib_info['camera']]['extrinsic']['transform']['rotation'],
            self.extrinsics[calib_info['camera']]['extrinsic']['transform']['translation'])
        camera_pose_mat = self._heading_position_to_mat(self.camera_pose['heading'], self.camera_pose['position'])

        self.C2E = camera_pose_mat
        self.E2C = np.linalg.inv(camera_pose_mat)

        self.L2C = lidar_pose_mat
        self.C2L = np.linalg.inv(lidar_pose_mat)

        self.E2L = np.matmul(self.C2L, self.E2C)
        self.L2E = np.linalg.inv(self.E2L)

        self.K = np.eye(3, dtype=np.float64)
        self.K[0, 0] = self.f_u
        self.K[1, 1] = self.f_v
        self.K[0, 2] = self.c_u
        self.K[1, 2] = self.c_v

        self.P = np.zeros((3,4), dtype=np.float64)
        self.P[0:3, 0:3] = self.K
        self.P[2,3] = 1
        self.R0 = np.eye(3, dtype=np.float64)
    # ===========================
    # ------- 3d to 3d ----------
    # ===========================
    def project_lidar_to_ego(self, pts_3d_lidar):
        points3d_ego = np.matmul(self.L2E[:3, :3], (pts_3d_lidar.T)) + \
                            self.L2E[:3, 3].reshape(3, 1)
        return np.transpose(points3d_ego)

    def project_ego_to_lidar(self, pts_3d_ego):
        points3d_lidar = np.matmul(self.E2L[:3, :3], (pts_3d_ego.T)) + \
                            self.E2L[:3, 3].reshape(3, 1)
        return np.transpose(points3d_lidar)

    def project_ego_to_camera(self, pts_3d_ego):
        points3d_camera = np.matmul(self.E2C[:3, :3], (pts_3d_ego.T)) + \
                            self.E2C[:3, 3].reshape(3, 1)
        return np.transpose(points3d_camera)


    def project_lidar_to_camera(self, pts_3d_lidar):
        points3d_camera = np.matmul(self.L2C[:3, :3], (pts_3d_lidar.T)) + \
                            self.L2C[:3, 3].reshape(3, 1)
        return np.transpose(points3d_camera)

    def project_camera_to_lidar(self, pts_3d_camera):
        points3d_lidar = np.matmul(self.C2L[:3, :3], (pts_3d_camera.T)) + \
                            self.C2L[:3, 3].reshape(3, 1)
        return np.transpose(points3d_lidar)

    # ===========================
    # ------- 3d to 2d ----------
    # ===========================
    def project_camera_to_image(self, pts_3d_camera):
        ''' Input: nx3 points in camera coord.
            Output: nx2 points in image2 coord.
        '''

        # inliner_indices_arr = np.arange(pts_3d_camera.shape[1])
        # condition = pts_3d_camera[2, :] > 0.0
        # pts_3d_camera = pts_3d_camera[:, condition]
        # inliner_indices_arr = inliner_indices_arr[condition]

        points2d_camera = np.matmul(self.K, pts_3d_camera)
        points2d_camera = (points2d_camera[:2, :] / points2d_camera[2, :]).T

        return points2d_camera

    def project_lidar_to_image(self, pts_3d_lidar):
        ''' Input: nx3 points in lidardyne coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_camera = self.project_lidar_to_camera(pts_3d_lidar)
        return self.project_camera_to_image(np.transpose(pts_3d_camera))

    def project_ego_to_image(self, pts_3d_ego):
        ''' Input: nx3 points in lidardyne coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_camera = self.project_ego_to_camera(pts_3d_ego)
        return self.project_camera_to_image(np.transpose(pts_3d_camera))

    # ===========================
    # ------- 2d to 3d ----------
    # ===========================
    def project_image_to_camera(self, uv_depth):
        ''' Input: nx3 first two channels are uv, 3rd channel
                   is depth in camera coord.
            Output: nx3 points in camera coord.
        '''
        n = uv_depth.shape[0]
        x = ((uv_depth[:,0]-self.c_u)*uv_depth[:,2])/self.f_u
        y = ((uv_depth[:,1]-self.c_v)*uv_depth[:,2])/self.f_v
        pts_3d_camera = np.zeros((n,3))
        pts_3d_camera[:,0] = x
        pts_3d_camera[:,1] = y
        pts_3d_camera[:,2] = uv_depth[:,2]
        return pts_3d_camera

    def project_image_to_lidar(self, uv_depth):
        pts_3d_camera = self.project_image_to_camera(uv_depth)
        return self.project_camera_to_lidar(pts_3d_camera)

def projection(lidar_points, camera_calib, filter_outliers=True):
    camera_pose_mat = camera_calib.C2E

    trans_lidar_to_camera = np.linalg.inv(camera_pose_mat)
    points3d_lidar = lidar_points
    points3d_camera = trans_lidar_to_camera[:3, :3] @ (points3d_lidar.T) + \
                        trans_lidar_to_camera[:3, 3].reshape(3, 1)

    K = np.eye(3, dtype=np.float64)
    K[0, 0] = camera_calib.f_u
    K[1, 1] = camera_calib.f_v
    K[0, 2] = camera_calib.c_u
    K[1, 2] = camera_calib.c_v

    inliner_indices_arr = np.arange(points3d_camera.shape[1])
    if filter_outliers:
        condition = points3d_camera[2, :] > 0.0
        points3d_camera = points3d_camera[:, condition]
        inliner_indices_arr = inliner_indices_arr[condition]

    points2d_camera = K @ points3d_camera
    points2d_camera = (points2d_camera[:2, :] / points2d_camera[2, :]).T

    if filter_outliers:
        image_w, image_h = (1920,1080) #camera_data.size
        condition = np.logical_and(
            (points2d_camera[:, 1] < image_h) & (points2d_camera[:, 1] > 0),
            (points2d_camera[:, 0] < image_w) & (points2d_camera[:, 0] > 0))
        points2d_camera = points2d_camera[condition]
        points3d_camera = (points3d_camera.T)[condition]
        inliner_indices_arr = inliner_indices_arr[condition]
    return points2d_camera, points3d_camera, inliner_indices_arr

def rotate_velo(t, rot):
    x = t[:,0] * np.cos(rot) - t[:,1] *  np.sin(rot)
    y = t[:,0] * np.sin(rot) + t[:,1] *  np.cos(rot)
    t[:,0] = x
    t[:,1] = y
    return t

## Visualize common utils.

# PandaSet frame and sequence ids.
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

# Bird view representation
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
