""" Helper methods for loading and parsing KITTI data.

Author: Charles R. Qi
Date: September 2017
"""
from __future__ import print_function

import numpy as np
import os
import json, yaml

### CALIBRATION ###
class Calibration(object):
    ''' Calibration matrices and utils
        3d XYZ in <label>.txt are in camera coord.
        2d box xy are in image2 coord
        Points in <lidar>.bin are in lidar coord.

        y_image2 = P^2_rect * x_rect
        y_image2 = P^2_rect * R0_rect * Tr_lidar_to_cam * x_lidar
        x_ref = Tr_lidar_to_cam * x_lidar
        x_rect = R0_rect * x_ref

        P^2_rect = [f^2_u,  0,      c^2_u,  -f^2_u b^2_x;
                    0,      f^2_v,  c^2_v,  -f^2_v b^2_y;
                    0,      0,      1,      0]
                 = K * [1|t]

        image2 coord:
         ----> x-axis (u)
        |
        |
        v y-axis (v)

        lidar coord:
        front x, left y, up z

        rect/ref camera coord:
        right x, down y, front z

        Ref (KITTI paper): http://www.cvlibs.net/publications/Geiger2013IJRR.pdf

        TODO(rqi): do matrix multiplication only once for each projection.
    '''
    def __init__(self, root_dir, calib_info, from_video=False):
        ''' Read in a calibration file and parse into a dictionary.
        Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        '''

        with open(os.path.join(root_dir, 'extrinsic_calibration.yaml')) as f:
            self.extrinsics = yaml.safe_load(f)

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
                # print('camera_pose', self.camera_pose)

        if os.path.isfile(ego_poses):
            with open(ego_poses, 'r') as f:
                file_data = json.load(f)
                self.ego_pose = file_data[calib_info['idx']]
                # print('lidar_pose', self.lidar_pose)

        if os.path.isfile(gps_poses):
            with open(gps_poses, 'r') as f:
                file_data = json.load(f)
                self.gps_pose = file_data[calib_info['idx']]
                # print('gps_pose', self.gps_pose)

        lidar_pose_mat  = _heading_position_to_mat(self.extrinsics[calib_info['camera']]['extrinsic']['transform']['rotation'],
            self.extrinsics[calib_info['camera']]['extrinsic']['transform']['translation'])
        camera_pose_mat = _heading_position_to_mat(self.camera_pose['heading'], self.camera_pose['position'])

        self.C2E = camera_pose_mat
        self.E2C = np.linalg.inv(camera_pose_mat)

        self.L2C = self.V2C = lidar_pose_mat
        self.C2L = np.linalg.inv(lidar_pose_mat)

        self.E2L = np.matmul(self.C2L, self.E2C)
        self.L2E = np.linalg.inv(self.E2L)

        self.K  = np.eye(3, dtype=np.float64)
        self.P2 = np.zeros((3,4), dtype=np.float64)
        self.R0 = np.eye(3, dtype=np.float64)
        self.K[0, 0] = self.P2[0, 0] = self.f_u
        self.K[1, 1] = self.P2[1, 1] = self.f_v
        self.K[0, 2] = self.P2[0, 2] = self.c_u
        self.K[1, 2] = self.P2[1, 2] = self.c_v


    def cart_to_hom(self, pts_3d):
        ''' Input: nx3 points in Cartesian
            Oupput: nx4 points in Homogeneous by pending 1
        '''
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n,1))))
        return pts_3d_hom

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


    def lidar_to_rect(self, pts_3d_lidar):
        points3d_camera = np.matmul(self.L2C[:3, :3], (pts_3d_lidar.T)) + \
                            self.L2C[:3, 3].reshape(3, 1)
        return np.transpose(points3d_camera)

    def rect_to_lidar(self, pts_3d_camera):
        points3d_lidar = np.matmul(self.C2L[:3, :3], (pts_3d_camera.T)) + \
                            self.C2L[:3, 3].reshape(3, 1)
        return np.transpose(points3d_lidar)

    # ===========================
    # ------- 3d to 2d ----------
    # ===========================
    def rect_to_img(self, pts_3d_camera):
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

    def lidar_to_img(self, pts_3d_lidar):
        ''' Input: nx3 points in lidardyne coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_camera = self.lidar_to_rect(pts_3d_lidar)
        return self.rect_to_img(np.transpose(pts_3d_camera))

    def project_ego_to_image(self, pts_3d_ego):
        ''' Input: nx3 points in lidardyne coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_camera = self.project_ego_to_camera(pts_3d_ego)
        return self.rect_to_img(np.transpose(pts_3d_camera))

    # ===========================
    # ------- 2d to 3d ----------
    # ===========================
    def img_to_rect(self, uv_depth):
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
        pts_3d_camera = self.img_to_rect(uv_depth)
        return self.rect_to_lidar(pts_3d_camera)
