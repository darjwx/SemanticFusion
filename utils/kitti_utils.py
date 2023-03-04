""" Helper methods for loading and parsing KITTI data.

Author: Charles R. Qi
Date: September 2017
"""
from __future__ import print_function

import numpy as np
import os

from .calibration import *

class KittiCalibration(Calibration):
    ''' Calibration matrices and utils
        3d XYZ in <label>.txt are in rect camera coord.
        2d box xy are in image2 coord
        Points in <lidar>.bin are in Velodyne coord.

        y_image2 = P^2_rect * x_rect
        y_image2 = P^2_rect * R0_rect * Tr_velo_to_cam * x_velo
        x_ref = Tr_velo_to_cam * x_velo
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

        velodyne coord:
        front x, left y, up z

        rect/ref camera coord:
        right x, down y, front z

        Ref (KITTI paper): http://www.cvlibs.net/publications/Geiger2013IJRR.pdf

        TODO(rqi): do matrix multiplication only once for each projection.
    '''
    def __init__(self, calib_filepath, calib_dict=None, from_video=False):
        if calib_dict is None:
            assert os.path.exists(calib_filepath), calib_filepath
            if from_video:
                calibs = self.read_calib_from_video(calib_filepath)
            else:
                calibs = self.read_calib_file(calib_filepath)
        else:
            calibs = calib_dict

        self.calib_dict = calibs
        # Projection matrix from rect camera coord to image2 coord
        self.P = calibs['P2'] 
        self.P = np.reshape(self.P, [3,4])
        # Rigid transform from Velodyne coord to reference camera coord
        self.velo2cam = calibs['Tr']
        self.velo2cam = np.reshape(self.velo2cam, [3,4])
        self.cam2velo = inverse_rigid_trans(self.velo2cam)
        # Rotation from reference camera coord to rect camera coord
        # self.R0 = calibs['R0_rect']
        # self.R0 = np.reshape(self.R0,[3,3])

        # Camera intrinsics and extrinsics
        self.c_u = self.P[0,2]
        self.c_v = self.P[1,2]
        self.f_u = self.P[0,0]
        self.f_v = self.P[1,1]
        self.b_x = self.P[0,3]/(-self.f_u) # relative 
        self.b_y = self.P[1,3]/(-self.f_v)

        self.K = np.eye(3, dtype=np.float64)
        self.K[0, 0] = self.f_u
        self.K[1, 1] = self.f_v
        self.K[0, 2] = self.c_u
        self.K[1, 2] = self.c_v

    def read_calib_file(self, filepath):
        ''' Read in a calibration file and parse into a dictionary.
        Ref: https://github.com/utiasSTARS/pynuscenes/blob/master/pynuscenes/utils.py
        '''
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line)==0: continue
                key, value = line.split(':', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass

        return data
    
    def read_calib_from_video(self, calib_root_dir):
        ''' Read calibration for camera 2 from video calib files.
            there are calib_cam_to_cam and calib_velo_to_cam under the calib_root_dir
        '''
        data = {}
        if self.camera is None:
            print('calib_cam_to_cam.txt')
            print('calib_velo_to_cam.txt')
            cam2cam = self.read_calib_file(os.path.join(calib_root_dir, 'calib_cam_to_cam.txt'))
            velo2cam = self.read_calib_file(os.path.join(calib_root_dir, 'calib_velo_to_cam.txt'))
        else:
            print('calib_cam_to_cam%d.txt' % self.camera)
            print('calib_velo_to_cam%d.txt' % self.camera)
            cam2cam = self.read_calib_file(os.path.join(calib_root_dir, 'calib_cam_to_cam%d.txt' % self.camera))
            velo2cam = self.read_calib_file(os.path.join(calib_root_dir, 'calib_velo_to_cam%d.txt' % self.camera))
            Tr_velo_to_cam = np.zeros((3,4))
            Tr_velo_to_cam[0:3,0:3] = np.reshape(velo2cam['R'], [3,3])
            Tr_velo_to_cam[:,3] = velo2cam['T']
            data['Tr_velo_to_cam'] = np.reshape(Tr_velo_to_cam, [12])
            data['R0_rect'] = cam2cam['R_rect_00']
            data['P2'] = cam2cam['P_rect_02']
            return data

    def project_lidar_to_camera(self, pts_3d_lidar):
        points3d_camera = np.matmul(self.velo2cam[:3, :3], (pts_3d_lidar.T)) + \
                            self.velo2cam[:3, 3].reshape(3, 1)
        return np.transpose(points3d_camera)

    def project_lidar_to_image(self, pts_3d_lidar):
        ''' Input: nx3 points in lidardyne coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_camera = self.project_lidar_to_camera(pts_3d_lidar)
        return self.project_camera_to_image(np.transpose(pts_3d_camera))

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

def inverse_rigid_trans(Tr):
    ''' Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    '''
    inv_Tr = np.zeros_like(Tr) # 3x4
    inv_Tr[0:3,0:3] = np.transpose(Tr[0:3,0:3])
    inv_Tr[0:3,3] = np.dot(-np.transpose(Tr[0:3,0:3]), Tr[0:3,3])
    return inv_Tr


