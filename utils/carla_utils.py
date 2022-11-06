import os
import numpy as np

class CarlaCalibration(object):

    def __init__(self, calib_filepath, camera):
        calibs = self.read_calib_file(calib_filepath, camera)
        # Projection matrix from rect camera coord to image2 coord
        self.P = calibs['P_rect_02']
        self.P = np.reshape(self.P, [3, 4])

        # Rigid transform from Velodyne coord to reference camera coord
        self.V2C_R = np.reshape(calibs['R'], [3,3])
        self.V2C_T = np.reshape(calibs['T'], [3,1])
        self.V2C = np.hstack([self.V2C_R, self.V2C_T])

        self.C2V = inverse_rigid_trans(self.V2C)
        # Rotation from reference camera coord to rect camera coord
        self.R0 = calibs['R_rect_00']
        self.R0 = np.reshape(self.R0, [3, 3])

        # Camera intrinsics and extrinsics
        self.c_u = self.P[0, 2]
        self.c_v = self.P[1, 2]
        self.f_u = self.P[0, 0]
        self.f_v = self.P[1, 1]
        self.b_x = self.P[0, 3] / (-self.f_u)  # relative
        self.b_y = self.P[1, 3] / (-self.f_v)

    def read_calib_file(self, filepath, camera):
        ''' Read in a calibration file and parse into a dictionary.
        Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        '''
        data = {}
        with open(os.path.join(filepath, 'calib_cam_to_cam'+str(camera[-1]) + '.txt'), 'r') as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line) == 0: continue
                key, value = line.split(':', 1)
                # We do not care about the non-float values in this files
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass

        with open(os.path.join(filepath, 'calib_lidar_to_cam' + str(camera[-1]) + '.txt'), 'r') as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line) == 0: continue
                key, value = line.split(':', 1)
                # We do not care about the non-float values in this files
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
        return data

    def cart2hom(self, pts_3d):
        ''' Input: nx3 points in Cartesian
            Oupput: nx4 points in Homogeneous by pending 1
        '''
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
        return pts_3d_hom

    # 3D to 2D
    def project_lidar_to_image(self, pts_3d_lidar):
        ''' Input: nx3 points in velodyne coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_rect = self.project_lidar_to_rect(pts_3d_lidar)
        return self.project_rect_to_image(pts_3d_rect)
    def project_rect_to_image(self, pts_3d_rect):
        ''' Input: nx3 points in rect camera coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_rect = self.cart2hom(pts_3d_rect)
        pts_2d = np.dot(pts_3d_rect, np.transpose(self.P))  # nx3
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        return pts_2d[:, 0:2]

    # 3D to 3D
    def project_lidar_to_rect(self, pts_3d_lidar):
        pts_3d_ref = self.project_lidar_to_ref(pts_3d_lidar)
        return self.project_ref_to_rect(pts_3d_ref)
    def project_lidar_to_ref(self, pts_3d_lidar):
        pts_3d_lidar = self.cart2hom(pts_3d_lidar)  # nx4

        return np.dot(pts_3d_lidar, np.transpose(self.V2C))
    def project_ref_to_rect(self, pts_3d_ref):
        ''' Input and Output are nx3 points '''
        return np.transpose(np.dot(self.R0, np.transpose(pts_3d_ref)))

def inverse_rigid_trans(Tr):
    ''' Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    '''
    inv_Tr = np.zeros_like(Tr) # 3x4
    inv_Tr[0:3,0:3] = np.transpose(Tr[0:3,0:3])
    inv_Tr[0:3,3] = np.dot(-np.transpose(Tr[0:3,0:3]), Tr[0:3,3])
    return inv_Tr
