import numpy as np

class Colours():
    def __init__(self):
        # RGB colours
        self.color_maps = {
            'att_mask': [
                [250, 250, 110], # Yellow: 3D sems
                [237, 247, 111],
                [224, 244, 112],
                [212, 241, 113],
                [200, 237, 115],
                [188, 234, 117],
                [176, 230, 120],
                [165, 226, 122],
                [153, 222, 124],
                [142, 218, 127],
                [131, 214, 129],
                [121, 210, 131],
                [110, 205, 133],
                [100, 201, 135],
                [90, 196, 137],
                [80, 191, 139],
                [70, 187, 140],
                [60, 182, 141],
                [50, 177, 142],
                [40, 172, 143],
                [30, 167, 143],
                [18, 162, 143],
                [18, 162, 143],
                [0, 152, 142],
                [0, 147, 141],
                [0, 142, 140],
                [0, 137, 138],
                [0, 132, 136],
                [0, 126, 134],
                [0, 121, 131],
                [5, 116, 128],
                [14, 111, 125],
                [21, 106, 121],
                [26, 101, 117],
                [30, 96, 113],
                [34, 91, 108],
                [37, 86, 103],
                [39, 81, 99],
                [41, 77, 93],
                [42, 72, 88]], # Dark blue: 2D sems

            'labels_pandaset': [
                [0,0,0], #Ignore
                [255, 0, 0], # Car
                [182, 89, 6], # Bicycle/Motorcycle
                [255, 128, 0], # Truck
                [204, 153, 255], # Pedestrian
                [255, 0, 255], # Road
                [180, 150, 200], # Sidewalk
                [241, 230, 255], # Buildings
                [147, 253, 194], # Nature
                [255, 246, 143]], # Signs

            'labels_carla': [
                [0,0,0], #Ignore
                [255, 0, 0], # Car
                [182, 89, 6], # Bicycle/Motorcycle
                [255, 128, 0], # Truck
                [204, 153, 255], # Pedestrian
                [255, 0, 255], # Road
                [180, 150, 200], # Sidewalk
                [241, 230, 255], # Buildings
                [147, 253, 194], # Nature
                [255, 246, 143]], # Signs

            'labels_kitti': [
                [0,0,0], # Ignore
                [0,0,142], # Car
                [119,11,32], # Bicycle
                [0,0,230], # Motorcycle
                [0,0,70], # Truck
                [220,20,60], # Person
                [255,0,0], # Rider
                [128,64,128], # Road
                [244,35,232], # Sidewalk
                [70,70,70], # Building
                [190,153,153], # Fence
                [107,142,35], # Vegetation
                [153,153,153], # Pole
                [220,220,0]] # Traffic sign
        }

    def get_color_map(self, id):
        return np.asarray(self.color_maps[id])

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
