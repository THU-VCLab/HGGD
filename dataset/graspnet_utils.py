import copy
import math

import cv2
import numpy as np
import open3d as o3d

EPS = 1e-8
from .config import get_camera_intrinsic

# 6D->2D


def points_2_pixel_depth(points):
    '''
    **Input:**

    - points: np.array(-1,3) of the points in camera frame

    **Output:**

    - coords: float of xy in pixel frame [-1, 2]

    - depths: float of the depths of pixel frame [-1]
    '''
    intrinsics = get_camera_intrinsic()
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    s = 1000.0
    depths = s * points[:, 2]  # point_z
    ###################################
    # x and y should be inverted here #
    ###################################
    # y = point[0] / point[2] * fx + cx
    # x = point[1] / point[2] * fy + cy
    # cx = 640, cy = 360
    coords_x = points[:, 0] / points[:, 2] * fx + cx  # 1280
    coords_y = points[:, 1] / points[:, 2] * fy + cy  # 720
    coords = np.stack([
        coords_x, coords_y
    ], axis=-1)  # attention: make sure (1280, 720) order [for data generation]
    return coords, depths


def get_6d_key_points(centers, Rs, widths):
    '''
    **Input:**

    - centers: np.array(-1,3) of the translation

    - Rs: np.array(-1,3,3) of the rotation matrix

    - widths: np.array(-1) of the grasp width

    **Output:**

    - key_points: np.array(-1,4,3) of the key point of the grasp
    '''
    import numpy as np
    depth_base = 0.0
    height = 0.02
    key_points = np.zeros((centers.shape[0], 4, 3), dtype=np.float32)
    key_points[:, :, 0] -= depth_base
    key_points[:, 1:, 1] -= widths[:, np.newaxis] / 2
    key_points[:, 2, 2] += height / 2
    key_points[:, 3, 2] -= height / 2
    key_points = np.matmul(Rs, key_points.transpose(0, 2,
                                                    1)).transpose(0, 2, 1)
    key_points = key_points + centers[:, np.newaxis, :]
    return key_points


def points_2_2d_tuple(key_points):
    '''
    **Input:**

    - key_points: np.array(-1,4,3) of grasp key points, definition is shown in key_points.png

    - scores: numpy array of batch grasp scores.

    **Output:**

    - np.array([center_x,center_y,open_x,open_y,height])
    '''
    import numpy as np
    centers, _ = points_2_pixel_depth(key_points[:, 0, :])
    opens, _ = points_2_pixel_depth(key_points[:, 1, :])
    lefts, _ = points_2_pixel_depth(key_points[:, 2, :])
    rights, _ = points_2_pixel_depth(key_points[:, 3, :])
    heights = np.linalg.norm(lefts - rights, axis=-1, keepdims=True).squeeze(1)
    # tuples = np.concatenate([centers, opens, heights, scores[:, np.newaxis], object_ids[:, np.newaxis]],
    #                         axis=-1).astype(np.float32) # 每个属性代表一列或两列
    return centers, opens, heights


#######################################

# 2D->6D


def pixel_depth_2_points(pixel_x, pixel_y, depth):
    '''
    **Input:**

    - pixel_x: numpy array of int of the pixel x coordinate. shape: (-1,)

    - pixel_y: numpy array of int of the pixle y coordicate. shape: (-1,)

    - depth: numpy array of float of depth. The unit is millimeter. shape: (-1,)

    **Output:**

    x, y, z: numpy array of float of x, y and z coordinates in camera frame. The unit is millimeter.
    '''
    # attetion: (pixel_x, pixle_y) range in (1280, 720)
    intrinsics = get_camera_intrinsic()
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    z = depth  # mm
    x = z / fx * (pixel_x - cx)  # mm
    y = z / fy * (pixel_y - cy)  # mm
    return np.array([x, y, z]).T


def get_2d_key_points(centers_2d, heights_2d, open_points):
    '''
    **Output:**

    - center, open_point, upper_point, each of them is a numpy array of shape (2,)
    '''
    # # open_points = rect_grasp_group.opens  # (-1, 2)
    # centers = rect_grasp_group.centers  # (-1, 2)
    # heights = (rect_grasp_group.heights).reshape((-1, 1))  # (-1, )
    open_point_vector = open_points - centers_2d
    norm_open_point_vector = np.linalg.norm(open_point_vector,
                                            axis=1).reshape(-1, 1)
    unit_open_point_vector = open_point_vector / np.hstack(
        (norm_open_point_vector, norm_open_point_vector))  # (-1, 2)
    counter_clock_wise_rotation_matrix = np.array([[0, -1], [1, 0]])
    # upper_points = np.dot(counter_clock_wise_rotation_matrix, unit_open_point_vector.reshape(-1, 2, 1)).reshape(-1, 2) * np.hstack([heights, heights]) / 2 + centers # (-1, 2)
    upper_points = np.einsum(
        'ij,njk->nik', counter_clock_wise_rotation_matrix,
        unit_open_point_vector.reshape(-1, 2, 1)).reshape(-1, 2) * np.hstack(
            [heights_2d, heights_2d]) / 2 + centers_2d  # (-1, 2)
    return upper_points


def key_point_2_rotation(centers_xyz, open_points_xyz, upper_points_xyz):
    '''
    **Input:**

    - centers_xyz: numpy array of the center points of shape (-1, 3).

    - open_points_xyz: numpy array of the open points of shape (-1, 3).

    - upper_points_xyz: numpy array of the upper points of shape (-1, 3).

    **Output:**

    - rotations: numpy array of the rotation matrix of shape (-1, 3, 3).
    '''
    # print('open_points_xyz:{}'.format(open_points_xyz))
    # print('upper_points_xyz:{}'.format(upper_points_xyz))
    open_points_vector = open_points_xyz - centers_xyz  # (-1, 3)
    upper_points_vector = upper_points_xyz - centers_xyz  # (-1, 3)
    open_point_norm = np.linalg.norm(open_points_vector, axis=1).reshape(-1, 1)
    upper_point_norm = np.linalg.norm(upper_points_vector,
                                      axis=1).reshape(-1, 1)
    # print('open_point_norm:{}, upper_point_norm:{}'.format(open_point_norm, upper_point_norm))
    unit_open_points_vector = open_points_vector / (np.hstack(
        (open_point_norm, open_point_norm, open_point_norm)) + EPS)  # (-1, 3)
    unit_upper_points_vector = upper_points_vector / (np.hstack(
        (upper_point_norm, upper_point_norm, upper_point_norm)) + EPS
                                                      )  # (-1, 3)
    num = open_points_vector.shape[0]
    x_axis = np.hstack((np.zeros((num, 1)), np.zeros(
        (num, 1)), np.ones((num, 1)))).astype(np.float32).reshape(-1, 3, 1)
    rotations = np.dstack((x_axis, unit_open_points_vector.reshape(
        (-1, 3, 1)), unit_upper_points_vector.reshape((-1, 3, 1))))
    return rotations


if __name__ == '__main__':
    pass
