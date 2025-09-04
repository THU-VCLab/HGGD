import copy
import math

import cv2
import numpy as np
import open3d as o3d

from customgraspnetAPI import Grasp, GraspGroup, GraspNet, RectGraspGroup

EPS = 1e-8


def compute_point_dists(A, B):
    """Compute pair-wise point distances in two matrices.

    Input:
        A: [np.ndarray, (N,3), np.float32]
            point cloud A
        B: [np.ndarray, (M,3), np.float32]
            point cloud B

    Output:
        dists: [np.ndarray, (N,M), np.float32]
            distance matrix
    """
    A = A[:, np.newaxis, :]
    B = B[np.newaxis, :, :]
    dists = np.linalg.norm(A - B, axis=-1)
    return dists


def remove_invisible_grasp_points(cloud, grasp_points, th=0.01):
    """Remove invisible part of object model according to scene point cloud.

    Input:
        cloud: [np.ndarray, (N,3), np.float32]
            scene point cloud
        grasp_points: [np.ndarray, (M,3), np.float32]
            grasp point label in object coordinates
        th: [float]
            if the minimum distance between a grasp point and the scene points is greater than outlier, the point will be removed

    Output:
        visible_mask: [np.ndarray, (M,), np.bool]
            mask to show the visible part of grasp points
    """
    dists = compute_point_dists(grasp_points, cloud)
    min_dists = dists.min(axis=1)
    visible_mask = (min_dists < th)
    return visible_mask


# 6D->2D


def get_camera_intrinsic(camera):
    '''
    **Input:**

    - camera: string of type of camera, "realsense" or "kinect".

    **Output:**

    - numpy array of shape (3, 3) of the camera intrinsic matrix.
    '''
    param = o3d.camera.PinholeCameraParameters()
    if camera == 'kinect':
        param.intrinsic.set_intrinsics(1280, 720, 631.55, 631.21, 638.43,
                                       366.50)
    elif camera == 'realsense':
        param.intrinsic.set_intrinsics(1280, 720, 927.17, 927.37, 651.32,
                                       349.62)
    intrinsic = param.intrinsic.intrinsic_matrix
    return intrinsic


def batch_rgbdxyz_2_rgbxy_depth(points, camera):
    '''
    **Input:**

    - points: np.array(-1,3) of the points in camera frame

    - camera: string of the camera type

    **Output:**

    - coords: float of xy in pixel frame [-1, 2]

    - depths: float of the depths of pixel frame [-1]
    '''
    intrinsics = get_camera_intrinsic(camera)
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
    coords_x = points[:, 0] / points[:, 2] * fx + cx
    coords_y = points[:, 1] / points[:, 2] * fy + cy
    coords = np.stack([coords_x, coords_y], axis=-1)
    return coords, depths


def get_batch_key_points(centers, Rs, widths):
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


def batch_key_points_2_tuple(key_points, scores, object_ids, camera):
    '''
    **Input:**

    - key_points: np.array(-1,4,3) of grasp key points, definition is shown in key_points.png

    - scores: numpy array of batch grasp scores.

    - camera: string of 'realsense' or 'kinect'.

    **Output:**

    - np.array([center_x,center_y,open_x,open_y,height])
    '''
    import numpy as np
    centers, _ = batch_rgbdxyz_2_rgbxy_depth(key_points[:, 0, :], camera)
    opens, _ = batch_rgbdxyz_2_rgbxy_depth(key_points[:, 1, :], camera)
    lefts, _ = batch_rgbdxyz_2_rgbxy_depth(key_points[:, 2, :], camera)
    rights, _ = batch_rgbdxyz_2_rgbxy_depth(key_points[:, 3, :], camera)
    heights = np.linalg.norm(lefts - rights, axis=-1, keepdims=True)
    tuples = np.concatenate([
        centers, opens, heights, scores[:, np.newaxis], object_ids[:,
                                                                   np.newaxis]
    ],
                            axis=-1).astype(np.float32)  # 每个属性代表一列或两列
    return tuples


def batch_key_point_2_rotation(centers_xyz, open_points_xyz, upper_points_xyz):
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
    unit_open_points_vector = open_points_vector / np.hstack(
        (open_point_norm, open_point_norm, open_point_norm))  # (-1, 3)
    unit_upper_points_vector = upper_points_vector / np.hstack(
        (upper_point_norm, upper_point_norm, upper_point_norm))  # (-1, 3)
    num = open_points_vector.shape[0]
    x_axis = np.hstack((np.zeros((num, 1)), np.zeros(
        (num, 1)), np.ones((num, 1)))).astype(np.float32).reshape(-1, 3, 1)
    rotations = np.dstack((x_axis, unit_open_points_vector.reshape(
        (-1, 3, 1)), unit_upper_points_vector.reshape((-1, 3, 1))))
    return rotations


def to_rect_grasp_group(_6d_grasp, camera):
    '''
    **Input:**

    - camera: string of type of camera, 'realsense' or 'kinect'.

    **Output:**

    - RectGraspGroup instance or None.
    '''
    tranlations = _6d_grasp.translations
    rotations = _6d_grasp.rotation_matrices
    depths = _6d_grasp.depths
    scores = _6d_grasp.scores
    widths = _6d_grasp.widths
    object_ids = _6d_grasp.object_ids

    if tranlations.shape[0] == 0:
        return None

    k_points = get_batch_key_points(tranlations, rotations, widths)
    k_points = k_points.reshape([-1, 3])
    k_points = k_points.reshape([-1, 4, 3])
    rect_grasp_group_array = batch_key_points_2_tuple(k_points, scores,
                                                      object_ids, camera)
    rect_grasp_group = RectGraspGroup()
    rect_grasp_group.rect_grasp_group_array = rect_grasp_group_array
    return k_points, rect_grasp_group


def to_open3d_geometry_list(_6d_grasp, color):
    '''
    **Output:**

    - list of open3d.geometry.Geometry of the grippers.
    '''
    geometry = []
    for i in range(len(_6d_grasp.grasp_group_array)):
        g = Grasp(_6d_grasp.grasp_group_array[i])
        geometry.append(g.to_open3d_geometry(color))
    return geometry


def mask_grasp_group(_6d_grasp_group, threshold):
    tranlations = _6d_grasp_group.translations
    rotations = _6d_grasp_group.rotation_matrices
    depths = _6d_grasp_group.depths
    scores = _6d_grasp_group.scores
    widths = _6d_grasp_group.widths
    object_ids = _6d_grasp_group.object_ids
    mask = (rotations[:, 2, 0] > threshold) & (widths > 0.03) & (
        widths < 0.15) & (depths > 0.015) & (depths < 0.025)
    tranlations = tranlations[mask]
    depths = depths[mask]
    widths = widths[mask]
    scores = scores[mask]
    rotations = rotations[mask]
    object_ids = object_ids[mask]

    _6d_grasp_group_mask = _6d_grasp_group[mask]
    return _6d_grasp_group_mask, mask


#######################################

# 2D->6D


def batch_framexy_depth_2_xyz(pixel_x, pixel_y, depth, camera):
    '''
    **Input:**

    - pixel_x: numpy array of int of the pixel x coordinate. shape: (-1,)

    - pixel_y: numpy array of int of the pixle y coordicate. shape: (-1,)

    - depth: numpy array of float of depth.

    - camera: string of type of camera. "realsense" or "kinect".

    **Output:**

    x, y, z: numpy array of float of x, y and z coordinates in camera frame. The unit is millimeter.
    '''
    intrinsics = get_camera_intrinsic(camera)
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    z = depth
    x = z / fx * (pixel_x - cx)
    y = z / fy * (pixel_y - cy)
    return x, y, z


def batch_get_key_points(rect_grasp_group):
    '''
    **Output:**

    - center, open_point, upper_point, each of them is a numpy array of shape (2,)
    '''
    open_points = rect_grasp_group.open_points  # (-1, 2)
    centers = rect_grasp_group.center_points  # (-1, 2)
    heights = (rect_grasp_group.heights).reshape((-1, 1))  # (-1, )
    open_point_vector = open_points - centers
    norm_open_point_vector = np.linalg.norm(open_point_vector,
                                            axis=1).reshape(-1, 1)
    unit_open_point_vector = open_point_vector / np.hstack(
        (norm_open_point_vector, norm_open_point_vector))  # (-1, 2)
    counter_clock_wise_rotation_matrix = np.array([[0, -1], [1, 0]])
    # upper_points = np.dot(counter_clock_wise_rotation_matrix, unit_open_point_vector.reshape(-1, 2, 1)).reshape(-1, 2) * np.hstack([heights, heights]) / 2 + centers # (-1, 2)
    upper_points = np.einsum(
        'ij,njk->nik', counter_clock_wise_rotation_matrix,
        unit_open_point_vector.reshape(-1, 2, 1)).reshape(-1, 2) * np.hstack(
            [heights, heights]) / 2 + centers  # (-1, 2)
    return centers, open_points, upper_points


def batch_center_depth(depths, centers, open_points, upper_points):
    '''
    **Input:**

    - depths: numpy array of the depths.

    - centers: numpy array of the center points of shape(-1, 2).

    - open_points: numpy array of the open points of shape(-1, 2).

    - upper_points: numpy array of the upper points of shape(-1, 2).

    **Output:**

    - depths: numpy array of the grasp depth of shape (-1).
    '''
    x = np.round(centers[:, 0]).astype(np.int32)
    y = np.round(centers[:, 1]).astype(np.int32)
    return depths[y, x]


def to_grasp_group(rect_grasp_group, camera, center_z_depths):
    '''
    **Input:**

    - camera: string of type of camera, 'kinect' or 'realsense'.

    - depths: numpy array of the depths image.

    - depth_method: function of calculating the depth.

    **Output:**

    - grasp_group: GraspGroup instance or None.

    .. note:: The number may not be the same to the input as some depth may be invalid.
    '''

    centers, open_points, upper_points = batch_get_key_points(rect_grasp_group)
    depths_2d = center_z_depths / 1000.0

    valid_num = centers.shape[0]
    if valid_num == 0:
        return None
    centers_xyz = np.array(
        batch_framexy_depth_2_xyz(centers[:, 0], centers[:, 1], depths_2d,
                                  camera)).T
    open_points_xyz = np.array(
        batch_framexy_depth_2_xyz(open_points[:, 0], open_points[:, 1],
                                  depths_2d, camera)).T
    upper_points_xyz = np.array(
        batch_framexy_depth_2_xyz(upper_points[:, 0], upper_points[:, 1],
                                  depths_2d, camera)).T
    depths = 0.00 * np.ones((valid_num, 1))
    heights = (np.linalg.norm(upper_points_xyz - centers_xyz, axis=1) *
               2).reshape((-1, 1))
    widths = (np.linalg.norm(open_points_xyz - centers_xyz, axis=1) *
              2).reshape((-1, 1))
    scores = (rect_grasp_group.scores).reshape((-1, 1))
    object_ids = (rect_grasp_group.object_ids).reshape((-1, 1))

    translations = centers_xyz
    rotations = batch_key_point_2_rotation(centers_xyz, open_points_xyz,
                                           upper_points_xyz).reshape((-1, 9))
    grasp_group = GraspGroup()
    grasp_group.grasp_group_array = copy.deepcopy(
        np.hstack((scores, widths, heights, depths, rotations, translations,
                   object_ids))).astype(np.float64)
    return grasp_group


def to_opencv_image(rect_grasp_group, opencv_rgb, shuffle, numGrasp=0):
    '''
    **input:**

    - opencv_rgb: numpy array of opencv BGR format.

    - numGrasp: int of the number of grasp, 0 for all.

    **Output:**

    - numpy array of opencv RGB format that shows the rectangle grasps.
    '''
    img = copy.deepcopy(opencv_rgb)
    if numGrasp == 0:
        numGrasp = rect_grasp_group.__len__()

    shuffled_rect_grasp_group_array = copy.deepcopy(
        rect_grasp_group.rect_grasp_group_array)
    if shuffle:
        np.random.shuffle(shuffled_rect_grasp_group_array)
    for rect_grasp_array in shuffled_rect_grasp_group_array[:numGrasp]:
        center_x, center_y, open_x, open_y, height, score, object_id = rect_grasp_array
        center = np.array([center_x, center_y])
        left = np.array([open_x, open_y])
        axis = left - center
        normal = np.array([-axis[1], axis[0]])
        normal = normal / np.linalg.norm(normal) * height / 2
        p1 = center + normal + axis
        p2 = center + normal - axis
        p3 = center - normal - axis
        p4 = center - normal + axis
        # cv2.circle(img, (int(center_x), int(center_y)), 2, (0, 0, 255), 2)
        cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])),
                 (0, 0, 255), 1, 8)
        cv2.line(img, (int(p2[0]), int(p2[1])), (int(p3[0]), int(p3[1])),
                 (255, 0, 0), 3, 8)
        cv2.line(img, (int(p3[0]), int(p3[1])), (int(p4[0]), int(p4[1])),
                 (0, 0, 255), 1, 8)
        cv2.line(img, (int(p4[0]), int(p4[1])), (int(p1[0]), int(p1[1])),
                 (255, 0, 0), 3, 8)
    return img
