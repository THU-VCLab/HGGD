import logging

import numpy as np
import open3d as o3d

camera = 'realsense'
logging.warn(f'Using {camera} camera now!')


def get_camera_intrinsic(camera=camera):
    if camera == 'kinect':
        intrinsics = np.array([[631.55, 0, 638.43], [0, 631.21, 366.50],
                               [0, 0, 1]])
    elif camera == 'realsense':
        intrinsics = np.array([[927.17, 0, 651.32], [0, 927.37, 349.62],
                               [0, 0, 1]])
    else:
        raise ValueError(
            'Camera format must be either "kinect" or "realsense".')
    return intrinsics


# # realsense
# intrinsics = np.array([[927.16973877, 0, 651.31506348],
#                        [0, 927.36688232, 349.62133789], [0, 0, 1]])

# # kinect
# intrinsics = np.array([[631.55, 0, 638.43],
#                        [0, 631.21, 366.50], [0, 0, 1]])
