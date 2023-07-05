import numpy as np

intrinsics = np.array([[927.16973877, 0, 651.31506348],
                       [0, 927.36688232, 349.62133789], [0, 0, 1]])
import math

import numpy as np
import torch

TABLE_HEIGHT = 0.75
TABLE_COLLISION_OFFSET = 0.005

# Workspace should be (6,): low_x, high_x, low_y, high_y, low_z, high_z
# WORKSPACE = [-0.40, 0.40, -0.35, 0.35, TABLE_HEIGHT - 0.001, TABLE_HEIGHT + 0.45]
VOXEL_SIZE = 0.005
NUM_POINTS_THRESHOLD = 8  #8
RADIUS_THRESHOLD = 0.04

# Normal Estimation
NORMAL_RADIUS = 0.01
NORMAL_MAX_NN = 30

# Scene Point Cloud
SCENE_MULTIPLE = 8  # The density of points in scene over view cloud
# Local Frame Search
#LENGTH_SEARCH = [-0.08, -0.06, -0.04, -0.02]
LENGTH_SEARCH = [-0.040, -0.030, -0.015, -0.000]
THICKNESS_SEARCH = [0]
THETA_SEARCH = list(range(-90, 90, 5))  # 15
CURVATURE_RADIUS = 0.01
BACK_COLLISION_THRESHOLD = 0 * math.sqrt(
    SCENE_MULTIPLE
)  # if more than this number of points exist behind the back of hand, grasp fail
# BACK_COLLISION_MARGIN = 0.035  # depth base, points that collide with back hand within this range will not be detected
FINGER_COLLISION_THRESHOLD = 0
CLOSE_REGION_MIN_POINTS = 16
for i in range(len(THETA_SEARCH)):
    THETA_SEARCH[i] /= 57.29578

# Gripper Configuration
FINGER_WIDTH = 0.004
# HALF_BOTTOM_WIDTH = 0.06 / 2 + FINGER_WIDTH  # width
HALF_HAND_THICKNESS = 0.005  # height
FINGER_LENGTH = 0.06
HAND_LENGTH = 0.12
# HALF_BOTTOM_SPACE = HALF_BOTTOM_WIDTH - FINGER_WIDTH

# GRIPPER_BOUND = np.ones([4, 8])
# i = 0
# for x in [FINGER_LENGTH, -BOTTOM_LENGTH]:
#     for y in [HALF_BOTTOM_WIDTH, -HALF_BOTTOM_WIDTH]:
#         for z in [HALF_HAND_THICKNESS, -HALF_HAND_THICKNESS]:
#             GRIPPER_BOUND[0:3, i] = [x, y, z]
#             i += 1
# TORCH_GRIPPER_BOUND = torch.tensor(GRIPPER_BOUND).float()

# Antipodal Grasp
NEIGHBOR_DEPTH = torch.tensor(0.005)

# GPD Projection Configuration
GRASP_NUM = 600
PROJECTION_RESOLUTION = 60
PROJECTION_MARGIN = 1

CAMERA_POSE = [[0.8, 0, 1.7, 0.948, 0, 0.317, 0],
               [-0.8, 0, 1.6, -0.94, 0, 0.342, 0],
               [0.0, 0.75, 1.7, 0.671, -0.224, 0.224, 0.671],
               [0.0, -0.75, 1.6, -0.658, -0.259, -0.259, 0.658]]
