from time import time

import cupoch
import numpy as np
import open3d as o3d
import torch
from torch.utils.dlpack import to_dlpack


class ModelFreeCollisionDetector():

    def __init__(self, scene_points, voxel_size=0.005, mode='regnet'):
        self.mode = mode
        self.voxel_size = voxel_size
        # regnet
        if mode == 'regnet':
            self.delta_width = 0
            self.height = 0.01
            self.finger_width = 0.01  # 0.01
            self.delta_width = 0  # 0
            self.finger_length = 0.06
            self.depth = 0.025
        # graspnet train
        elif mode == 'graspnet':
            self.height = 0.01
            self.finger_width = 0.01
            self.delta_width = 0
            self.finger_length = 0.04
            self.depth = 0.02
        elif mode == 'test':
            self.height = 0.02  # 0.02
            self.finger_width = 0.01  # 0.01
            self.delta_width = 0
            # self.delta_width = -self.finger_width / 2
            self.finger_length = 0.04  # 0.04
            self.depth = 0.02  # 0.02
        else:
            raise ValueError(f'Invalid collision detection mode: {mode}')

        # down sample to voxel
        scene_cloud = cupoch.geometry.PointCloud()
        scene_cloud.from_points_dlpack(to_dlpack(scene_points))
        scene_cloud = scene_cloud.voxel_down_sample(voxel_size)
        self.scene_points = np.array(scene_cloud.points.cpu(),
                                     dtype=np.float16)
        # self.scene_points = scene_points.cpu().numpy()

    def detect(self, grasp_group, approach_dist=0.05):
        T = torch.from_numpy(grasp_group.translations).to(dtype=torch.float16,
                                                          device='cuda')
        R = torch.from_numpy(grasp_group.rotations.reshape(-1, 3, 3)).to(
            dtype=torch.float16, device='cuda')
        heights = torch.full((grasp_group.size, 1),
                             self.height,
                             dtype=torch.float16,
                             device='cuda')
        depths = torch.full((grasp_group.size, 1),
                            self.depth,
                            dtype=torch.float16,
                            device='cuda')
        widths = torch.from_numpy(grasp_group.widths[:, None]).to(
            dtype=torch.float16, device='cuda')
        widths += self.delta_width
        points = torch.from_numpy(self.scene_points[None, ...]).to(
            dtype=torch.float16, device='cuda')
        targets = torch.matmul(points - T[:, None, :], R)
        # collision detection
        # height mask
        mask1 = ((targets[..., 2] > -heights / 2) &
                 (targets[..., 2] < heights / 2))
        # left finger mask
        mask2 = ((targets[..., 0] > depths - self.finger_length) &
                 (targets[..., 0] < depths))
        mask3 = (targets[..., 1] > -(widths / 2 + self.finger_width))
        mask4 = (targets[..., 1] < -widths / 2)
        # right finger mask
        mask5 = (targets[..., 1] < (widths / 2 + self.finger_width))
        mask6 = (targets[..., 1] > widths / 2)
        # shifting mask
        mask7 = ((targets[..., 0] <= depths - self.finger_length)\
                & (targets[..., 0] > depths - self.finger_length - self.finger_width - approach_dist))

        # get collision mask of each point
        depth_mask = (mask1 & mask2)
        left_mask = (depth_mask & mask3 & mask4)
        right_mask = (depth_mask & mask5 & mask6)
        shifting_mask = (mask1 & mask3 & mask5 & mask7)

        # check mininum point count between points
        mask_between = (depth_mask & (~mask4) & (~mask6))
        min_points = 2
        mask_between = (mask_between.sum(1) > min_points)

        # get collision of finger
        finger_mask = (left_mask | right_mask)
        mask_finger = (finger_mask.sum(1) <= 0)
        # get collision of shifting area
        mask_shift = (shifting_mask.sum(1) <= 0)

        # get collison mask
        no_collision_mask = (mask_between & mask_finger & mask_shift)
        return no_collision_mask.cpu().numpy()
