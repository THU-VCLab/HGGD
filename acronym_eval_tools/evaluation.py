import os
from time import time

import numpy as np
import open3d
import torch
import transforms3d
from tqdm import tqdm, trange

from .configs import config
from .pointcloud import PointCloud
from .torch_scene_point_cloud import TorchScenePointCloud


def eval_validate(formal_dict,
                  predicted_grasp,
                  camera_pose,
                  table_height,
                  depths,
                  widths,
                  colli_detect=False):
    '''
    formal_dict:  {}
    predicted_grasp: [B, 8]
    '''
    view_cloud = EvalDataValidate(formal_dict, predicted_grasp, camera_pose,
                                  table_height, depths, widths)
    vgr, score, no_colli_idxs, grasp_nocoll_view_num, grasp_nocoll_view, grasp_nocoll_scene = view_cloud.run_collision(
        colli_detect=colli_detect)
    return vgr, score, no_colli_idxs, grasp_nocoll_view_num, grasp_nocoll_view, grasp_nocoll_scene


class EvalDataValidate(PointCloud):

    def __init__(self,
                 pc_path,
                 pred_grasp,
                 camera_pose,
                 table_height,
                 depths,
                 widths,
                 visualization=False):
        '''
          data:  dict {'point_cloud': [3,N1], 'view_cloud' : [N1,3],
                     'scene_cloud': [N2,3], 'scene_cloud_table': [N3,3], ... }
          grasp: [B, 8] or [B, 4, 4]
        '''
        pc_data = np.load(pc_path[0])
        points = pc_data['view_cloud_camera']
        self.table_height = table_height
        cloud = open3d.geometry.PointCloud()
        cloud.points = open3d.utility.Vector3dVector(points)
        PointCloud.__init__(self, cloud, visualization)
        self.depths = depths
        self.widths = widths

        self.cloud_array = torch.FloatTensor(points).cuda()
        self.cloud_array_homo = torch.cat([
            self.cloud_array.transpose(0, 1),
            torch.ones(1, self.cloud_array.shape[0]).cuda()
        ],
                                          dim=0)
        self.pred_grasp = torch.FloatTensor(pred_grasp).cuda()

        self.frame, self.center, self.score = self.pred_grasp[:,:3,:3].contiguous(),\
            self.pred_grasp[:,:3,3].contiguous(), self.pred_grasp[:,3,3].contiguous()
        # self.frame [B,3,3], self.center [B,3], self.score [B,1]

        self.global_to_local = torch.eye(4).unsqueeze(0).expand(
            self.frame.shape[0], 4, 4).cuda().contiguous()
        self.global_to_local[:, 0:3, 0:3] = self.frame.transpose(1, 2)
        self.global_to_local[:, 0:3, 3:4] = -torch.bmm(
            self.frame.transpose(1, 2), self.center.unsqueeze(2))

        self.baseline_frame = torch.zeros(self.frame.shape[0],
                                          4,
                                          4,
                                          dtype=torch.float32).cuda()
        self.baseline_frame_index = torch.zeros(self.frame.shape[0],
                                                dtype=torch.float32).cuda()
        self.valid_grasp = 0

        self.antipodal_score = None
        self.collision_bool = None
        self.label_bool = None

        self.left_normal = torch.tensor([[0, 1, 0]]).to(dtype=torch.float32,
                                                        device='cuda')
        self.right_normal = torch.tensor([[0, -1, 0]]).to(dtype=torch.float32,
                                                          device='cuda')

        self.scene = TorchScenePointCloud(pc_data)

    def run_collision(self, colli_detect):
        if colli_detect:
            self.run_collision_view()
        self.run_collision_scene(self.scene, colli_detect)
        select_grasp = torch.nonzero(self.collision_bool).view(-1)
        self.grasp_no_collision_scene = self.pred_grasp[
            select_grasp]  # valid scene grasp

        if colli_detect:
            grasp_no_collision_view_num = self.valid_grasp  # valid view grasp num
        else:
            grasp_no_collision_view_num = len(self.pred_grasp)
        grasp_no_collision_scene_num = len(
            select_grasp)  # valid scene grasp num
        total_vgr = grasp_no_collision_scene_num
        total_score = torch.sum(self.antipodal_score).item()
        return total_vgr, total_score, select_grasp, grasp_no_collision_view_num, self.pred_grasp, self.grasp_no_collision_scene

    def run_collision_view(self):
        #print('\n Start validate view collision checking \n')
        for frame_index in range(self.frame.shape[0]):
            self.finger_hand_view(frame_index)
        #print('\n Finish view collision checking \n')
        self.pred_grasp = self.pred_grasp[
            self.baseline_frame_index[:self.valid_grasp].long()]

    def run_collision_scene(self, scene: TorchScenePointCloud, colli_detect):
        #print('\n Start validate scene collision checking \n')
        if not colli_detect:
            self.collision_bool = torch.zeros(len(self.pred_grasp),
                                              dtype=torch.uint8).cuda()
            self.antipodal_score = torch.zeros(len(self.pred_grasp),
                                               dtype=torch.float).cuda()
            for i in range(len(self.pred_grasp)):
                self.finger_hand_scene(self.global_to_local[i, :, :], i, scene)
        else:
            self.collision_bool = torch.zeros(self.valid_grasp,
                                              dtype=torch.uint8).cuda()
            self.antipodal_score = torch.zeros(self.valid_grasp,
                                               dtype=torch.float).cuda()
            for i in range(self.valid_grasp):
                self.finger_hand_scene(self.baseline_frame[i, :, :], i, scene)
        #print('\n Finish scene collision checking \n')

    def _table_collision_check(self, point, frame):
        """Check whether the gripper collide with the table top with offset.

        :param point: torch.tensor(3)
        :param frame: torch.tensor(3, 3)
        """
        T_local_to_global = torch.eye(4).to(dtype=torch.float32, device='cuda')
        T_local_to_global[0:3, 0:3] = frame
        T_local_to_global[0:3, 3] = point
        T_local_search_to_global = T_local_to_global.squeeze(0).expand(
            1, 4, 4).contiguous()
        config_gripper = torch.tensor(
            np.array(
                config.TORCH_GRIPPER_BOUND.squeeze(0).expand(
                    1, -1, -1).contiguous())).cuda()
        boundary_global = torch.bmm(T_local_search_to_global, config_gripper)
        table_collision_bool = boundary_global[:,
                                               2, :] < self.table_height - 0.005  #+ config.TABLE_COLLISION_OFFSET
        return table_collision_bool.any(dim=1, keepdim=False)

    def _antipodal_score(self, close_region_cloud, close_region_cloud_normal):
        """Estimate the antipodal score of a single grasp using scene point
        cloud Antipodal score is proportional to the reciprocal of friction
        angle Antipodal score is also divided by the square of objects in the
        closing region.

        :param close_region_cloud: The point cloud in the gripper closing region, torch.tensor (3, n)
        :param close_region_cloud_normal: The point cloud normal in the gripper closing region, torch.tensor (3, n)
        """
        assert close_region_cloud.shape == close_region_cloud_normal.shape, \
            'Points and corresponding normals should have same shape'

        left_y = torch.max(close_region_cloud[1, :])
        right_y = torch.min(close_region_cloud[1, :])
        normal_search_depth = torch.min((left_y - right_y) / 3,
                                        config.NEIGHBOR_DEPTH.cuda())

        left_region_bool = close_region_cloud[
            1, :] > left_y - normal_search_depth
        right_region_bool = close_region_cloud[
            1, :] < right_y + normal_search_depth
        left_normal_theta = torch.abs(
            torch.matmul(self.left_normal,
                         close_region_cloud_normal[:, left_region_bool]))
        right_normal_theta = torch.abs(
            torch.matmul(self.right_normal,
                         close_region_cloud_normal[:, right_region_bool]))
        geometry_average_theta = torch.mean(left_normal_theta) * torch.mean(
            right_normal_theta)
        return geometry_average_theta

    def finger_hand_view(self, frame_index):
        """
        :param frame_index: The index of point in frame list, which is a subset of all point cloud
        """
        frame = self.frame[frame_index, :, :]
        point = self.center[frame_index, :]
        width = self.widths[frame_index]
        depth = float(self.depths[frame_index])

        # not check table
        # if type(self.depth) is float:
        #     if point[2] + frame[
        #             2,
        #             0] * self.depth < self.table_height - 0.005:  # config.FINGER_LENGTH  self.depth
        #         return
        # else:
        #     if point[2] + frame[2, 0] * self.depth[
        #             frame_index] < self.table_height - 0.005:  # config.FINGER_LENGTH  self.depth
        #         return

        # table_collision_bool = self._table_collision_check(point, frame)

        T_global_to_local = self.global_to_local[frame_index, :, :]
        local_cloud = torch.matmul(T_global_to_local, self.cloud_array_homo)

        # seg point between grasp length axis
        if type(depth) is float:
            close_plane_bool = (
                local_cloud[0, :] > -(config.HAND_LENGTH - depth)) & (
                    local_cloud[0, :] < depth)  # config.FINGER_LENGTH
        else:
            close_plane_bool = (local_cloud[0, :] >
                                -(config.HAND_LENGTH - depth)) & (
                                    local_cloud[0, :] < depth[frame_index]
                                )  # config.FINGER_LENGTH
        if torch.sum(close_plane_bool) < config.NUM_POINTS_THRESHOLD:
            return
        local_search_close_plane_points = local_cloud[:, close_plane_bool][
            0:3, :]  # only filter along x axis
        #T_local_to_local_search = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.],
        #                                        [0., 0., 1., 0.], [0., 0., 0., 1.]])
        #local_search_close_plane_points = torch.matmul(T_local_to_local_search.contiguous().view(-1, 4), \
        #                                               close_plane_points).contiguous().view(1, 4, -1)[:, 0:3, :]

        # back collision check
        hand_half_bottom_width = width / 2 + config.FINGER_WIDTH
        hand_half_bottom_space = width / 2
        z_collision_bool = (local_search_close_plane_points[2, :] < config.HALF_HAND_THICKNESS) & \
                            (local_search_close_plane_points[2, :] > -config.HALF_HAND_THICKNESS)
        back_collision_bool = (local_search_close_plane_points[1, :] < hand_half_bottom_width) & \
                                (local_search_close_plane_points[1, :] > -hand_half_bottom_width) & \
                                (local_search_close_plane_points[0, :] < -(config.FINGER_LENGTH - depth)) & \
                                z_collision_bool
        if torch.sum(back_collision_bool) > config.BACK_COLLISION_THRESHOLD:
            return

        # finger collision check (better)
        y_finger_region_bool_left = (local_search_close_plane_points[1, :] < hand_half_bottom_width) & \
                                    (local_search_close_plane_points[1, :] > hand_half_bottom_space)
        y_finger_region_bool_right = (local_search_close_plane_points[1, :] > -hand_half_bottom_width) & \
                                    (local_search_close_plane_points[1, :] < -hand_half_bottom_space)

        y_finger_region_bool = y_finger_region_bool_left | y_finger_region_bool_right
        collision_region_bool = (z_collision_bool & y_finger_region_bool)
        if torch.sum(
                collision_region_bool) > config.FINGER_COLLISION_THRESHOLD:
            return

        # close region points num should be greater than thres (better)
        close_region_bool = z_collision_bool & \
                            (local_search_close_plane_points[1, :] < hand_half_bottom_space) & \
                            (local_search_close_plane_points[1, :] > -hand_half_bottom_space)
        close_region_point_num = torch.sum(close_region_bool)
        if close_region_point_num < config.CLOSE_REGION_MIN_POINTS:
            return

        self.baseline_frame[
            self.valid_grasp] = self.global_to_local[frame_index]
        self.baseline_frame_index[self.valid_grasp] = frame_index
        self.valid_grasp += 1

    def finger_hand_scene(self, T_global_to_local, valid_index,
                          scene: TorchScenePointCloud):
        """Local search one point and store the closing region point num of
        each configurations Search height first, then width, finally theta Save
        the number of points in the close region if the grasp do not fail in
        local search Save the score of antipodal_grasp, note that multi-objects
        heuristic is also stored here."""
        local_cloud = torch.matmul(T_global_to_local, scene.cloud_array_homo)
        local_cloud_normal = torch.matmul(T_global_to_local[0:3, 0:3],
                                          scene.normal_array)
        width = self.widths[valid_index]
        depth = float(self.depths[valid_index])

        # seg point between grasp length axis
        close_plane_bool = (
            local_cloud[0, :] > -(config.HAND_LENGTH - depth)) & (
                local_cloud[0, :] < depth)  # config.FINGER_LENGTH
        if torch.sum(close_plane_bool) < config.NUM_POINTS_THRESHOLD:
            return

        local_search_close_plane_points = local_cloud[:, close_plane_bool][
            0:3, :]  # only filter along x axis

        # back collision check
        hand_half_bottom_width = width / 2 + config.FINGER_WIDTH
        hand_half_bottom_space = width / 2
        z_collision_bool = (local_search_close_plane_points[2, :] < config.HALF_HAND_THICKNESS) & \
                           (local_search_close_plane_points[2, :] > -config.HALF_HAND_THICKNESS)
        back_collision_bool = (local_search_close_plane_points[1, :] < hand_half_bottom_width) & \
                              (local_search_close_plane_points[1, :] > -hand_half_bottom_width) & \
                              (local_search_close_plane_points[0, :] < -(config.FINGER_LENGTH - depth)) & \
                              z_collision_bool
        if torch.sum(back_collision_bool) > config.BACK_COLLISION_THRESHOLD:
            return

        # finger collision check
        y_finger_region_bool_left = (local_search_close_plane_points[1, :] < hand_half_bottom_width) & \
                                    (local_search_close_plane_points[1, :] > hand_half_bottom_space)
        y_finger_region_bool_right = (local_search_close_plane_points[1, :] > -hand_half_bottom_width) & \
                                     (local_search_close_plane_points[1, :] < -hand_half_bottom_space)
        y_finger_region_bool = y_finger_region_bool_left | y_finger_region_bool_right
        collision_region_bool = (z_collision_bool & y_finger_region_bool)
        if torch.sum(
                collision_region_bool) > config.FINGER_COLLISION_THRESHOLD:
            return

        # close region points num should be greater than thres
        close_region_bool = z_collision_bool & \
                            (local_search_close_plane_points[1, :] < hand_half_bottom_space) & \
                            (local_search_close_plane_points[1, :] > -hand_half_bottom_space)
        close_region_point_num = torch.sum(close_region_bool)
        if close_region_point_num < config.CLOSE_REGION_MIN_POINTS:
            return

        self.collision_bool[valid_index] = 1
        close_region_normals = local_cloud_normal[:,
                                                  close_plane_bool][:,
                                                                    close_region_bool]
        close_region_cloud = local_search_close_plane_points[:,
                                                             close_region_bool]

        self.antipodal_score[valid_index] = self._antipodal_score(
            close_region_cloud, close_region_normals)


if __name__ == '__main__':
    print(config.CLOSE_REGION_MIN_POINTS)
