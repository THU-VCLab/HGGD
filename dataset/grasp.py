import copy
import math
import os
import sys
from time import time

import cv2
import numpy as np
import open3d as o3d
from skimage.draw import polygon

from customgraspnetAPI.grasp import Grasp as GraspNetGrasp
from customgraspnetAPI.grasp import GraspGroup as GraspNetGraspGroup

from .config import get_camera_intrinsic
from .graspnet_utils import (get_2d_key_points, get_6d_key_points,
                             key_point_2_rotation, pixel_depth_2_points,
                             points_2_2d_tuple, points_2_pixel_depth)


class Grasp():

    def __init__(self,
                 translation=np.zeros((3, )),
                 rotation=np.eye(3),
                 height=0.02,
                 width=0.085,
                 depth=0.02,
                 score=1,
                 object_id=0):
        self.translation = translation
        self.rotation = rotation
        self.height = height
        self.width = width
        self.depth = depth
        self.score = score
        self.object_id = object_id

    def __repr__(self):
        lines = []
        lines.append(
            f'Grasp: score:{self.score}, width:{self.width}, height:{self.height}, depth:{self.depth}'
        )
        lines.append(f'translation:{self.translation}')
        lines.append(f'rotation:\n{self.rotation}')
        return '\n'.join(lines)

    def to_our_gripper(self, color=None):
        # default gripper setting
        finger_width = 0.004
        tail_length = 0.04
        depth_base = 0.02

        def create_mesh_box(width, height, depth):
            ''' Author: wei-tang
            Create box instance with mesh representation.
            '''
            box = o3d.geometry.TriangleMesh()
            vertices = np.array([[0, 0, 0], [0, 0, width], [0, depth, 0],
                                 [0, depth, width], [height, 0, 0],
                                 [height, 0, width], [height, depth, 0],
                                 [height, depth, width]])
            triangles = np.array([[4, 7, 5], [4, 6, 7], [0, 2, 4], [2, 6, 4],
                                  [0, 1, 2], [1, 3, 2], [1, 5, 7], [1, 7, 3],
                                  [2, 3, 7], [2, 7, 6], [0, 4, 1], [1, 4, 5]])
            box.vertices = o3d.utility.Vector3dVector(vertices)
            box.triangles = o3d.utility.Vector3iVector(triangles)
            return box

        left = create_mesh_box(depth_base + finger_width, finger_width,
                               self.height)
        right = create_mesh_box(depth_base + finger_width, finger_width,
                                self.height)
        bottom = create_mesh_box(finger_width, self.width, self.height)
        tail = create_mesh_box(tail_length, finger_width, self.height)

        left_points = np.array(left.vertices)
        left_triangles = np.array(left.triangles)
        left_points[:, 0] -= self.width / 2 + finger_width
        left_points[:, 1] -= self.height / 2
        left_points[:, 2] -= depth_base + finger_width

        right_points = np.array(right.vertices)
        right_triangles = np.array(right.triangles) + 8
        right_points[:, 0] += self.width / 2
        right_points[:, 1] -= self.height / 2
        right_points[:, 2] -= depth_base + finger_width

        bottom_points = np.array(bottom.vertices)
        bottom_triangles = np.array(bottom.triangles) + 16
        bottom_points[:, 0] -= self.width / 2
        bottom_points[:, 1] -= self.height / 2
        bottom_points[:, 2] -= finger_width + depth_base

        tail_points = np.array(tail.vertices)
        tail_triangles = np.array(tail.triangles) + 24
        tail_points[:, 0] -= finger_width / 2
        tail_points[:, 1] -= self.height / 2
        tail_points[:, 2] -= tail_length + finger_width + depth_base

        vertices = np.concatenate(
            [left_points, right_points, bottom_points, tail_points], axis=0)
        # transform
        vertices = vertices @ self.rotation.T + self.translation
        triangles = np.concatenate([
            left_triangles, right_triangles, bottom_triangles, tail_triangles
        ],
                                   axis=0)

        # red for high score
        colors = np.array([[self.score, 0, 1 - self.score]])
        if color is not None:
            colors = np.array([color])
        colors = np.repeat(colors, len(vertices), axis=0)

        gripper = o3d.geometry.TriangleMesh()
        gripper.vertices = o3d.utility.Vector3dVector(vertices)
        gripper.triangles = o3d.utility.Vector3iVector(triangles)
        gripper.vertex_colors = o3d.utility.Vector3dVector(colors)
        return gripper

    def to_open3d_geometry(self, color=None):
        # default gripper setting
        finger_width = 0.004
        tail_length = 0.04
        depth_base = 0.02
        self.height = 0.002

        def create_mesh_box(width, height, depth):
            box = o3d.geometry.TriangleMesh()
            # set vertices
            vertices = np.array([[0, 0, 0], [width, 0, 0], [0, 0, depth],
                                 [width, 0, depth], [0, height, 0],
                                 [width, height, 0], [0, height, depth],
                                 [width, height, depth]])
            # set triangles according to certices ids
            triangles = np.array([[4, 7, 5], [4, 6, 7], [0, 2, 4], [2, 6, 4],
                                  [0, 1, 2], [1, 3, 2], [1, 5, 7], [1, 7, 3],
                                  [2, 3, 7], [2, 7, 6], [0, 4, 1], [1, 4, 5]])
            box.vertices = o3d.utility.Vector3dVector(vertices)
            box.triangles = o3d.utility.Vector3iVector(triangles)
            return box

        # TODO: check depth here
        left = create_mesh_box(self.depth + depth_base + finger_width,
                               finger_width, self.height)
        right = create_mesh_box(self.depth + depth_base + finger_width,
                                finger_width, self.height)
        bottom = create_mesh_box(finger_width, self.width, self.height)
        tail = create_mesh_box(tail_length, finger_width, self.height)

        left_points = np.array(left.vertices)
        left_triangles = np.array(left.triangles)
        left_points[:, 0] -= depth_base + finger_width
        left_points[:, 1] -= self.width / 2 + finger_width
        left_points[:, 2] -= self.height / 2

        right_points = np.array(right.vertices)
        right_triangles = np.array(right.triangles) + 8
        right_points[:, 0] -= depth_base + finger_width
        right_points[:, 1] += self.width / 2
        right_points[:, 2] -= self.height / 2

        bottom_points = np.array(bottom.vertices)
        bottom_triangles = np.array(bottom.triangles) + 16
        bottom_points[:, 0] -= finger_width + depth_base
        bottom_points[:, 1] -= self.width / 2
        bottom_points[:, 2] -= self.height / 2

        tail_points = np.array(tail.vertices)
        tail_triangles = np.array(tail.triangles) + 24
        tail_points[:, 0] -= tail_length + finger_width + depth_base
        tail_points[:, 1] -= finger_width / 2
        tail_points[:, 2] -= self.height / 2

        vertices = np.concatenate(
            [left_points, right_points, bottom_points, tail_points], axis=0)
        # transform
        vertices = vertices @ self.rotation.T + self.translation
        triangles = np.concatenate([
            left_triangles, right_triangles, bottom_triangles, tail_triangles
        ],
                                   axis=0)

        # red for high score
        colors = np.array([[self.score, 0, 1 - self.score]])
        if color is not None:
            colors = np.array([color])
        colors = np.repeat(colors, len(vertices), axis=0)

        gripper = o3d.geometry.TriangleMesh()
        gripper.vertices = o3d.utility.Vector3dVector(vertices)
        gripper.triangles = o3d.utility.Vector3iVector(triangles)
        gripper.vertex_colors = o3d.utility.Vector3dVector(colors)
        return gripper


class GraspGroup():
    attr_list = [
        'translations', 'rotations', 'heights', 'widths', 'depths', 'scores',
        'object_ids'
    ]

    def __init__(self,
                 translations=None,
                 rotations=None,
                 heights=None,
                 widths=None,
                 depths=None,
                 scores=None,
                 object_ids=None):
        self.translations = np.zeros(
            (0, 3)) if translations is None else translations
        self.rotations = np.zeros(
            (self.size, 3, 3)) if rotations is None else rotations
        self.heights = np.zeros((self.size, )) if heights is None else heights
        self.widths = np.zeros((self.size, )) if widths is None else widths
        self.depths = np.zeros((self.size, )) if depths is None else depths
        self.scores = np.zeros((self.size, )) if scores is None else scores
        self.object_ids = -1 * np.ones(
            (self.size, )) if object_ids is None else object_ids
        for attr in self.attr_list:
            assert len(getattr(self, attr)) == self.size

    def __len__(self):
        return self.size

    @property
    def size(self):
        return self.translations.shape[0]

    @classmethod
    def from_list(cls, grasp_list):
        group_size = len(grasp_list)
        translations = np.zeros((group_size, 3))
        rotations = np.zeros((group_size, 3, 3))
        heights = np.zeros((group_size, ))
        widths = np.zeros((group_size, ))
        depths = np.zeros((group_size, ))
        scores = np.zeros((group_size, ))
        object_ids = np.zeros((group_size, ))
        for i, grasp in enumerate(grasp_list):
            translations[i] = grasp.translation
            rotations[i] = grasp.rotation
            heights[i] = grasp.height
            widths[i] = grasp.width
            depths[i] = grasp.depth
            scores[i] = grasp.score
            object_ids[i] = grasp.object_id
        return cls(
            translations,
            rotations,
            heights,
            widths,
            depths,
            scores,
            object_ids,
        )

    def __getitem__(self, index):
        # will generate a new Grasp or GraspGroup
        if type(index) == int:
            return Grasp(translation=self.translations[index],
                         rotation=self.rotations[index],
                         height=self.heights[index],
                         width=self.widths[index],
                         depth=self.depths[index],
                         score=self.scores[index],
                         object_id=self.object_ids[index])
        elif type(index) == slice or type(index) == list or type(
                index) == np.ndarray:
            return GraspGroup(translations=self.translations[index],
                              rotations=self.rotations[index],
                              heights=self.heights[index],
                              widths=self.widths[index],
                              depths=self.depths[index],
                              scores=self.scores[index],
                              object_ids=self.object_ids[index])
        else:
            raise TypeError(
                'unknown type "{}" for calling __getitem__ for GraspGroup'.
                format(type(index)))

    def __repr__(self):
        lines = []
        for i in range(self.size):
            line_str = self[i].__repr__()
            line_str = line_str[:5] + f' {i}' + line_str[5:]
            lines.append(line_str)
        return '\n'.join(lines)

    def append(self, grasp: Grasp):
        self.translations = np.append(self.translations,
                                      grasp.translation[np.newaxis, :],
                                      axis=0)
        self.rotations = np.append(self.rotations,
                                   grasp.rotation[np.newaxis, :],
                                   axis=0)
        self.heights = np.append(self.heights, grasp.height)
        self.widths = np.append(self.widths, grasp.width)
        self.depths = np.append(self.depths, grasp.depth)
        self.scores = np.append(self.scores, grasp.score)
        self.object_ids = np.append(self.object_ids, grasp.object_id)

    def to_rect_grasp_group(self, threshold=0, donms=False):

        mask = (self.rotations[:, 2, 0] > 0)
        _6d_grasp_group_mask = self[mask]

        ## 在平移之前记录下centers
        k_points_6d = get_6d_key_points(_6d_grasp_group_mask.translations,
                                        _6d_grasp_group_mask.rotations,
                                        _6d_grasp_group_mask.widths)
        k_points_6d = k_points_6d.reshape([-1, 3])
        k_points_6d = k_points_6d.reshape([-1, 4, 3])
        centers_2d, center_z_depths = points_2_pixel_depth(k_points_6d[:,
                                                                       0, :])

        ## 将抓取平移到[0, 0, center_z_depths]再进行转换
        _6d_grasp_group_mask.translations = np.zeros(
            (len(_6d_grasp_group_mask), 3))
        _6d_grasp_group_mask.translations[:, 2] = center_z_depths / 1000
        # translations = copy.deepcopy(_6d_grasp_group_mask.translations)

        k_points_6d_trans = get_6d_key_points(
            _6d_grasp_group_mask.translations, _6d_grasp_group_mask.rotations,
            _6d_grasp_group_mask.widths)
        k_points_6d_trans = k_points_6d_trans.reshape([-1, 3])
        k_points_6d_trans = k_points_6d_trans.reshape([-1, 4, 3])

        # 因为后面都是计算角度，所以用平移后的k_points进行计算
        centers, opens, heights = points_2_2d_tuple(k_points_6d_trans)
        rect_grasp_group = RectGraspGroup(
            centers=centers,
            # opens=opens,
            heights=heights,
            scores=_6d_grasp_group_mask.scores,
            object_ids=_6d_grasp_group_mask.object_ids,
            depths=center_z_depths,
        )

        dy = rect_grasp_group.centers[:, 1] - opens[:, 1]
        dx = rect_grasp_group.centers[:, 0] - opens[:, 0]
        thetas_rad = np.arctan2(dy, -dx)

        gammas_rad = np.zeros((len(rect_grasp_group), ))
        gammas_ang = np.zeros((len(rect_grasp_group), ))
        betas_rad = np.zeros((len(rect_grasp_group), ))
        betas_ang = np.zeros((len(rect_grasp_group), ))

        for i in range(len(rect_grasp_group)):
            tmp_gamma = (k_points_6d_trans[i][1][2] -
                         k_points_6d_trans[i][0][2]
                         ) * 2 / _6d_grasp_group_mask.widths[i]
            if abs(tmp_gamma) > 1:
                print(f'gamma == {tmp_gamma} 越界')
            tmp_gamma = np.clip(tmp_gamma, -1, 1)
            gammas_rad[i] = math.asin(tmp_gamma)
            gammas_ang[i] = gammas_rad[i] * 180 / np.pi
            tmp_beta = (k_points_6d_trans[i][2][2] -
                        k_points_6d_trans[i][3][2]) / math.cos(
                            gammas_rad[i]) / _6d_grasp_group_mask.heights[i]
            if abs(tmp_beta) > 1:
                print(f'beta == {tmp_beta} 越界')
            tmp_beta = np.clip(tmp_beta, -1, 1)
            betas_rad[i] = math.asin(tmp_beta)
            betas_ang[i] = betas_rad[i] * 180 / np.pi

        ## 在平移之后计算widths_2d
        widths_2d = np.sqrt(
            np.square(rect_grasp_group.centers[:, 0] - opens[:, 0]) +
            np.square(rect_grasp_group.centers[:, 1] - opens[:, 1])) * 2

        # for 6d dataset
        widths_2d /= np.cos(gammas_rad)
        rect_grasp_group.heights *= 2 / (
            1 + np.sqrt(1 - np.square(np.cos(gammas_rad) * np.sin(betas_rad))))

        ## 使用center_points、theta_rad和widths_2d_trans计算2D矩形框的open_points
        opens_x = centers_2d[:, 0] + widths_2d / 2 * np.cos(thetas_rad)
        opens_y = centers_2d[:, 1] - widths_2d / 2 * np.sin(thetas_rad)
        opens_2d = np.stack([opens_x, opens_y], axis=-1)

        rect_grasp_group.centers = centers_2d
        # rect_grasp_group.opens = opens_2d
        rect_grasp_group.thetas = thetas_rad
        rect_grasp_group.gammas = gammas_rad
        rect_grasp_group.betas = betas_rad
        rect_grasp_group.widths = widths_2d

        return rect_grasp_group

    def to_our_gripper_list(self, color=None):
        gripper_list = []
        for i in range(self.size):
            gripper_list.append(self[i].to_our_gripper(color))
        return gripper_list

    def to_open3d_geometry_list(self, color=None):
        gripper_list = []
        for i in range(self.size):
            fixed_score = self.scores[i]
            color = [fixed_score, 0, 1 - fixed_score]
            gripper_list.append(self[i].to_open3d_geometry(color=color))
        return gripper_list

    def sort(self):
        return GraspGroup.from_list(
            sorted(self, key=lambda x: x.score, reverse=True))

    def nms(self):
        # use graspnetAPI to nms
        gg = GraspNetGraspGroup()
        for g in self:
            g = GraspNetGrasp(
                g.score,
                g.width,
                g.height,
                g.depth,
                g.rotation.reshape((9, )),
                g.translation,
                -1,
            )
            gg.add(g)
        gg = gg.nms(0.03, 60 / 180 * np.pi)
        return GraspGroup(translations=gg.translations,
                          rotations=gg.rotation_matrices,
                          heights=gg.heights,
                          widths=gg.widths,
                          depths=gg.depths,
                          scores=gg.scores)


class RectGrasp():

    def __init__(self,
                 center=np.zeros((2, )),
                 depth=0,
                 height=20,
                 width=20,
                 score=1,
                 theta=0,
                 gamma=0,
                 beta=0,
                 object_id=0):
        self.center = center
        self.height = height
        self.width = width
        self.depth = depth
        self.score = score
        self.theta = theta
        self.gamma = gamma
        self.beta = beta
        self.object_id = object_id

    def __repr__(self):
        lines = []
        lines.append(
            f'RectGrasp: score:{self.score}, width:{self.width}, height:{self.height}, depth:{self.depth}'
        )
        lines.append(f'center:{self.center}')
        lines.append(
            f'theta:{self.theta / np.pi * 180}, beta:{self.beta / np.pi * 180}, gamma:{self.gamma / np.pi * 180}'
        )
        return '\n'.join(lines)

    @classmethod
    def from_bb(cls, points):
        center = points.mean(axis=0).astype(np.int32)
        dx = points[1, 0] - points[0, 0]
        dy = points[1, 1] - points[0, 1]
        theta = np.arctan2(-dy, dx)
        width = np.sqrt(np.sum(np.square(points[0, :] - points[1, :])))
        height = np.sqrt(np.sum(np.square(points[2, :] - points[1, :])))
        return cls(center=center, theta=theta, width=width, height=height)

    @property
    def points(self):
        axis = self.width / 2 * np.array(
            [np.cos(self.theta), -np.sin(self.theta)])
        normal = np.array([-axis[1], axis[0]])
        normal = normal / np.linalg.norm(normal) * self.height / 2
        p1 = self.center + normal + axis
        p2 = self.center + normal - axis
        p3 = self.center - normal - axis
        p4 = self.center - normal + axis
        points = np.array([p1, p2, p3, p4], dtype=np.int32)  # (4,2)
        return points

    def plot(self, ax, color=None):
        points = self.points
        points = np.vstack((points, points[0]))
        ax.plot(points[:, 1], points[:, 0], color=color)

    def iou(self, rect_grasp, angle_threshold=np.pi / 6):
        angle_dis = np.abs(self.theta - rect_grasp.theta)
        if angle_dis >= np.pi / 2:
            angle_dis = np.pi - angle_dis
        if angle_dis > angle_threshold:
            return 0

        rr1, cc1 = polygon(self.points[:, 0], self.points[:, 1])
        rr2, cc2 = polygon(rect_grasp.points[:, 0], rect_grasp.points[:, 1])

        try:
            r_max = max(rr1.max(), rr2.max()) + 1
            c_max = max(cc1.max(), cc2.max()) + 1
        except:
            return 0

        canvas = np.zeros((r_max, c_max))
        canvas[rr1, cc1] += 1
        canvas[rr2, cc2] += 1
        union = np.sum(canvas > 0)
        if union == 0:
            return 0
        intersection = np.sum(canvas == 2)
        return intersection / union


class RectGraspGroup():
    attr_list = [
        'centers', 'heights', 'widths', 'depths', 'scores', 'thetas', 'betas',
        'gammas', 'object_ids'
    ]

    def __init__(self,
                 centers=None,
                 heights=None,
                 widths=None,
                 depths=None,
                 scores=None,
                 thetas=None,
                 betas=None,
                 gammas=None,
                 object_ids=None):
        self.centers = np.zeros((0, 2)) if centers is None else centers
        self.heights = np.zeros((self.size, )) if heights is None else heights
        self.widths = np.zeros((self.size, )) if widths is None else widths
        self.depths = np.zeros((self.size, )) if depths is None else depths
        self.scores = np.zeros((self.size, )) if scores is None else scores
        self.thetas = np.zeros((self.size, )) if thetas is None else thetas
        self.betas = np.zeros((self.size, )) if betas is None else betas
        self.gammas = np.zeros((self.size, )) if gammas is None else gammas
        self.object_ids = -1 * np.ones(
            (self.size, )) if object_ids is None else object_ids
        for attr in self.attr_list:
            assert len(getattr(self, attr)) == self.size

    def __len__(self):
        return self.size

    @property
    def size(self):
        return self.centers.shape[0]

    @classmethod
    def from_list(cls, grasp_list):
        group_size = len(grasp_list)
        centers = np.zeros((group_size, 2))
        heights = np.zeros((group_size, ))
        widths = np.zeros((group_size, ))
        thetas = np.zeros((group_size, ))
        depths = np.zeros((group_size, ))
        scores = np.zeros((group_size, ))
        betas = np.zeros((group_size, ))
        gammas = np.zeros((group_size, ))
        object_ids = np.zeros((group_size, ))
        for i, grasp in enumerate(grasp_list):
            centers[i] = grasp.center
            heights[i] = grasp.height
            widths[i] = grasp.width
            thetas[i] = grasp.theta
            depths[i] = grasp.depth
            scores[i] = grasp.score
            betas[i] = grasp.beta
            gammas[i] = grasp.gamma
            object_ids[i] = grasp.object_id
        return cls(
            centers,
            heights,
            widths,
            depths,
            scores,
            thetas,
            betas,
            gammas,
            object_ids,
        )

    def __getitem__(self, index):
        # will generate a new Grasp or GraspGroup
        if type(index) == int:
            return RectGrasp(center=self.centers[index],
                             height=self.heights[index],
                             width=self.widths[index],
                             depth=self.depths[index],
                             score=self.scores[index],
                             theta=self.thetas[index],
                             gamma=self.gammas[index],
                             beta=self.betas[index],
                             object_id=self.object_ids[index])
        elif type(index) in [slice, list, np.ndarray]:
            return RectGraspGroup(centers=self.centers[index],
                                  heights=self.heights[index],
                                  widths=self.widths[index],
                                  depths=self.depths[index],
                                  scores=self.scores[index],
                                  thetas=self.thetas[index],
                                  gammas=self.gammas[index],
                                  betas=self.betas[index],
                                  object_ids=self.object_ids[index])
        else:
            raise TypeError(
                'unknown type "{}" for calling __getitem__ for RectGraspGroup'.
                format(type(index)))

    def __repr__(self):
        lines = []
        for i in range(self.size):
            line_str = self[i].__repr__()
            line_str = line_str[:9] + f' {i}' + line_str[9:]
            lines.append(line_str)
        return '\n'.join(lines)

    def load_from_dict(self, grasp_dict, min_width=0, min_score=0.8):
        # get mask
        width_mask = (grasp_dict['widths_2d'] >= min_width)
        score_mask = (grasp_dict['scores_from_6d'] >= min_score)
        mask = np.logical_and(width_mask, score_mask)
        # filter
        self.centers = grasp_dict['centers_2d'][mask]
        self.thetas = grasp_dict['thetas_rad'][mask]
        self.gammas = grasp_dict['gammas_rad'][mask]
        self.betas = grasp_dict['betas_rad'][mask]
        self.widths = grasp_dict['widths_2d'][mask]
        self.heights = grasp_dict['heights_2d'][mask]
        self.scores = grasp_dict['scores_from_6d'][mask]
        self.object_ids = grasp_dict['object_ids'][mask]
        self.depths = grasp_dict['center_z_depths'][mask]
        # -pi / 2 ~ pi / 2, symmetry
        theta_mask = np.logical_or(self.thetas > np.pi / 2,
                                   self.thetas < -np.pi / 2)
        self.thetas[theta_mask] = (self.thetas[theta_mask] +
                                   np.pi / 2) % np.pi - np.pi / 2
        self.gammas[theta_mask] = -self.gammas[theta_mask]
        self.betas[theta_mask] = -self.betas[theta_mask]

    def append(self, grasp: RectGrasp):
        self.centers = np.append(self.centers,
                                 grasp.center[np.newaxis, :],
                                 axis=0)
        self.heights = np.append(self.heights, grasp.height)
        self.widths = np.append(self.widths, grasp.width)
        self.depths = np.append(self.depths, grasp.depth)
        self.scores = np.append(self.scores, grasp.score)
        self.thetas = np.append(self.thetas, grasp.theta)
        self.betas = np.append(self.betas, grasp.beta)
        self.gammas = np.append(self.gammas, grasp.gamma)
        self.object_ids = np.append(self.object_ids, grasp.object_id)

    def get_6d_width(self):
        if not hasattr(self, 'actual_depths'):
            raise RuntimeError('Grasp actual_depths are not set yet!')
        # avoid self.centers change
        centers_2d = self.centers.copy()
        # centers和opens平移到图片中心
        intrinsics = get_camera_intrinsic()
        centers_2d[:, 0] = intrinsics[0, 2]
        centers_2d[:, 1] = intrinsics[1, 2]
        opens_x = centers_2d[:, 0] - self.widths / 2 * np.cos(self.thetas)
        opens_y = centers_2d[:, 1] + self.widths / 2 * np.sin(self.thetas)
        opens_2d = np.stack([opens_x, opens_y], axis=-1)
        centers_xyz = pixel_depth_2_points(centers_2d[:, 0], centers_2d[:, 1],
                                           self.actual_depths / 1000)
        opens_xyz = pixel_depth_2_points(opens_2d[:, 0], opens_2d[:, 1],
                                         self.actual_depths / 1000)
        widths = (np.linalg.norm(opens_xyz - centers_xyz, axis=1) * 2).reshape(
            (-1, ))
        return widths

    def to_6d_grasp_group(self, depth=0.025):
        # avoid self.centers change
        centers_2d = self.centers.copy()
        # record translations
        translations = pixel_depth_2_points(
            centers_2d[:, 0],
            centers_2d[:,
                       1],  # attention: exchange xy to make sure (1280, 720)
            self.depths / 1000)  # translation: (len(grasp), 3)

        # centers和opens平移到图片中心
        intrinsics = get_camera_intrinsic()
        centers_2d[:, 0] = intrinsics[0, 2]
        centers_2d[:, 1] = intrinsics[1, 2]

        opens_x = centers_2d[:, 0] - self.widths / 2 * np.cos(self.thetas)
        opens_y = centers_2d[:, 1] + self.widths / 2 * np.sin(self.thetas)
        opens_2d = np.stack([opens_x, opens_y], axis=-1)

        upper_points = get_2d_key_points(centers_2d,
                                         self.heights.reshape((-1, 1)),
                                         opens_2d)
        centers_xyz = pixel_depth_2_points(centers_2d[:, 0], centers_2d[:, 1],
                                           self.depths / 1000)
        opens_xyz = pixel_depth_2_points(opens_2d[:, 0], opens_2d[:, 1],
                                         self.depths / 1000)
        uppers_xyz = pixel_depth_2_points(upper_points[:, 0], upper_points[:,
                                                                           1],
                                          self.depths / 1000)

        depths = depth * np.ones((len(self), ))
        heights = (np.linalg.norm(uppers_xyz - centers_xyz, axis=1) *
                   2).reshape((-1, ))
        widths = (np.linalg.norm(opens_xyz - centers_xyz, axis=1) * 2).reshape(
            (-1, ))
        scores = (self.scores).reshape((-1, ))
        object_ids = (self.object_ids).reshape((-1, ))
        rotations = key_point_2_rotation(centers_xyz, opens_xyz,
                                         uppers_xyz).reshape((-1, 3, 3))

        # 绕夹具本身坐标系z轴，旋转gamma角，右乘！！！
        R = np.zeros((rotations.shape[0], 3, 3))
        R[:, 0, 0] = np.cos(self.gammas)
        R[:, 0, 1] = -np.sin(self.gammas)
        R[:, 1, 0] = np.sin(self.gammas)
        R[:, 1, 1] = np.cos(self.gammas)
        R[:, 2, 2] = 1
        rotations = np.einsum('ijk,ikN->ijN', rotations, R)
        # 绕夹具本身坐标系y轴，旋转beta角, 右乘！！！
        R = np.zeros((rotations.shape[0], 3, 3))
        R[:, 0, 0] = np.cos(self.betas)
        R[:, 0, 2] = np.sin(self.betas)
        R[:, 1, 1] = 1
        R[:, 2, 0] = -np.sin(self.betas)
        R[:, 2, 2] = np.cos(self.betas)
        rotations = np.einsum('ijk,ikN->ijN', rotations, R)

        gg_from_rect = GraspGroup()
        gg_from_rect.translations = translations
        gg_from_rect.rotations = rotations
        gg_from_rect.heights = heights
        gg_from_rect.widths = widths
        gg_from_rect.depths = depths
        gg_from_rect.scores = scores
        gg_from_rect.object_ids = object_ids

        return gg_from_rect

    def plot_rect_grasp_group(self, rgb, numGrasp, shuffle=True) -> np.ndarray:
        img = rgb.copy()
        if numGrasp == 0:
            numGrasp = len(self)
        if shuffle:
            startIndex = np.random.randint(low=0,
                                           high=max(1,
                                                    len(self) - numGrasp))
        else:
            startIndex = 0
        for i in range(startIndex, startIndex + numGrasp):
            # need to flip for cv2
            points = self[i].points
            p1, p2, p3, p4 = tuple(points[0, :]), tuple(points[1, :]), tuple(
                points[2, :]), tuple(points[3, :])
            p14 = ((points[0, :] + points[3, :]) / 2)
            p23 = ((points[1, :] + points[2, :]) / 2)
            if (p23 < 0).any() or (p14 < 0).any():
                continue
            p14, p23 = tuple(p14.astype('uint16')), tuple(p23.astype('uint16'))
            cv2.line(img, p14, p23, (0, 0, 255), 3, 8)
            cv2.line(img, p2, p3, (255, 0, 0), 2, 8)
            # cv2.line(img, p3, p4, (0, 0, 255), 1, 8)
            cv2.line(img, p4, p1, (255, 0, 0), 2, 8)
        return img

    def get_grasp_points(self):
        intrinsics = get_camera_intrinsic()
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]

        z = self.depths / 1000  # m
        x = z / fx * (self.centers[:, 0] - cx)  # m
        y = z / fy * (self.centers[:, 1] - cy)  # m
        return np.array([x, y, z]).T

    def sort(self):
        return RectGraspGroup.from_list(
            sorted(self, key=lambda x: x.score, reverse=True))


if __name__ == '__main__':
    import sys
    sys.path.append('../')
    from customgraspnetAPI import GraspNet
    graspnet_root = '/data1/dataset/graspnet'  # ROOT PATH FOR GRASPNET
    dataset = GraspNet(graspnet_root, split='train')
    gg = dataset.loadGrasp(sceneId=0,
                           annId=0,
                           fric_coef_thresh=0.2,
                           format='6d')

    mask = (gg.rotation_matrices[:, 2, 0] > 0)
    gg = gg[mask][:50]
    our_gg = GraspGroup(translations=gg.translations,
                        rotations=gg.rotation_matrices,
                        widths=np.ones(len(gg)) * 0.085,
                        heights=np.ones(len(gg)) * 0.02)

    rect_gg = our_gg.to_rect_grasp_group()

    gg_from_rect = rect_gg.to_6d_grasp_group()
    translation_d = np.sqrt(
        np.mean(np.square(our_gg.translations - gg_from_rect.translations)))
    rotation_d = np.sqrt(
        np.mean(np.square(our_gg.rotations - gg_from_rect.rotations)))
    print(translation_d)
    print(rotation_d)
