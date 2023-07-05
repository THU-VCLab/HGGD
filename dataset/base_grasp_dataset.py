import os
import random
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
from PIL import Image
from skimage.draw import polygon

from dataset.grasp import RectGraspGroup

from .config import camera


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    y, x = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom,
                               radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def draw_rectangle(heatmap, points):
    rr, cc = polygon(points[:, 0], points[:, 1])
    mask = np.logical_and(rr < heatmap.shape[0], cc < heatmap.shape[1])
    heatmap[rr[mask], cc[mask]] = 1
    return heatmap


noise_scale = 20


class GraspDataset:

    def __init__(self) -> None:
        raise NotImplementedError

    def setaug(self):
        self.is_aug = True

    def unaug(self):
        self.is_aug = False

    def eval(self):
        self.trainning = False

    def train(self):
        self.trainning = True

    def scene_list(self):
        return self.scenename

    def __len__(self):
        return len(self.depthpath)

    def get_depth(self, index, rot=0, zoom=1.0):
        # using pillow-simd to speed up
        depth = Image.open(self.depthpath[index])
        # add noise
        depth = np.array(depth, dtype=np.float32)
        if self.trainning:
            # gaussian noise
            if self.noise > 0:
                depth += np.random.randn(
                    *depth.shape) * np.random.uniform() * self.noise * 1000.0
            # extra aug: random erase
            if self.is_aug:
                erase_cnt = np.random.randint(0, 10)
                for _ in range(erase_cnt):
                    largest_len = noise_scale * 2
                    top = np.random.randint(
                        0, depth.shape[0] - largest_len * 2) + largest_len
                    left = np.random.randint(
                        0, depth.shape[1] - largest_len * 2) + largest_len
                    edge_len = np.random.randint(largest_len // 3, largest_len)
                    depth[top:top + edge_len, left:left + edge_len] = 0
        else:
            # gaussian noise
            if self.noise > 0:
                depth += np.random.randn(*depth.shape) * self.noise * 1000.0
        # save
        self.cur_depth = depth.T.copy()
        # using pillow to rotate and crop
        depth = Image.fromarray(depth)
        if rot != 0:
            depth = depth.rotate(rot * 180 / np.pi)
        if zoom != 1.0:
            sr = int(depth.size[0] * (1 - zoom)) // 2
            sc = int(depth.size[1] * (1 - zoom)) // 2
            depth = depth.crop(
                (sr, sc, depth.size[0] - sr, depth.size[1] - sc))
        # resize
        depth = depth.resize(self.output_size)
        depth = np.array(depth, np.float32) / 1000.0
        depth = np.clip((depth - depth.mean()), -1, 1)
        return depth.T

    def get_rgb(self, index, rot=0, zoom=1.0):
        # using pillow-simd to speed up
        rgb = Image.open(self.colorpath[index])
        # aug
        if self.trainning:
            # jitter and gaussian noise
            if self.aug is not None:
                rgb = self.aug(rgb)
        # save
        self.cur_rgb = np.array(rgb).transpose(2, 1, 0)
        # using pillow to rotate and crop
        if rot != 0:
            rgb = rgb.rotate(rot * 180 / np.pi)
        if zoom != 1.0:
            sr = int(rgb.size[0] * (1 - zoom)) // 2
            sc = int(rgb.size[1] * (1 - zoom)) // 2
            rgb = rgb.crop((sr, sc, rgb.size[0] - sr, rgb.size[1] - sc))
        # resize
        rgb = rgb.resize(self.output_size)
        rgb = np.array(rgb, np.float32) / 255.0
        rgb = rgb.transpose((2, 1, 0))
        return rgb

    def gtbb_processing(self, points, rot=0, zoom=1.0):
        # do scale
        scale = (self.output_size[0] / 1280, self.output_size[1] / 720)
        points = points * scale
        c = (self.output_size[0] / 2, self.output_size[1] / 2)  # (x,y)
        c = np.array(c).reshape((1, 2))
        # 旋转
        R = np.array([
            [np.cos(-rot), np.sin(-rot)],
            [-1 * np.sin(-rot), np.cos(-rot)],
        ])
        points = ((np.dot(R, (points - c).T)).T + c).astype(np.int32)  # (4,2)
        # 缩放
        T = np.array([[1 / zoom, 0], [0, 1 / zoom]])
        points = ((np.dot(T, (points - c).T)).T + c).astype(np.int32)

        return points

    def draw_gaussian(self, rect_points, min_dis=8):
        """Convert GraspRectangles to the Location Map for training.

        :param rect_points: rect_points
        :return: Location Map
        """
        loc_map = np.zeros(self.output_size)
        centers = np.zeros((0, 2))
        for points in rect_points:
            # nms
            cur_center = points.mean(0).astype(np.int32)
            center_dis = np.linalg.norm(centers - cur_center, axis=1)
            if (center_dis >= min_dis).all():
                centers = np.vstack([centers, cur_center])
                loc_map = draw_umich_gaussian(loc_map, cur_center, self.sigma)
        return loc_map

    def get_anchor_map(self,
                       rect_points,
                       rect_gg: RectGraspGroup,
                       ratio=1,
                       anchor_k=6,
                       anchor_w=50.0,
                       anchor_z=20.0):
        x_size = self.output_size[0] // ratio
        y_size = self.output_size[1] // ratio
        map_shape = (anchor_k, x_size, y_size)
        cls_mask_map = np.zeros(map_shape)
        theta_offset_map = np.zeros(map_shape)
        depth_offset_map = np.zeros(map_shape)
        width_offset_map = np.zeros(map_shape)
        for points, rect_g in zip(rect_points, rect_gg):
            # get center and width from rect_points (maybe rotate or zoom)
            center = points.mean(0).astype(np.int32)
            u, v = center // ratio
            if u < 0 or v < 0 or u >= x_size or v >= y_size:
                continue
            # cal depth offset from original depth
            neighbor = 2
            x, y = center * 1280 // self.output_size[0]
            left, right = max(0, x - neighbor), min(1279, x + neighbor)
            top, bottom = max(0, y - neighbor), min(719, y + neighbor)
            d = np.median(self.cur_depth[left:right, top:bottom])
            if d > 0:
                delta_depth = rect_g.depth - d
                # limit delta_depth range
                depth_offset = delta_depth / (anchor_z * 2)
                depth_offset = np.clip(depth_offset, -0.5, 0.5)
            else:
                depth_offset = None
            # get width
            width = np.sqrt(np.sum(np.square(points[0, :] - points[1, :])))
            width_offset = np.log(width / anchor_w)
            # set anchor for theta angle, need to change to 2 * pi / k
            theta_range = np.pi
            anchor_step = theta_range / anchor_k
            # clip to avoid exceeding anchor range
            theta = np.clip(rect_g.theta, -theta_range / 2 + 1e-8,
                            theta_range / 2 - 1e-8)
            g_pos, delta_theta = divmod(theta + theta_range / 2, anchor_step)
            theta_offset = delta_theta / anchor_step - 0.5
            # add, not set
            g_pos = int(g_pos)
            cls_mask_map[g_pos, u, v] += 1
            # attention: using shift mean here
            theta_offset_map[g_pos, u, v] += theta_offset
            width_offset_map[g_pos, u, v] += width_offset
            if depth_offset is not None:
                depth_offset_map[g_pos, u, v] += depth_offset
            elif cls_mask_map[g_pos, u, v] > 1 and depth_offset_map[g_pos, u,
                                                                    v] > 0:
                # maintain average if current depth is None
                depth_offset_map[g_pos, u, v] *= cls_mask_map[g_pos, u, v] / (
                    cls_mask_map[g_pos, u, v] - 1)
        # average for offset
        count_map = cls_mask_map + (cls_mask_map == 0)
        theta_offset_map = theta_offset_map / count_map
        width_offset_map = width_offset_map / count_map  # geometric avg
        depth_offset_map = depth_offset_map / count_map
        # sigmoid for cls mask
        cls_mask_map = 2 / (1 + np.exp(-cls_mask_map)) - 1
        return cls_mask_map, theta_offset_map, depth_offset_map, width_offset_map

    def get_gtbb(self, rect_gg, rot=0, zoom=1.0):
        rect_points = np.zeros((0, 4, 2))
        if rot != 0 or zoom != 1.0:
            for i in range(len(rect_gg)):
                points = self.gtbb_processing(rect_gg[i].points,
                                              rot=rot,
                                              zoom=zoom)
                rect_points = np.vstack([rect_points, points[None]])
        else:
            scale = 1280 / self.output_size[0]
            for i in range(len(rect_gg)):
                rect_points = np.vstack([rect_points, rect_gg[i].points[None]])
            rect_points /= scale
        return rect_points

    @classmethod
    def to_tensor(cls, s):
        if len(s.shape) == 2:
            return torch.from_numpy(s[None].astype(np.float32))
        else:
            return torch.from_numpy(s.astype(np.float32))

    def __getitem__(self, idx):
        # Random rotate
        rot = 0.0
        if self.trainning and self.is_aug and self.random_rotate:
            # img w != h, cannot rotate n / 2 * pi
            rot = random.choice([0, np.pi])

        # Random zoom
        zoom_factor = 1.0
        if self.trainning and self.is_aug and self.random_zoom:
            zoom_factor = np.random.uniform(0.5, 1)
        # Load the depth image
        if self.include_depth:
            norm_depth = self.get_depth(idx, rot, zoom_factor)

        # Load the RGB image
        if self.include_rgb:
            norm_rgb = self.get_rgb(idx, rot, zoom_factor)

        # Load the grasps
        grasp_label = self.load_grasp_labels(idx)

        # zoom and rotate grasp
        rect_gg = RectGraspGroup()
        rect_gg.load_from_dict(grasp_label)
        if self.grasp_count < len(rect_gg):
            idxs = random.sample(range(len(rect_gg)), self.grasp_count)
            rect_gg = rect_gg[idxs]

        rect_points = self.get_gtbb(rect_gg=rect_gg, rot=rot, zoom=zoom_factor)

        # Convert to Gaussian-based Location Map
        loc_map = self.draw_gaussian(rect_points, self.ratio)

        # Create ground truth based on oriented anchor box
        cls_mask, theta_offset, depth_offset, width_offset = self.get_anchor_map(
            rect_points,
            rect_gg,
            self.ratio,
            self.anchor_k,
            anchor_w=self.anchor_w,
            anchor_z=self.anchor_z)

        if self.include_depth and self.include_rgb:
            x = np.vstack([norm_depth[None], norm_rgb])
        elif self.include_depth:
            x = norm_depth[None]
        elif self.include_rgb:
            x = norm_rgb

        # conver to tensor
        x = self.to_tensor(x)
        loc_map_torch = self.to_tensor(loc_map)
        cls_mask_torch = self.to_tensor(cls_mask)
        theta_offset_torch = self.to_tensor(theta_offset)
        depth_offset_torch = self.to_tensor(depth_offset)
        width_offset_torch = self.to_tensor(width_offset)
        return x, (loc_map_torch, cls_mask_torch, theta_offset_torch, depth_offset_torch,
                   width_offset_torch), \
               idx, rot, zoom_factor

    def load_grasp_labels(self, index):
        if index == -1:  # load all grasp labels
            grasp_labels = []
            for g_path in self.grasppath:
                label = np.load(g_path)
                grasp_labels.append(label)
            return grasp_labels
        else:
            return np.load(self.grasppath[index])

    def loadGrasp(self, sceneId, annId=0):
        path = os.path.join(self.labelroot, '6d_dataset',
                            'scene_{}'.format(sceneId), 'grasp_labels',
                            '{}_view.npz'.format(annId))
        grasp_labels = np.load(path)
        rect_grasp_group = RectGraspGroup()
        rect_grasp_group.centers = grasp_labels['centers_2d']
        rect_grasp_group.thetas = grasp_labels['thetas_rad']
        rect_grasp_group.gammas = grasp_labels['gammas_rad']
        rect_grasp_group.betas = grasp_labels['betas_rad']
        rect_grasp_group.widths = grasp_labels['widths_2d']
        rect_grasp_group.heights = grasp_labels['heights_2d']
        rect_grasp_group.scores = grasp_labels['scores_from_6d']
        rect_grasp_group.object_ids = grasp_labels['object_ids']
        rect_grasp_group.depths = grasp_labels['center_z_depths']
        return rect_grasp_group

    def loadRGB(self, sceneId, annId=0):
        return cv2.cvtColor(
            cv2.imread(
                os.path.join(self.graspnetroot, 'scenes',
                             'scene_%04d' % sceneId, camera, 'rgb',
                             '%04d.png' % annId)), cv2.COLOR_BGR2RGB)

    def loadDepth(self, sceneId, annId=0):
        return cv2.imread(
            os.path.join(self.graspnetroot, 'scenes', 'scene_%04d' % sceneId,
                         camera, 'depth', '%04d.png' % annId),
            cv2.IMREAD_UNCHANGED)

    def loadScenePointCloud(self, sceneId, annId=0, format='open3d'):
        colors = self.loadRGB(sceneId=sceneId, annId=annId).astype(
            np.float32) / 255.0

        depths = self.loadDepth(sceneId=sceneId, annId=annId)
        intrinsics = np.load(
            os.path.join(self.graspnetroot, 'scenes', 'scene_%04d' % sceneId,
                         camera, 'camK.npy'))
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        s = 1000.0
        xmap, ymap = np.arange(colors.shape[1]), np.arange(colors.shape[0])
        xmap, ymap = np.meshgrid(xmap, ymap)
        points_z = depths / s
        points_x = (xmap - cx) / fx * points_z
        points_y = (ymap - cy) / fy * points_z

        mask = (points_z > 0)
        points = np.stack([points_x, points_y, points_z], axis=-1)
        points = points[mask]
        colors = colors[mask]

        if format == 'open3d':
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(points)
            cloud.colors = o3d.utility.Vector3dVector(colors)
            return cloud
        elif format == 'numpy':
            return points, colors
        else:
            raise ValueError('Format must be either "open3d" or "numpy".')
