from functools import wraps
from time import time
from typing import List

import numpy as np
import torch
import torch.nn.functional as nnf
from numba import njit
from pytorch3d.ops import ball_query, knn_points, sample_farthest_points
from pytorch3d.ops.utils import masked_gather

from .config import get_camera_intrinsic
from .utils import convert_2d_to_3d, euclid_distance


def timing(f):

    @wraps(f)
    def wrap(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print(f'func:{f.__name__} took: {te-ts} sec')
        return result

    return wrap


def feature_fusion(points, perpoint_features, xyzs, mode='knn'):
    # points: [B, N, 3]  perpoint_features: [B, C, H, W]  xyzs: [B, 3, H, W]
    perpoint_features = torch.concat([xyzs, perpoint_features], 1)
    B, C = points.shape[0], perpoint_features.shape[1]
    perpoint_features = perpoint_features.reshape((B, C, -1)).transpose(1, 2)
    # knn neigbor selection
    if mode == 'knn':
        _, nn_idxs, _ = knn_points(points[..., :3],
                                   perpoint_features[..., :3],
                                   K=8)
    else:
        _, nn_idxs, _ = ball_query(points[..., :3],
                                   perpoint_features[..., :3],
                                   radius=0.2,
                                   K=32,
                                   return_nn=False)
    nn_features = masked_gather(perpoint_features[..., 3:], nn_idxs)
    # max pooling
    nn_features = nn_features.max(2)[0]
    # concat
    points_all = torch.concat([points, nn_features], axis=2)
    return points_all


def get_group_pc(pc: torch.Tensor,
                 local_centers: List,
                 group_num,
                 grasp_widths,
                 min_points=32,
                 is_training=True):
    batch_size, feature_len = pc.shape[0], pc.shape[2]
    pc_group = torch.zeros((0, group_num, feature_len),
                           dtype=torch.float32,
                           device='cuda')
    valid_local_centers = []
    valid_center_masks = []
    # get the points around one scored center
    for i in range(batch_size):
        # deal with empty input
        if len(local_centers[i]) == 0:
            # no need to append pc_group
            valid_local_centers.append(local_centers[i])
            valid_center_masks.append(
                torch.ones((0, ), dtype=torch.bool, device='cuda'))
            continue
        # cal distance and get masks (for all centers)
        dis = euclid_distance(local_centers[i], pc[i])
        # using grasp width for ball segment
        grasp_widths_tensor = torch.from_numpy(grasp_widths[i]).to(
            device='cuda', dtype=torch.float32)[..., None]
        # add noise when trainning
        width_scale = 1
        if is_training:
            # 0.8 ~ 1.2
            width_scale = 0.8 + 0.4 * torch.rand(
                (len(grasp_widths_tensor), 1), device='cuda')
        masks = (dis < grasp_widths_tensor * width_scale)
        # select valid center from all center
        center_cnt = len(local_centers[i])
        valid_mask = torch.ones((center_cnt, ), dtype=torch.bool).cuda()
        # concat pc first
        max_pc_cnt = max(group_num, masks.sum(1).max())
        partial_pcs = torch.zeros((center_cnt, max_pc_cnt, feature_len),
                                  device='cuda')
        lengths = torch.zeros((center_cnt, ), device='cuda')
        for j in range(center_cnt):
            # seg points
            partial_points = pc[i, masks[j]]
            point_cnt = partial_points.shape[0]
            if point_cnt < group_num:
                if point_cnt > min_points:
                    idxs = torch.randint(point_cnt, (group_num, ),
                                         device='cuda')
                    # idxs = np.random.choice(point_cnt, group_num, replace=True)
                    partial_points = partial_points[idxs]
                    point_cnt = group_num
                else:
                    valid_mask[j] = False
                    lengths[j] = group_num
                    continue
            partial_pcs[j, :point_cnt] = partial_points
            lengths[j] = point_cnt
        # add a little noise to avoid repeated points
        partial_pcs[..., :3] += torch.randn(partial_pcs.shape[:-1] + (3, ),
                                            device='cuda') * 5e-4
        # doing fps
        _, idxs = sample_farthest_points(partial_pcs[..., :3],
                                         lengths=lengths,
                                         K=group_num,
                                         random_start_point=True)
        # mv center of pc to (0, 0, 0), stack to pc_group
        temp_idxs = idxs[..., None].repeat(1, 1, feature_len)
        cur_pc = torch.gather(partial_pcs, 1, temp_idxs)
        cur_pc = cur_pc[valid_mask]
        cur_pc[..., :3] = cur_pc[..., :3] - local_centers[i][valid_mask][:,
                                                                         None]
        pc_group = torch.concat([pc_group, cur_pc], 0)
        # stack pc and get valid center list
        valid_local_centers.append(local_centers[i][valid_mask])
        valid_center_masks.append(valid_mask)
    return pc_group, valid_local_centers, valid_center_masks


def center2dtopc(rect_ggs: List,
                 center_num,
                 depths: torch.Tensor,
                 output_size,
                 append_random_center=True,
                 is_training=True):
    # add extra axis when valid, avoid dim errors
    batch_size = depths.shape[0]
    center_batch_pc = []

    scale_x, scale_y = 1280 / output_size[0], 720 / output_size[1]
    for i in range(batch_size):
        center_2d = rect_ggs[i].centers.copy()
        center_depth = rect_ggs[i].depths.copy()

        # add random center when local max count not enough
        if append_random_center and len(center_2d) < center_num:
            # print(f'current center_2d == {len(center_2d)}. using random center')
            random_local_max = np.random.rand(center_num - len(center_2d), 2)
            random_local_max = np.vstack([
                (random_local_max[:, 0] * output_size[0]).astype(np.int32),
                (random_local_max[:, 1] * output_size[1]).astype(np.int32)
            ]).T
            center_2d = np.vstack([center_2d, random_local_max])

        # scale
        center_2d[:, 0] = center_2d[:, 0] * scale_x
        center_2d[:, 1] = center_2d[:, 1] * scale_y
        # mask d != 0
        d = depths[i, center_2d[:, 0], center_2d[:, 1]]
        mask = (d != 0)
        # convert
        intrinsics = get_camera_intrinsic()
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        center_tensor = torch.from_numpy(center_2d).float().cuda()
        # add delta depth
        delta_d = torch.from_numpy(center_depth).cuda()
        z = (d[mask] + delta_d[mask]) / 1000.0
        x = z / fx * (center_tensor[mask, 0] - cx)
        y = z / fy * (center_tensor[mask, 1] - cy)
        cur_pc_tensor = torch.vstack([x, y, z]).T
        # deal with d == 0
        idxs = torch.nonzero(~mask).cpu().numpy().squeeze(-1)
        for j in idxs:
            x, y = center_2d[j, 0], center_2d[j, 1]
            # choose neighbor average to fix zero depth
            neighbor = 4
            x_range = slice(max(0, x - neighbor), min(1279, x + neighbor))
            y_range = slice(max(0, y - neighbor), min(719, y + neighbor))
            neighbor_depths = depths[i, x_range, y_range]
            depth_mask = (neighbor_depths > 0)
            if depth_mask.sum() == 0:
                # continue
                # this will use all centers
                cur_d = depths[i].mean()
            else:
                cur_d = neighbor_depths[depth_mask].float().median(
                ) + delta_d[j]
            # set valid mask
            mask[j] = True
            # convert
            new_center = torch.from_numpy(convert_2d_to_3d(
                x, y, cur_d.cpu())).cuda()
            cur_pc_tensor = torch.concat([cur_pc_tensor, new_center[None]], 0)

        # modify rect_ggs and append
        rect_ggs[i] = rect_ggs[i][mask.cpu().numpy()]
        # convert delta depth to actual depth for further width conversion
        rect_ggs[i].actual_depths = cur_pc_tensor[:, 2].cpu().numpy() * 1000.0
        # attention: rescale here
        rect_ggs[i].actual_depths *= (1280 // output_size[0])
        # add small noise to local centers (when train)
        if is_training:
            cur_pc_tensor += torch.randn(*cur_pc_tensor.shape,
                                         device='cuda') * 5e-3
        center_batch_pc.append(cur_pc_tensor)
    return center_batch_pc


def get_ori_grasp_label(grasppath):
    # load grasp
    grasp_label = np.load(grasppath[0])  # B=1
    grasp_num = grasp_label['centers_2d'].shape[0]
    gg_ori_labels = -np.ones((grasp_num, 8), dtype=np.float32)

    # get grasp original labels
    centers_2d = grasp_label['centers_2d']
    grasp_num = centers_2d.shape[0]
    gg_ori_labels[:, :3] = convert_2d_to_3d(centers_2d[:, 0], centers_2d[:, 1],
                                            grasp_label['center_z_depths'])
    gg_ori_labels[:, 3] = grasp_label['thetas_rad']
    gg_ori_labels[:, 4] = grasp_label['gammas_rad']
    gg_ori_labels[:, 5] = grasp_label['betas_rad']
    gg_ori_labels[:, 6] = grasp_label['widths_2d']
    gg_ori_labels[:, 7] = grasp_label['scores_from_6d']
    gg_ori_labels = torch.from_numpy(gg_ori_labels).cuda()

    return gg_ori_labels


def get_center_group_label(local_center: List, grasp_labels: List,
                           local_grasp_num) -> List:
    batch_size = len(local_center)
    gg_group_labels = []
    total_labels = torch.zeros((0, 8), dtype=torch.float32, device='cuda')
    for i in range(batch_size):
        # get grasp
        grasp_label = grasp_labels[i]
        # set up numpy grasp label
        centers_2d = grasp_label['centers_2d']
        grasp_num = centers_2d.shape[0]
        gg_label = -np.ones((grasp_num, 8), dtype=np.float32)
        gg_label[:, :3] = convert_2d_to_3d(centers_2d[:, 0], centers_2d[:, 1],
                                           grasp_label['center_z_depths'])
        gg_label[:, 3] = grasp_label['thetas_rad']
        gg_label[:, 4] = grasp_label['gammas_rad']
        gg_label[:, 5] = grasp_label['betas_rad']
        gg_label[:, 6] = grasp_label['widths_2d']
        gg_label[:, 7] = grasp_label['scores_from_6d']

        # convert to cuda tensor
        gg_label = torch.from_numpy(gg_label).cuda()

        # cal distance to valid center
        valid_center = local_center[i]
        distance = euclid_distance(
            valid_center, gg_label)  # distance: (center_num, grasp_num)

        # select nearest grasp labels for all center
        mask = (distance < 0.02)
        for j in range(len(distance)):
            # mask with min dis
            mask_gg = gg_label[mask[j]]
            mask_distance = distance[j][mask[j]]
            if len(mask_distance) == 0:
                gg_group_labels.append(torch.zeros((0, 8)).cuda())
            else:
                # sorted and select nearest
                _, topk_idxs = torch.topk(mask_distance,
                                          k=min(local_grasp_num,
                                                mask_distance.shape[0]),
                                          largest=False)
                gg_nearest = mask_gg[topk_idxs]
                # move to (0, 0, 0)
                gg_nearest[:, :3] = gg_nearest[:, :3] - valid_center[j]
                gg_group_labels.append(gg_nearest)
                total_labels = torch.cat([total_labels, gg_nearest], 0)
    return gg_group_labels, total_labels


@njit
def select_area(loc_map, top, bottom, left, right, grid_size, overlap):
    center_num = len(top)
    local_areas = np.zeros((center_num, (grid_size + overlap * 2)**2))
    for j in range(center_num):
        # extend to make overlap
        local_area = loc_map[top[j]:bottom[j], left[j]:right[j]]
        local_area = np.ascontiguousarray(local_area).reshape((-1, ))
        local_areas[j, :len(local_area)] = local_area
    return local_areas


def select_2d_center(loc_maps, center_num, reduce='max', grid_size=8) -> List:
    # deal with validation stage
    if isinstance(loc_maps, np.ndarray):
        loc_maps = loc_maps.copy()
    else:
        loc_maps = loc_maps.clone()
    if len(loc_maps.shape) == 2:
        loc_maps = loc_maps[None]
    # using torch to downsample
    if isinstance(loc_maps, np.ndarray):
        loc_maps = torch.from_numpy(loc_maps).cuda()
    batch_size = loc_maps.shape[0]
    center_2ds = []
    # using downsampled grid to avoid center too near
    new_size = (loc_maps.shape[1] // grid_size, loc_maps.shape[2] // grid_size)
    if reduce == 'avg':
        heat_grids = nnf.avg_pool2d(loc_maps[None], grid_size).squeeze()
    elif reduce == 'max':
        heat_grids = nnf.max_pool2d(loc_maps[None], grid_size).squeeze()
    else:
        raise RuntimeError(f'Unrecognized reduce: {reduce}')
    heat_grids = heat_grids.view((batch_size, -1))
    # get topk grid point
    for i in range(batch_size):
        local_idx = torch.topk(heat_grids[i],
                               k=min(heat_grids.shape[1], center_num),
                               dim=0)[1]
        local_max = np.zeros((len(local_idx), 2), dtype=np.int64)
        local_max[:, 0] = torch.div(local_idx,
                                    new_size[1],
                                    rounding_mode='floor').cpu().numpy()
        local_max[:, 1] = (local_idx % new_size[1]).cpu().numpy()
        # get local max in this grid point
        overlap = 1
        top, bottom = local_max[:, 0] * grid_size - overlap, (
            local_max[:, 0] + 1) * grid_size + overlap
        top, bottom = np.maximum(0, top), np.minimum(bottom,
                                                     loc_maps.shape[1] - 1)
        left, right = local_max[:, 1] * grid_size - overlap, (
            local_max[:, 1] + 1) * grid_size + overlap
        left, right = np.maximum(0, left), np.minimum(right,
                                                      loc_maps.shape[2] - 1)
        # using jit to faster get local areas
        local_areas = select_area(loc_maps[i].cpu().numpy(), top, bottom, left,
                                  right, grid_size, overlap)
        local_areas = torch.from_numpy(local_areas).float().cuda()
        # batch calculate
        grid_idxs = torch.argmax(local_areas, dim=1).cpu().numpy()
        local_max[:, 0] = top + grid_idxs // (right - left)
        local_max[:, 1] = left + grid_idxs % (right - left)
        center_2ds.append(local_max)
    return center_2ds


def data_process(points: torch.Tensor,
                 depths: torch.Tensor,
                 rect_ggs: List,
                 center_num,
                 group_num,
                 output_size,
                 min_points=32,
                 is_training=True):
    # select partial pc centers
    local_center = center2dtopc(rect_ggs,
                                center_num,
                                depths,
                                output_size,
                                append_random_center=False,
                                is_training=is_training)
    # get grasp width for pc segmentation
    grasp_widths = []
    for rect_gg in rect_ggs:
        grasp_widths.append(rect_gg.get_6d_width())
    # seg point cloud
    pc_group, valid_local_centers, valid_center_masks = get_group_pc(
        points,
        local_center,
        group_num,
        grasp_widths,
        min_points=min_points,
        is_training=is_training)
    # modify rect_ggs
    for i, mask in enumerate(valid_center_masks):
        rect_ggs[i] = rect_ggs[i][mask.cpu().numpy()]
    return pc_group, valid_local_centers
