import math
from typing import List

import numpy as np
import torch
from numba import njit
from torchvision.transforms.functional import gaussian_blur

from dataset.collision_detector import ModelFreeCollisionDetector

from .config import get_camera_intrinsic
from .grasp import RectGrasp, RectGraspGroup
from .pc_dataset_tools import select_2d_center
from .utils import angle_distance, euclid_distance, rotation_distance

eps = 1e-6


def anchor_output_process(loc_map,
                          cls_mask,
                          theta_offset,
                          depth_offset,
                          width_offset,
                          sigma=10):
    # normlize every location map
    loc_map = torch.clamp(torch.sigmoid(loc_map), min=eps, max=1 - eps)
    loc_map = gaussian_blur(loc_map, kernel_size=9, sigma=sigma)
    loc_map_gaussian = loc_map.detach().cpu().numpy().squeeze()
    # using sigmoid to norm class map
    cls_mask = torch.clamp(torch.sigmoid(cls_mask), min=eps, max=1 - eps)
    cls_mask = cls_mask.detach().cpu().numpy().squeeze()
    # clamp regress offset
    theta_offset = torch.clamp(theta_offset.detach(), -0.5,
                               0.5).cpu().numpy().squeeze()
    depth_offset = torch.clamp(depth_offset.detach(), -0.5,
                               0.5).cpu().numpy().squeeze()
    width_offset = torch.clamp(width_offset.detach(), -1,
                               1).cpu().numpy().squeeze()

    return loc_map_gaussian, cls_mask, theta_offset, depth_offset, width_offset


@njit
def jit_detect_2d(loc_map, local_max, anchor_clss, grid_points, theta_offset,
                  depth_offset, width_offset, rotation_num, anchor_k, anchor_w,
                  anchor_z, grasp_nms, center_num):
    centers = np.zeros((0, 2))
    scores = np.zeros((0, ))
    depths = np.zeros((0, ))
    thetas = np.zeros((0, ))
    widths = np.zeros((0, ))
    for i in range(len(local_max)):
        if len(centers) >= center_num:
            break
        for anchor_cls in anchor_clss[i, ::-1][:rotation_num]:
            pos = (anchor_cls, grid_points[i, 0], grid_points[i, 1])
            da = theta_offset[pos]
            dz = depth_offset[pos]
            dw = width_offset[pos]
            # recover depth and width
            depth = dz * anchor_z * 2
            w = anchor_w * np.exp(dw)
            # attention: our theta in [0, pi]
            theta_range = np.pi
            anchor_step = theta_range / anchor_k
            theta = anchor_step * (anchor_cls + da + 0.5) - theta_range / 2
            score = loc_map[local_max[i, 0], local_max[i, 1]]
            # grasp nms, dis > grid_size and delta angle > pi /6
            isnew = True
            if grasp_nms > 0 and len(centers) > 0:
                center_dis = np.sqrt(
                    np.sum(np.square(centers - local_max[i]), axis=1))
                angle_dis = np.abs(thetas - theta)
                angle_dis = np.minimum(np.pi - angle_dis, angle_dis)
                mask = np.logical_and(center_dis < grasp_nms,
                                      angle_dis < np.pi / 6)
                isnew = (not mask.any())
            if isnew:
                centers = np.vstack((centers, np.expand_dims(local_max[i], 0)))
                thetas = np.hstack((thetas, np.array([theta])))
                scores = np.hstack((scores, np.array([score])))
                depths = np.hstack((depths, np.array([depth])))
                widths = np.hstack((widths, np.array([w])))
    return centers, widths, depths, scores, thetas


def detect_2d_grasp(loc_map,
                    cls_mask,
                    theta_offset,
                    depth_offset,
                    width_offset,
                    ratio,
                    anchor_k=6,
                    anchor_w=50.0,
                    anchor_z=20.0,
                    mask_thre=0,
                    center_num=1,
                    rotation_num=1,
                    reduce='max',
                    grid_size=8,
                    grasp_nms=8) -> RectGraspGroup:
    """detect 2d grasp from GHM heatmaps.

    Args:
        loc_map (Tensor): B x 1 x H x W
        cls_mask (Tensor): B x anchor_k x H_r x W_r
        theta_offset (Tensor): B x anchor_k x H_r x W_r
        depth_offset (Tensor): B x 1 x H_r x W_r
        width_offset (Tensor): B x 1 x H_r x W_r
        ratio (int): image downsample ratio (local grid size)
        anchor_k (int, optional): anchor num for theta angle. Defaults to 6.
        anchor_w (float, optional): anchor for grasp width. Defaults to 50.0.
        anchor_z (float, optional): anchor for grasp depth. Defaults to 20.0.
        mask_thre (int, optional): mask applied on heat score. Defaults to 0.
        center_num (int, optional): detect 2d center num. Defaults to 1.
        rotation_num (int, optional): theta angle num in each 2d center. Defaults to 1.
        reduce (str, optional): reduce type for grid-based sampling. Defaults to 'max'.
        grid_size (int, optional): grid size for grid-based sampling. Defaults to 8.
        grasp_nms (int, optional): min grasp dis for grid-based sampling. Defaults to 8.

    Returns:
        RectGraspGroup
    """
    local_max = select_2d_center(loc_map,
                                 center_num * 10,
                                 grid_size=grid_size,
                                 reduce=reduce)
    centers = []
    scores = []
    depths = []
    thetas = []
    widths = []
    # batch reduction
    loc_map = loc_map.squeeze()
    local_max = local_max[0]
    # filter by heatmap
    qualitys = loc_map[local_max[:, 0], local_max[:, 1]]
    quality_mask = (qualitys > mask_thre)
    local_max = local_max[quality_mask]
    grid_points = local_max // ratio
    cls_qualitys = cls_mask[:, grid_points[:, 0], grid_points[:, 1]].T
    # sort cls score
    anchor_clss = np.argsort(cls_qualitys)
    centers, widths, depths, scores, thetas = jit_detect_2d(
        loc_map, local_max, anchor_clss, grid_points, theta_offset,
        depth_offset, width_offset, rotation_num, anchor_k, anchor_w, anchor_z,
        grasp_nms, center_num)
    grasps = RectGraspGroup(centers=np.array(centers, dtype=np.int64),
                            heights=np.full((len(centers), ), 25),
                            widths=np.array(widths),
                            depths=np.array(depths),
                            scores=np.array(scores),
                            thetas=np.array(thetas))
    return grasps


@njit
def jit_detect_6d(pred_gammas, pred_betas, offset, scores, local_centers,
                  anchor_idxs, thetas, widths, k, scale_x, intrinsics):
    pred_grasp = np.zeros((0, 8))
    cur_grasp = np.zeros((k, 8))
    center_2ds = np.zeros((0, 2))
    center_depths = np.zeros((0, ))
    cur_offsets = np.zeros((k, 3))
    for i in range(len(pred_gammas)):
        for j in range(k):
            cur_offsets[j] = offset[i, anchor_idxs[i, j]]
        # pred grasp: x,y,z,theta,gamma,beta,width
        cur_centers = local_centers[i] + cur_offsets
        cur_grasp[:, :3] = cur_centers
        cur_grasp[:, 3] = np.repeat(thetas[i], k)
        cur_grasp[:, 4] = pred_gammas[i]
        cur_grasp[:, 5] = pred_betas[i]
        cur_grasp[:, 6] = np.repeat(widths[i] * scale_x, k)
        cur_grasp[:, 7] = scores[i]
        # stack on all grasp array
        pred_grasp = np.vstack((pred_grasp, cur_grasp))
        # cal 2d centers after offset
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        cur_center_2ds = np.zeros((k, 2))
        cur_center_2ds[:, 0] = cur_centers[:, 0] / cur_centers[:, 2] * fx + cx
        cur_center_2ds[:, 1] = cur_centers[:, 1] / cur_centers[:, 2] * fy + cy
        # update center info for generating RectGraspGroup
        center_2ds = np.vstack((center_2ds, cur_center_2ds))
        center_depths = np.concatenate(
            (center_depths, cur_centers[:, 2] * 1e3))
    return pred_grasp, center_2ds, center_depths


def detect_6d_grasp_multi(rect_gg: RectGraspGroup,
                          pred_view: torch.Tensor,
                          offset: torch.Tensor,
                          valid_center_list: List,
                          output_size,
                          anchors,
                          k=5):
    # pre-process
    anchor_num = len(anchors['gamma'])
    pred_view = torch.sigmoid(pred_view)
    offset = 0.02 * torch.clip(offset, -1, 1)
    # select top-k every row
    scores, anchor_idxs = torch.topk(pred_view, k, 1)
    local_centers = valid_center_list[0].cpu().numpy()
    offset = offset.cpu().numpy()
    scores = scores.cpu().numpy()
    anchor_idxs = anchor_idxs.cpu().numpy()
    # split anchor to gamma and beta
    gamma_idxs = (anchor_idxs % anchor_num).flatten()
    beta_idxs = (anchor_idxs // anchor_num).flatten()
    norm_gammas = anchors['gamma'][gamma_idxs].cpu().numpy().reshape(
        anchor_idxs.shape)
    norm_betas = anchors['beta'][beta_idxs].cpu().numpy().reshape(
        anchor_idxs.shape)
    # make back to [-pi / 2, pi / 2]
    pred_gammas = norm_gammas * torch.pi / 2
    pred_betas = norm_betas * torch.pi / 2
    # get offset
    scale_x = 1280 / output_size[0]
    intrinsics = get_camera_intrinsic()
    pred_grasp, center_2ds, center_depths = jit_detect_6d(
        pred_gammas, pred_betas, offset, scores, local_centers, anchor_idxs,
        rect_gg.thetas, rect_gg.widths, k, scale_x, intrinsics)
    # filter by gamma
    gamma_mask = (abs(pred_grasp[:, 4]) < torch.pi / 3).reshape(-1)
    pred_grasp = pred_grasp[gamma_mask]
    center_2ds = center_2ds[gamma_mask]
    center_depths = center_depths[gamma_mask]

    # print(pred_gammas.shape, pred_grasp.shape, gamma_mask.reshape(-1).shape)
    if len(pred_grasp) > 0:
        pred_rect_gg = RectGraspGroup(
            centers=center_2ds[:, :2],
            depths=center_depths,
            thetas=pred_grasp[:, 3],
            gammas=pred_grasp[:, 4],
            betas=pred_grasp[:, 5],
            widths=pred_grasp[:, 6],
            scores=pred_grasp[:, 7],  # 1
            heights=20 * np.ones(pred_grasp.shape[0], ))
    else:
        pred_rect_gg = RectGraspGroup()

    return pred_grasp, pred_rect_gg


def calculate_6d_match(pred_grasp: torch.Tensor,
                       gg_ori_label,
                       threshold_dis,
                       threshold_rot,
                       seperate=False):
    # center distance
    distance = euclid_distance(pred_grasp, gg_ori_label)
    mask_distance = (distance < threshold_dis)
    soft_mask_distance = (distance < threshold_dis * 2)

    # angle distance
    if seperate:
        # compute angle distance bewteen pred and label seperately
        dis_theta = angle_distance(pred_grasp[:, 3], gg_ori_label[:, 3])
        dis_gamma = angle_distance(pred_grasp[:, 4], gg_ori_label[:, 4])
        dis_beta = angle_distance(pred_grasp[:, 5], gg_ori_label[:, 5])

        # get mask and logical_or along label axis
        mask_theta = (dis_theta < threshold_rot)
        mask_gamma = (dis_gamma < threshold_rot)
        mask_beta = (dis_beta < threshold_rot)

        mask_rot = torch.logical_and(torch.logical_and(mask_theta, mask_beta),
                                     mask_gamma)
    else:
        # using total rotation distance
        dis_rot = rotation_distance(pred_grasp[:, 3:6], gg_ori_label[:, 3:6])
        mask_rot = (dis_rot < threshold_rot)

    mask = torch.logical_and(mask_distance, mask_rot).any(1)

    # cal angle correct rate
    batch_size = mask_distance.size(0)
    correct_dis = mask_distance.any(1).sum()
    correct_rot = torch.logical_and(mask_rot, soft_mask_distance).any(1).sum()

    # get count
    correct_grasp = mask.sum()
    acc = correct_grasp / pred_grasp.shape[0]

    # create T & F array
    r_g = np.array([correct_grasp.cpu(), batch_size - correct_grasp.cpu()])
    r_d = np.array([correct_dis.cpu(), batch_size - correct_dis.cpu()])
    r_r = np.array([correct_rot.cpu(), batch_size - correct_rot.cpu()])
    return r_g, r_d, r_r


def calculate_coverage(pred_grasp: torch.Tensor,
                       gg_ori_label: torch.Tensor,
                       threshold_dis=0.02,
                       threshold_rot=0.1):
    # center distance
    distance = euclid_distance(gg_ori_label, pred_grasp)
    mask_distance = (distance < threshold_dis)

    # angle distance, using total rotation distance
    dis_rot = rotation_distance(gg_ori_label[:, 3:6], pred_grasp[:, 3:6])
    mask_rot = (dis_rot < threshold_rot)

    mask = torch.logical_and(mask_distance, mask_rot).any(1)
    cover_cnt = mask.sum()
    return cover_cnt


def calculate_guidance_accuracy(g, gt_bbs):
    # check center dis and theta dis
    gts = RectGraspGroup()
    for bb in gt_bbs:
        gt = RectGrasp.from_bb(bb)
        gts.append(gt)
    gts_6d = gts.to_6d_grasp_group()

    g_6d = RectGraspGroup()
    g_6d.append(g)
    g_6d = g_6d.to_6d_grasp_group()[0]

    center_dis = np.sqrt(
        np.square(g_6d.translation - gts_6d.translations).sum(1))
    theta_dis = np.abs(g.theta - gts.thetas)
    theta_dis = np.minimum(np.pi - theta_dis, theta_dis)
    center_mask = (center_dis < 0.02)
    theta_mask = (theta_dis < np.pi / 6)
    mask = np.logical_and(center_mask, theta_mask)
    return mask.any()


def calculate_iou_match(gs, gt_bbs, thre=0.25):
    # check iou
    for g in gs:
        for bb in gt_bbs:
            gt = RectGrasp.from_bb(bb)
            if g.iou(gt) > thre:
                return True
    return False


def collision_detect(points_all: torch.Tensor, pred_gg, mode='regnet'):
    # collison detect
    cloud = points_all[:, :3].clone()
    mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=0.01, mode=mode)
    no_collision_mask = mfcdetector.detect(pred_gg, approach_dist=0.05)
    collision_free_gg = pred_gg[no_collision_mask]
    return collision_free_gg, no_collision_mask
