from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from numba import njit
from numba.typed import List as typed_List

from dataset.utils import angle_distance, rotation_distance

eps = 1e-6


def focal_loss(pred,
               targets,
               thres=0.99,
               alpha=0.5,
               gamma=2,
               neg_suppress=False):
    pred = torch.clamp(torch.sigmoid(pred), min=eps, max=1 - eps)
    pos_inds = targets.ge(thres).float()
    neg_inds = targets.lt(thres).float()

    pos_loss = alpha * torch.log(pred) * torch.pow(1 - pred, gamma) * pos_inds
    neg_loss = (1 - alpha) * torch.log(1 - pred) * torch.pow(pred,
                                                             gamma) * neg_inds

    if neg_suppress:
        neg_loss *= torch.pow(1 - targets, 4)

    num_pos = pos_inds.sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos
    return loss


def offset_reg_loss(regs, gt_regs, mask):
    """Regression loss for offsets (z, w, theta) in Multi-Grasp Generator.

    :param regs: offset [B, anchor_k, H/r, W/r]
    :param gt_regs: ground truth [B, anchor_k, H/r, W/r]
    :param mask: classification mask
    :return: regression loss
    """
    mask = (mask > 0)
    regs = [torch.clamp(r, min=-0.5, max=0.5) for r in regs]
    loss = sum(
        F.smooth_l1_loss(r * mask, gt_r * mask, reduction='sum') /
        (mask.sum() + eps) for r, gt_r in zip(regs, gt_regs))
    return loss / len(regs)


def compute_anchor_loss(pred, target, loc_a=1, reg_b=0.5, cls_c=1):
    """Compute Anchor Loss.

    :param pred: A tuple (Location Map, Cls Mask, Offset theta, Offset h, Offset w)
    :param target: ground truth
    :param reg_a: weight of reg loss
    :param cls_b: weight of cls loss
    :return: total loss
    """
    pos_pred = pred[0]
    pos_target = target[0]
    pos_loss = focal_loss(pos_pred, pos_target, neg_suppress=True)

    angle_q_pred = pred[1]
    angle_q_target = target[1]
    cls_loss = focal_loss(angle_q_pred, angle_q_target, thres=0.5, alpha=0.25)
    reg_loss = offset_reg_loss(pred[2:], target[2:], angle_q_target)
    loss = loc_a * pos_loss + reg_b * reg_loss + cls_c * cls_loss
    return {
        'loss': loss,
        'losses': {
            'loc_map_loss': loc_a * pos_loss,
            'reg_loss': reg_b * reg_loss,
            'cls_loss': cls_c * cls_loss
        }
    }


def compute_multicls_loss(pred,
                          offset,
                          gg_labels: list,
                          grasp_info,
                          anchors,
                          args,
                          label_thres=0.99):
    center_num = len(gg_labels)
    anchor_num = len(anchors['gamma'])
    # get multi labels
    multi_labels = torch.zeros((center_num, anchor_num**2),
                               dtype=torch.float32,
                               device='cuda')
    offset_labels = torch.zeros((center_num, anchor_num**2, 3),
                                dtype=torch.float32,
                                device='cuda')
    # get q anchors
    anchors_gamma = anchors['gamma'] * torch.pi / 2
    anchors_beta = anchors['beta'] * torch.pi / 2
    # pre calculate anchor_eulers
    anchor_eulers = torch.zeros((anchor_num**2, 3), dtype=torch.float32)
    # attention: euler angle is [theta, gamma, beta]
    # but our anchor is gamma + beta * anchor_num
    beta_gamma = torch.cartesian_prod(anchors_beta, anchors_gamma)
    anchor_eulers[:, 1] = beta_gamma[:, 1]
    anchor_eulers[:, 2] = beta_gamma[:, 0]

    # use jit function to speed up
    # prepare input
    thetas = grasp_info[:, 0].cpu().numpy().astype(np.float64)
    label_offsets = typed_List(
        [g[:, :3].cpu().numpy().astype(np.float64) for g in gg_labels])
    label_eulers = typed_List(
        [g[:, 3:6].cpu().numpy().astype(np.float64) for g in gg_labels])
    multi_labels, offset_labels = faster_get_local_labels(
        anchor_eulers.numpy().astype(np.float64), thetas, label_eulers,
        label_offsets)
    # to cuda
    multi_labels = torch.from_numpy(multi_labels).to(device='cuda',
                                                     dtype=torch.float32)
    offset_labels = torch.from_numpy(offset_labels).to(device='cuda',
                                                       dtype=torch.float32)
    # compute focal loss for anchor
    loss_multi = focal_loss(pred, multi_labels, thres=label_thres)

    # mask smooth l1 loss for offset
    offset_labels /= 0.02
    labels_mask = (multi_labels > label_thres)[..., None]
    loss_offset = F.smooth_l1_loss(offset * labels_mask,
                                   offset_labels * labels_mask,
                                   reduction='sum')
    loss_offset /= (labels_mask.sum() + eps)
    loss_offset *= args.offset_d
    return loss_multi, loss_offset


@njit
def faster_get_local_labels(eulers, thetas, label_eulers, label_offsets):

    def jit_qmul(q0, q1):
        w0, x0, y0, z0 = q0.T
        w1, x1, y1, z1 = q1.T
        return np.stack(
            (-x0 * x1 - y0 * y1 - z0 * z1 + w0 * w1, x0 * w1 + y0 * z1 -
             z0 * y1 + w0 * x1, -x0 * z1 + y0 * w1 + z0 * x1 + w0 * y1,
             x0 * y1 - y0 * x1 + z0 * w1 + w0 * z1), 1)

    def jit_euler2quaternion(euler):
        N = len(euler)
        qx = np.stack((np.cos(euler[:, 0] / 2), np.sin(
            euler[:, 0] / 2), np.zeros(
                (N, ), dtype=np.float64), np.zeros((N, ), dtype=np.float64)),
                      1)
        qy = np.stack(
            (np.cos(euler[:, 1] / 2), np.zeros(
                (N, ), dtype=np.float64), np.sin(
                    euler[:, 1] / 2), np.zeros((N, ), dtype=np.float64)), 1)
        qz = np.stack(
            (np.cos(euler[:, 2] / 2), np.zeros(
                (N, ), dtype=np.float64), np.zeros(
                    (N, ), dtype=np.float64), np.sin(euler[:, 2] / 2)), 1)
        q = jit_qmul(jit_qmul(qx, qy), qz)
        return q

    def jit_rotation_distance(a, b):
        q_a = jit_euler2quaternion(a)
        q_b = jit_euler2quaternion(b)
        # symmetry for gripper
        a_r = a * np.array([[1, -1, -1]])
        a_r[:, 0] += np.pi
        q_a_r = jit_euler2quaternion(a_r)
        # q_a (grasp_cnt, 4) b (label_cnt, 4)
        dis = 1 - np.maximum(np.abs(np.dot(q_a, q_b.T)),
                             np.abs(np.dot(q_a_r, q_b.T)))
        return dis

    # get size
    M, N = len(eulers), len(label_eulers)
    # get multi labels
    multi_labels = np.zeros((N, M), dtype=np.float64)
    offset_labels = np.zeros((N, M, 3), dtype=np.float64)
    # set up eulers
    for i in range(N):
        if len(label_eulers[i]) > 0:
            # set current anchors
            eulers[:, 0] = thetas[i]
            # cal labels dis to nearest anchors
            rot_dis = jit_rotation_distance(
                eulers, label_eulers[i])  # M * local_grasp_num
            # get labels to earest anchors
            for j in range(M):
                # soft dis label
                multi_labels[i, j] = 1 - rot_dis[j].min()
                # set offset labels with thres
                idx = np.argmin(rot_dis[j])
                offset_labels[i, j] = label_offsets[i][idx]
    return multi_labels, offset_labels
