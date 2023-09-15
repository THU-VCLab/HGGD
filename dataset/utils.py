import numpy as np
import torch
import torch.nn.functional as nnf
from numba import jit
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_quaternion

from .config import get_camera_intrinsic


def convert_2d_to_3d(x, y, d):
    # convert xyd in 2d to xyz in 3d
    # should be 1280 * 720 here
    intrinsics = get_camera_intrinsic()
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    z = d / 1000.0
    x = z / fx * (x - cx)
    y = z / fy * (y - cy)
    return np.array([x, y, z]).T


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom,
                               radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  #
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def euclid_distance(points1: torch.Tensor,
                    points2: torch.Tensor) -> torch.Tensor:
    # cal center distance from all p1 to all p2: (len(p1), len(p2))
    # choose translation (first 3 dim) from the whole tensor
    p1, p2 = points1[:, :3].double(), points2[:, :3].double()
    # calculate dis in shape: (grasp_cnt, label_cnt)
    distance = -2 * torch.matmul(p1, p2.transpose(1, 0))
    distance += p1.square().sum(1).repeat(p2.size(0), 1).T
    distance += p2.square().sum(1).repeat(p1.size(0), 1)
    return distance.sqrt()


def rotation_distance(a: torch.Tensor,
                      b: torch.Tensor,
                      symmetry=True) -> torch.Tensor:
    # convert euler to quaterion
    # q_a (grasp_cnt, 3) b (label_cnt, 3)
    q_a = matrix_to_quaternion(euler_angles_to_matrix(a, convention='XYZ'))
    q_b = matrix_to_quaternion(euler_angles_to_matrix(b, convention='XYZ'))
    if symmetry:
        # symmetry for gripper
        a_r = a * torch.FloatTensor([[1, -1, -1]]).to(device=a.device)
        a_r[:, 0] += torch.pi
        q_a_r = matrix_to_quaternion(
            euler_angles_to_matrix(a_r, convention='XYZ'))
        # q_a (grasp_cnt, 4) b (label_cnt, 4)
        dis = 1 - torch.maximum(
            torch.mm(q_a, q_b.T).abs(),
            torch.mm(q_a_r, q_b.T).abs())
    else:
        dis = 1 - torch.mm(q_a, q_b.T).abs()
    return dis


def angle_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # expand a, b to (grasp_cnt, label_cnt)
    grasp_cnt, label_cnt = a.shape[0], b.shape[0]
    a = a.repeat(label_cnt, 1).T
    b = b.repeat(grasp_cnt, 1)
    # cosine distance
    dis = (1 - (a - b).cos()) / 2
    return dis


def angle_distance_using_vector(a: torch.Tensor,
                                b: torch.Tensor) -> torch.Tensor:
    # expand a to (grasp_cnt, label_cnt, 2)
    grasp_cnt, label_cnt = a.size(0), b.size(0)
    a = torch.vstack((a.cos(), a.sin())).T
    a = a.repeat(label_cnt, 1, 1).transpose(0, 1)
    # expand b to (grasp_cnt, label_cnt, 2)
    b = torch.vstack((b.cos(), b.sin())).T
    b = b.repeat(grasp_cnt, 1, 1)
    dis = vector_distance(a, b)
    return dis


def vector_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # using cosine distance among vectors
    # a: (label_cnt, grasp_cnt, 2)  b: (label_cnt, grasp_cnt, 2)
    eps = 1e-6
    a_b = (a * b).sum(2)
    div_ab = (a.square().sum(2) * b.square().sum(2) + eps).sqrt()
    dis = (1 - torch.div(a_b, div_ab)) / 2
    return dis


def iter_anchors(anchor_ranges, anchors, angles):
    # get B
    B = np.searchsorted(anchor_ranges, angles) - 1
    bin_counts = np.bincount(B)
    zero_count = len(anchor_ranges) - len(bin_counts) - 1
    if zero_count > 0:
        bin_counts = np.concatenate(
            [bin_counts, np.zeros((zero_count, ))], axis=0)
    # get zero pos
    zero_idxs = (bin_counts == 0)
    old_anchors = np.copy(anchors[zero_idxs])
    # convert to one hot
    B_one_hot = np.zeros((B.size, len(bin_counts)))
    B_one_hot[np.arange(B.size), B] = 1
    # get anchors
    B_inv = np.linalg.inv(np.eye(len(bin_counts)) * (bin_counts + 1))
    anchors = B_inv @ B_one_hot.T @ angles
    # overwrite using old anchors
    anchors[zero_idxs] = old_anchors
    # get range
    anchor_ranges = (anchors[1:] + anchors[:-1]) / 2
    anchor_ranges = np.array([-1] + list(anchor_ranges) + [1])
    print(bin_counts)
    print(anchors)
    return anchor_ranges, anchors, bin_counts


def shift_anchors(gg_labels, anchors, iter_times=1):
    # stack labels for this batch
    gammas, betas = np.zeros((0, )), np.zeros((0, ))
    gg_labels = gg_labels.numpy()
    # clip angle to avoid bugs
    gammas = gg_labels[:, 4] / np.pi * 2
    betas = gg_labels[:, 5] / np.pi * 2
    gammas = np.clip(gammas, -1 + 1e-6, 1 - 1e-6)
    betas = np.clip(betas, -1 + 1e-6, 1 - 1e-6)
    # get anchor range
    gamma_anchors = anchors['gamma'].cpu().numpy()
    beta_anchors = anchors['beta'].cpu().numpy()
    gamma_ranges = (gamma_anchors[1:] + gamma_anchors[:-1]) / 2
    gamma_ranges = np.array([-1] + list(gamma_ranges) + [1])
    beta_ranges = (beta_anchors[1:] + beta_anchors[:-1]) / 2
    beta_ranges = np.array([-1] + list(beta_ranges) + [1])
    # iter update
    for _ in range(iter_times):
        print('gamma:')
        gamma_ranges, gamma_anchors, gamma_cnt = iter_anchors(
            gamma_ranges, gamma_anchors, gammas)
        print('beta:')
        beta_ranges, beta_anchors, beta_cnt = iter_anchors(
            beta_ranges, beta_anchors, betas)
    # trans to torch cuda tensor
    new_gamma_anchors = torch.from_numpy(gamma_anchors).cuda()
    new_beta_anchors = torch.from_numpy(beta_anchors).cuda()
    # moving average
    anchors['gamma'] = (new_gamma_anchors + anchors['gamma']) / 2
    anchors['beta'] = (new_beta_anchors + anchors['beta']) / 2
    return anchors


def fast_sample(num_samples, sample_size):
    probabilities = np.full((num_samples, ), 1.0 / num_samples)
    # get random shifting numbers & scale them correctly
    random_shifts = np.random.random(probabilities.shape)
    random_shifts = random_shifts / random_shifts.sum()
    # shift by numbers & find largest (by finding the smallest of the negative)
    shifted_probabilities = random_shifts - probabilities
    return np.argpartition(shifted_probabilities, sample_size)[:sample_size]


class PointCloudHelper:

    def __init__(self, all_points_num) -> None:
        # precalculate x,y map
        self.all_points_num = all_points_num
        self.output_shape = (80, 45)
        # get intrinsics
        intrinsics = get_camera_intrinsic()
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        # cal x, y
        ymap, xmap = np.meshgrid(np.arange(720), np.arange(1280))
        points_x = (xmap - cx) / fx
        points_y = (ymap - cy) / fy
        self.points_x = torch.from_numpy(points_x).float()
        self.points_y = torch.from_numpy(points_y).float()
        # for get downsampled xyz map
        ymap, xmap = np.meshgrid(np.arange(self.output_shape[1]),
                                 np.arange(self.output_shape[0]))
        factor = 1280 / self.output_shape[0]
        points_x = (xmap - cx / factor) / (fx / factor)
        points_y = (ymap - cy / factor) / (fy / factor)
        self.points_x_downscale = torch.from_numpy(points_x).float()
        self.points_y_downscale = torch.from_numpy(points_y).float()

    def to_scene_points(self,
                        rgbs: torch.Tensor,
                        depths: torch.Tensor,
                        include_rgb=True):
        batch_size = rgbs.shape[0]
        feature_len = 3 + 3 * include_rgb
        points_all = -torch.ones(
            (batch_size, self.all_points_num, feature_len),
            dtype=torch.float32).cuda()
        # cal z
        idxs = []
        masks = (depths > 0)
        cur_zs = depths / 1000.0
        cur_xs = self.points_x.cuda() * cur_zs
        cur_ys = self.points_y.cuda() * cur_zs
        for i in range(batch_size):
            # convert point cloud to xyz maps
            points = torch.stack([cur_xs[i], cur_ys[i], cur_zs[i]], axis=-1)
            # remove zero depth
            mask = masks[i]
            points = points[mask]
            colors = rgbs[i][:, mask].T

            # random sample if points more than required
            if len(points) >= self.all_points_num:
                cur_idxs = fast_sample(len(points), self.all_points_num)
                points = points[cur_idxs]
                colors = colors[cur_idxs]
                # save idxs for concat fusion
                idxs.append(cur_idxs)

            # concat rgb and features after translation
            if include_rgb:
                points_all[i] = torch.concat([points, colors], axis=1)
            else:
                points_all[i] = points
        return points_all, idxs, masks

    def to_xyz_maps(self, depths):
        # downsample
        downsample_depths = nnf.interpolate(depths[:, None],
                                            size=self.output_shape,
                                            mode='nearest').squeeze(1).cuda()
        # convert xyzs
        cur_zs = downsample_depths / 1000.0
        cur_xs = self.points_x_downscale.cuda() * cur_zs
        cur_ys = self.points_y_downscale.cuda() * cur_zs
        xyzs = torch.stack([cur_xs, cur_ys, cur_zs], axis=-1)
        return xyzs.transpose(1, 3).transpose(2, 3)
