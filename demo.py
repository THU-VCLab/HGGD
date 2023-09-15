import argparse
import os
import random
from time import time

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from PIL import Image

from dataset.config import get_camera_intrinsic
from dataset.evaluation import (anchor_output_process, collision_detect,
                                detect_2d_grasp, detect_6d_grasp_multi)
from dataset.pc_dataset_tools import data_process, feature_fusion
from models.anchornet import AnchorGraspNet
from models.localgraspnet import PointMultiGraspNet
from train_utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint-path', default=None)

# image input
parser.add_argument('--rgb-path')
parser.add_argument('--depth-path')

# 2d
parser.add_argument('--input-h', type=int)
parser.add_argument('--input-w', type=int)
parser.add_argument('--sigma', type=int, default=10)
parser.add_argument('--use-depth', type=int, default=1)
parser.add_argument('--use-rgb', type=int, default=1)
parser.add_argument('--ratio', type=int, default=8)
parser.add_argument('--anchor-k', type=int, default=6)
parser.add_argument('--anchor-w', type=float, default=50.0)
parser.add_argument('--anchor-z', type=float, default=20.0)
parser.add_argument('--grid-size', type=int, default=8)

# pc
parser.add_argument('--anchor-num', type=int)
parser.add_argument('--all-points-num', type=int)
parser.add_argument('--center-num', type=int)
parser.add_argument('--group-num', type=int)

# grasp detection
parser.add_argument('--heatmap-thres', type=float, default=0.01)
parser.add_argument('--local-k', type=int, default=10)
parser.add_argument('--local-thres', type=float, default=0.01)
parser.add_argument('--rotation-num', type=int, default=1)

# others
parser.add_argument('--random-seed', type=int, default=123, help='Random seed')

args = parser.parse_args()


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
                cur_idxs = random.sample(range(len(points)),
                                         self.all_points_num)
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
        downsample_depths = F.interpolate(depths[:, None],
                                          size=self.output_shape,
                                          mode='nearest').squeeze(1).cuda()
        # convert xyzs
        cur_zs = downsample_depths / 1000.0
        cur_xs = self.points_x_downscale.cuda() * cur_zs
        cur_ys = self.points_y_downscale.cuda() * cur_zs
        xyzs = torch.stack([cur_xs, cur_ys, cur_zs], axis=-1)
        return xyzs.permute(0, 3, 1, 2)


def inference(view_points,
              xyzs,
              x,
              ori_depth,
              vis_heatmap=False,
              vis_grasp=True):
    with torch.no_grad():
        # 2d prediction
        pred_2d, perpoint_features = anchornet(x)

        loc_map, cls_mask, theta_offset, height_offset, width_offset = \
            anchor_output_process(*pred_2d, sigma=args.sigma)

        # detect 2d grasp (x, y, theta)
        rect_gg = detect_2d_grasp(loc_map,
                                  cls_mask,
                                  theta_offset,
                                  height_offset,
                                  width_offset,
                                  ratio=args.ratio,
                                  anchor_k=args.anchor_k,
                                  anchor_w=args.anchor_w,
                                  anchor_z=args.anchor_z,
                                  mask_thre=args.heatmap_thres,
                                  center_num=args.center_num,
                                  grid_size=args.grid_size,
                                  grasp_nms=args.grid_size,
                                  reduce='max')

        # check 2d result
        if rect_gg.size == 0:
            print('No 2d grasp found')
            return

        # show heatmap
        if vis_heatmap:
            rgb_t = x[0, 1:].cpu().numpy().squeeze().transpose(2, 1, 0)
            resized_rgb = Image.fromarray((rgb_t * 255.0).astype(np.uint8))
            resized_rgb = np.array(
                resized_rgb.resize((args.input_w, args.input_h))) / 255.0
            depth_t = ori_depth.cpu().numpy().squeeze().T
            plt.subplot(221)
            plt.imshow(rgb_t)
            plt.subplot(222)
            plt.imshow(depth_t)
            plt.subplot(223)
            plt.imshow(loc_map.squeeze().T, cmap='jet')
            plt.subplot(224)
            rect_rgb = rect_gg.plot_rect_grasp_group(resized_rgb, 0)
            plt.imshow(rect_rgb)
            plt.tight_layout()
            plt.show()

        # feature fusion
        points_all = feature_fusion(view_points[..., :3], perpoint_features,
                                    xyzs)
        rect_ggs = [rect_gg]
        pc_group, valid_local_centers = data_process(
            points_all,
            ori_depth,
            rect_ggs,
            args.center_num,
            args.group_num, (args.input_w, args.input_h),
            min_points=32,
            is_training=False)
        rect_gg = rect_ggs[0]
        # batch_size == 1 when valid
        points_all = points_all.squeeze()

        # get 2d grasp info (not grasp itself) for trainning
        grasp_info = np.zeros((0, 3), dtype=np.float32)
        g_thetas = rect_gg.thetas[None]
        g_ws = rect_gg.widths[None]
        g_ds = rect_gg.depths[None]
        cur_info = np.vstack([g_thetas, g_ws, g_ds])
        grasp_info = np.vstack([grasp_info, cur_info.T])
        grasp_info = torch.from_numpy(grasp_info).to(dtype=torch.float32,
                                                     device='cuda')

        # localnet
        _, pred, offset = localnet(pc_group, grasp_info)

        # detect 6d grasp from 2d output and 6d output
        _, pred_rect_gg = detect_6d_grasp_multi(rect_gg,
                                                pred,
                                                offset,
                                                valid_local_centers,
                                                (args.input_w, args.input_h),
                                                anchors,
                                                k=args.local_k)

        # collision detect
        pred_grasp_from_rect = pred_rect_gg.to_6d_grasp_group(depth=0.02)
        pred_gg, _ = collision_detect(points_all,
                                      pred_grasp_from_rect,
                                      mode='graspnet')

        # nms
        pred_gg = pred_gg.nms()

        # show grasp
        if vis_grasp:
            print('pred grasp num ==', len(pred_gg))
            grasp_geo = pred_gg.to_open3d_geometry_list()
            points = view_points[..., :3].cpu().numpy().squeeze()
            colors = view_points[..., 3:6].cpu().numpy().squeeze()
            vispc = o3d.geometry.PointCloud()
            vispc.points = o3d.utility.Vector3dVector(points)
            vispc.colors = o3d.utility.Vector3dVector(colors)
            o3d.visualization.draw_geometries([vispc] + grasp_geo)
        return pred_gg


if __name__ == '__main__':
    # set up pc transform helper
    pc_helper = PointCloudHelper(all_points_num=args.all_points_num)

    # set torch and gpu setting
    np.set_printoptions(precision=4, suppress=True)
    torch.set_printoptions(precision=4, sci_mode=False)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
    else:
        raise RuntimeError('CUDA not available')

    # random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    # Init the model
    anchornet = AnchorGraspNet(in_dim=4,
                               ratio=args.ratio,
                               anchor_k=args.anchor_k)
    localnet = PointMultiGraspNet(info_size=3, k_cls=args.anchor_num**2)

    # gpu
    anchornet = anchornet.cuda()
    localnet = localnet.cuda()

    # Load checkpoint
    check_point = torch.load(args.checkpoint_path)
    anchornet.load_state_dict(check_point['anchor'])
    localnet.load_state_dict(check_point['local'])
    # load checkpoint
    basic_ranges = torch.linspace(-1, 1, args.anchor_num + 1).cuda()
    basic_anchors = (basic_ranges[1:] + basic_ranges[:-1]) / 2
    anchors = {'gamma': basic_anchors, 'beta': basic_anchors}
    anchors['gamma'] = check_point['gamma']
    anchors['beta'] = check_point['beta']
    logging.info('Using saved anchors')
    print('-> loaded checkpoint %s ' % (args.checkpoint_path))

    # network eval mode
    anchornet.eval()
    localnet.eval()

    # read image and conver to tensor
    ori_depth = np.array(Image.open(args.depth_path))
    ori_rgb = np.array(Image.open(args.rgb_path)) / 255.0
    ori_depth = np.clip(ori_depth, 0, 1000)
    ori_rgb = torch.from_numpy(ori_rgb).permute(2, 1, 0)[None]
    ori_rgb = ori_rgb.to(device='cuda', dtype=torch.float32)
    ori_depth = torch.from_numpy(ori_depth).T[None]
    ori_depth = ori_depth.to(device='cuda', dtype=torch.float32)

    # get scene points
    view_points, _, _ = pc_helper.to_scene_points(ori_rgb,
                                                  ori_depth,
                                                  include_rgb=True)
    # get xyz maps
    xyzs = pc_helper.to_xyz_maps(ori_depth)

    # pre-process
    rgb = F.interpolate(ori_rgb, (args.input_w, args.input_h))
    depth = F.interpolate(ori_depth[None], (args.input_w, args.input_h))[0]
    depth = depth / 1000.0
    depth = torch.clip((depth - depth.mean()), -1, 1)
    # generate 2d input
    x = torch.concat([depth[None], rgb], 1)
    x = x.to(device='cuda', dtype=torch.float32)

    # inference
    pred_gg = inference(view_points,
                        xyzs,
                        x,
                        ori_depth,
                        vis_heatmap=True,
                        vis_grasp=True)

    # time test
    start = time()
    T = 100
    for _ in range(T):
        pred_gg = inference(view_points,
                            xyzs,
                            x,
                            ori_depth,
                            vis_heatmap=False,
                            vis_grasp=False)
        torch.cuda.synchronize()
    print('avg time ==', (time() - start) / T * 1e3, 'ms')
