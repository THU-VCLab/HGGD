import argparse
import os
import random
from time import time

import cupoch
import numpy as np
import open3d as o3d
import torch
from scipy.signal import medfilt2d
from torch.utils.data import DataLoader

from customgraspnetAPI import Grasp as GraspNetGrasp
from customgraspnetAPI import GraspGroup as GraspNetGraspGroup
from customgraspnetAPI import GraspNetEval
from dataset.config import camera
from dataset.evaluation import (anchor_output_process, collision_detect,
                                detect_2d_grasp, detect_6d_grasp_multi)
from dataset.grasp import GraspGroup as HGGDGraspGroup
from dataset.grasp import RectGraspGroup
from dataset.graspnet_dataset import GraspnetPointDataset
from dataset.pc_dataset_tools import data_process, feature_fusion
from models.anchornet import AnchorGraspNet
from models.localgraspnet import PointMultiGraspNet
from train_utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint-path', default=None)

# dataset
parser.add_argument('--dataset-path')
parser.add_argument('--scene-path')
parser.add_argument('--scene-l', type=int)
parser.add_argument('--scene-r', type=int)
parser.add_argument('--grasp-count', type=int, default=5000)
parser.add_argument('--dump-dir',
                    help='Dump dir to save outputs',
                    default='./pred/test')
parser.add_argument('--num-workers', type=int, default=4)

# 2d
parser.add_argument('--input-h', type=int)
parser.add_argument('--input-w', type=int)
parser.add_argument('--sigma', type=int, default=10)
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
parser.add_argument('--local-grasp-num', type=int, default=5000)

# grasp detection
parser.add_argument('--heatmap-thres', type=float, default=0.01)
parser.add_argument('--local-k', type=int, default=10)
parser.add_argument('--local-thres', type=float, default=0.01)
parser.add_argument('--rotation-num', type=int, default=1)

# others
parser.add_argument('--logdir',
                    type=str,
                    default='./logs/',
                    help='Log directory')
parser.add_argument('--random-seed', type=int, default=123, help='Random seed')
parser.add_argument('--description',
                    type=str,
                    default='',
                    help='Logging description')

args = parser.parse_args()


def inference():
    sceneIds = list(range(args.scene_l, args.scene_r))
    # Create Dataset and Dataloader
    test_dataset = GraspnetPointDataset(args.all_points_num,
                                        args.dataset_path,
                                        args.scene_path,
                                        sceneIds,
                                        sigma=args.sigma,
                                        ratio=args.ratio,
                                        anchor_k=args.anchor_k,
                                        anchor_z=args.anchor_z,
                                        anchor_w=args.anchor_w,
                                        grasp_count=args.grasp_count,
                                        output_size=(args.input_w,
                                                     args.input_h),
                                        random_rotate=False,
                                        random_zoom=False)

    SCENE_LIST = test_dataset.scene_list()
    test_data = DataLoader(test_dataset,
                           batch_size=1,
                           pin_memory=True,
                           num_workers=1)
    test_data.dataset.unaug()
    test_data.dataset.eval()

    # Init the model
    input_channels = 4
    anchornet = AnchorGraspNet(in_dim=input_channels,
                               ratio=args.ratio,
                               anchor_k=args.anchor_k)
    localnet = PointMultiGraspNet(info_size=3, k_cls=args.anchor_num**2)

    # multi gpu
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
    # stop rot and zoom for validation
    test_dataset.eval()

    time_2d, time_data, time_6d, time_colli, time_nms = 0, 0, 0, 0, 0

    batch_idx = -1
    vis_id = []
    with torch.no_grad():
        for anchor_data, rgb, ori_depth, grasppaths in test_data:
            batch_idx += 1

            # medfilt first
            depth = ori_depth.numpy().squeeze()
            # depth = medfilt2d(depth, kernel_size=5)
            depth = torch.from_numpy(depth).float()[None]
            # get scene points
            view_points, _, _ = test_data.dataset.helper.to_scene_points(
                rgb.cuda(), depth.cuda(), include_rgb=True)
            points = view_points[..., :3]
            view_points = view_points.squeeze()
            # get xyz maps
            xyzs = test_data.dataset.helper.to_xyz_maps(depth.cuda())
            # get labels
            all_grasp_labels = []
            for grasppath in grasppaths:
                all_grasp_labels.append(np.load(grasppath))

            # get time
            start = time()

            # 2d prediction
            x, _, _, _, _ = anchor_data
            x = x.cuda(non_blocking=True)
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

            # get 2d time
            if batch_idx >= 1:
                torch.cuda.synchronize()
                time_2d += time() - start

            # 2d bbox validation
            idx = anchor_data[2].numpy().squeeze()
            grasp_label = test_data.dataset.load_grasp_labels(idx)
            gt_rect_gg = RectGraspGroup()
            gt_rect_gg.load_from_dict(grasp_label)

            # check 2d result
            if rect_gg.size == 0:
                print('No 2d grasp found')
                continue

            # data time
            start = time()

            # feature fusion
            points_all = feature_fusion(points, perpoint_features, xyzs)
            rect_ggs = [rect_gg]
            pc_group, valid_local_centers = data_process(
                points_all,
                depth.cuda(),
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

            # get data time
            if batch_idx >= 1:
                torch.cuda.synchronize()
                time_data += time() - start
            start = time()

            # get gamma and beta classification result
            _, pred, offset = localnet(pc_group, grasp_info)

            # detect 6d grasp from 2d output and 6d output
            pred_grasp, pred_rect_gg = detect_6d_grasp_multi(
                rect_gg,
                pred,
                offset,
                valid_local_centers, (args.input_w, args.input_h),
                anchors,
                k=args.local_k)

            # get 6d time
            if batch_idx >= 1:
                torch.cuda.synchronize()
                time_6d += time() - start

            # get collision detect time
            start = time()

            # collision detect
            pred_grasp_from_rect = pred_rect_gg.to_6d_grasp_group(depth=0.02)
            pred_gg, valid_mask = collision_detect(points_all,
                                                   pred_grasp_from_rect,
                                                   mode='graspnet')
            # pred_grasp = pred_grasp[valid_mask]

            # get collision detect
            if batch_idx >= 1:
                torch.cuda.synchronize()
                time_colli += time() - start
            start = time()

            # Dump results for evaluation
            gg = GraspNetGraspGroup()
            for pred_g in pred_gg:
                g = GraspNetGrasp(pred_g.score, pred_g.width, pred_g.height,
                                  pred_g.depth, pred_g.rotation.reshape(9, ),
                                  pred_g.translation, -1)
                gg.add(g)
            gg = gg.nms(0.03, 30 / 180 * np.pi)
            gg = gg.sort_by_score()

            if batch_idx >= 1:
                torch.cuda.synchronize()
                time_nms += time() - start

            pred_gg = HGGDGraspGroup(translations=gg.translations,
                                     rotations=gg.rotation_matrices,
                                     heights=gg.heights,
                                     widths=gg.widths,
                                     depths=gg.depths,
                                     scores=gg.scores,
                                     object_ids=gg.object_ids)

            if batch_idx in vis_id:
                print('pred gg ==', len(pred_gg))
                grasp_geo = pred_gg.to_open3d_geometry_list()
                cloud = view_points[:, :3].cpu().numpy()
                rgb = view_points[:, 3:6].cpu().numpy()
                vispc = o3d.geometry.PointCloud()
                vispc.points = o3d.utility.Vector3dVector(cloud)
                vispc.colors = o3d.utility.Vector3dVector(rgb)
                o3d.visualization.draw_geometries([vispc] + grasp_geo)

            log_str = f'{batch_idx // 256}:{batch_idx % 256}  pred 2d num: {rect_gg.size}'
            log_str += f'\npred 6d num: {len(pred_grasp)}  no-colli: {valid_mask.sum()}  nms: {len(gg)}'
            print(log_str)

            # save grasps
            save_dir = os.path.join(args.dump_dir, SCENE_LIST[batch_idx])
            save_dir = os.path.join(save_dir, camera)
            save_path = os.path.join(save_dir,
                                     str(batch_idx % 256).zfill(4) + '.npy')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            gg.save_npy(save_path)

    # time stat
    batch_idx -= 1
    time_2d = time_2d / batch_idx * 1000
    time_data = time_data / batch_idx * 1000
    time_6d = time_6d / batch_idx * 1000
    time_colli = time_colli / batch_idx * 1000
    time_nms = time_nms / batch_idx * 1000
    logging.info('Time stats:')
    logging.info(
        f'Total: {time_2d + time_data + time_6d + time_colli + time_nms:.3f} ms'
    )
    logging.info(
        f'2d: {time_2d:.3f} ms  data: {time_data:.3f} ms  6d: {time_6d:.3f} ms  colli: {time_colli:.3f} ms  nms: {time_nms:.3f} ms'
    )


def evaluate():
    # res = np.load('temp_result.npy')
    ge = GraspNetEval(root=args.scene_path,
                      camera=camera,
                      split=(args.scene_l, args.scene_r))
    res, ap, colli = ge.eval_scene_lr(args.dump_dir,
                                      args.scene_l,
                                      args.scene_r,
                                      proc=args.num_workers)
    np.save('temp_result.npy', res)
    # get ap 0.8 and ap 0.4
    aps = res.mean(0).mean(0).mean(0)
    logging.info(f'Scene: {args.scene_l} ~ {args.scene_r}')
    logging.info(f'colli == {colli}')
    logging.info(f'ap == {ap}')
    logging.info(f'ap0.8 == {aps[3]}')
    logging.info(f'ap0.4 == {aps[1]}')


if __name__ == '__main__':
    # multiprocess
    # mp.set_start_method('spawn')
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

    # Set-up output directories
    net_desc = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
    net_desc = net_desc + '_' + args.description
    net_desc = 'graspnet_test' + net_desc

    save_folder = os.path.join(args.logdir, net_desc)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Save commandline args
    if args is not None:
        params_path = os.path.join(save_folder, 'commandline_args.json')
        with open(params_path, 'w') as f:
            json.dump(vars(args), f)

    # Initialize logging
    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        filename='{0}/{1}.log'.format(save_folder, 'log'),
        format=
        '[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
        datefmt='%H:%M:%S')
    # set up logging to console
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

    inference()
    evaluate()
