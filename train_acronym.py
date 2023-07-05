import itertools
import logging
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from matplotlib import pyplot as plt
from PIL import Image
from torchsummary import summary
from tqdm import tqdm

from acronym_eval_tools.evaluation import eval_validate
from dataset.acronym_dataset import AcronymPointDataset
from dataset.evaluation import (anchor_output_process, calculate_6d_match,
                                calculate_coverage, calculate_iou_match,
                                collision_detect, detect_2d_grasp,
                                detect_6d_grasp_multi)
from dataset.grasp import RectGraspGroup
from dataset.pc_dataset_tools import (data_process, feature_fusion,
                                      get_center_group_label,
                                      get_ori_grasp_label)
from dataset.utils import shift_anchors
from models.anchornet import AnchorGraspNet, BNMomentumScheduler
from models.localgraspnet import PointMultiGraspNet
from models.losses import compute_anchor_loss, compute_multicls_loss
from train_utils import *

dis_criterion = 0.05
rot_criterion = 0.25


def validate(epoch, anchornet: nn.Module, localnet: nn.Module,
             val_data: torch.utils.data.DataLoader, anchors: dict, args):
    fixed_center_num = 32

    # network eval mode
    anchornet.eval()
    localnet.eval()
    # stop augmentation for validation
    val_data.dataset.eval()

    results = {
        'correct': 0,
        'total': 0,
        'loss': 0,
        'losses': {},
        'multi_cls_loss': 0,
        'offset_loss': 0,
        'anchor_loss': 0,
        'vgr': 0,
        'score': 0,
        'grasp_nocoll_view_num': 0,
        'cover_cnt': 0,
        'label_cnt': 0
    }
    valid_center_num, total_center_num = 0, 0
    for scale_factor in eval_scale:
        thre_dis = dis_criterion * scale_factor
        thre_rot = rot_criterion * scale_factor
        results[f'grasp_{scale_factor}'] = np.zeros((2, ))
        results[f'trans_{thre_dis}'] = np.zeros((2, ))
        results[f'rot_{thre_rot}'] = np.zeros((2, ))
    time_2d, time_data, time_6d, time_colli = 0, 0, 0, 0

    batch_idx = -1
    vis_id = np.random.randint(0, len(val_data))
    with torch.no_grad():
        for anchor_data, rgb, depth, pc_path, grasppaths, camera_pose in tqdm(
                val_data):
            batch_idx += 1
            # get scene points
            points, _, _ = val_data.dataset.helper.to_scene_points(
                rgb.cuda(), depth.cuda(), include_rgb=False)
            # get xyz maps
            xyzs = val_data.dataset.helper.to_xyz_maps(depth.cuda())
            # get labels
            gg_ori_labels = get_ori_grasp_label(grasppaths)
            all_grasp_labels = []
            for grasppath in grasppaths:
                all_grasp_labels.append(np.load(grasppath))

            # get time
            start = time()

            # 2d prediction
            x, y, _, _, _ = anchor_data

            x = x.cuda(non_blocking=True)
            target = [yy.cuda(non_blocking=True) for yy in y]
            pred_2d, perpoint_features = anchornet(x)

            loc_map, cls_mask, theta_offset, depth_offset, width_offset = \
                anchor_output_process(*pred_2d)

            # detect 2d grasp (x, y, theta)
            rect_gg = detect_2d_grasp(loc_map,
                                      cls_mask,
                                      theta_offset,
                                      depth_offset,
                                      width_offset,
                                      ratio=args.ratio,
                                      anchor_k=args.anchor_k,
                                      anchor_w=args.anchor_w,
                                      anchor_z=args.anchor_z,
                                      mask_thre=args.heatmap_thres,
                                      center_num=fixed_center_num,
                                      grid_size=args.grid_size,
                                      grasp_nms=args.grid_size)

            # get 2d time
            torch.cuda.synchronize()
            time_2d += time() - start

            # cal loss
            anchor_lossd = compute_anchor_loss(pred_2d,
                                               target,
                                               loc_a=args.loc_a,
                                               reg_b=args.reg_b,
                                               cls_c=args.cls_c)
            anchor_losses = anchor_lossd['losses']
            anchor_loss = anchor_lossd['loss']

            # convert back to np.array, rot should be 0, zoom should be 1
            idx = anchor_data[2].numpy().squeeze()
            rot = anchor_data[3].numpy().squeeze()
            zoom_factor = anchor_data[4].numpy().squeeze()

            # 2d bbox validation
            grasp_label = val_data.dataset.load_grasp_labels(idx)
            gt_rect_gg = RectGraspGroup()
            gt_rect_gg.load_from_dict(grasp_label)
            gt_bbs = val_data.dataset.get_gtbb(gt_rect_gg, rot, zoom_factor)

            if batch_idx == vis_id:
                rgb_t = rgb.cpu().numpy().squeeze().transpose(2, 1, 0)
                rect_rgb = gt_rect_gg.plot_rect_grasp_group(rgb_t, 0)
                plt.subplot(221)
                plt.imshow(rect_rgb)
                plt.subplot(222)
                plt.imshow(anchor_data[1][0].cpu().numpy().squeeze().T,
                           cmap='jet')
                plt.subplot(223)
                plt.imshow(loc_map.squeeze().T, cmap='jet')
                plt.subplot(224)
                resized_rgb = Image.fromarray(
                    (rgb_t * 255.0).astype(np.uint8)).resize((640, 360))
                rect_rgb = rect_gg.plot_rect_grasp_group(
                    np.array(resized_rgb), 0)
                plt.imshow(rect_rgb)
                plt.tight_layout()
                plt.savefig('heatmap.png', dpi=400)
                # plt.show()

            # cal 2d iou
            s = calculate_iou_match(rect_gg[0:1], gt_bbs, thre=0.25)
            results['correct'] += s
            results['total'] += 1

            multi_cls_loss = 0
            offset_loss = 0
            if epoch >= args.pre_epochs:
                # check 2d result
                if rect_gg.size == 0:
                    print('No 2d grasp found')
                    continue

                # get data time
                start = time()

                # feature fusion using knn and max pooling
                points_all = feature_fusion(points, perpoint_features, xyzs)
                rect_ggs = [rect_gg]
                pc_group, valid_local_centers = data_process(
                    points_all,
                    depth.cuda(),
                    rect_ggs,
                    args.center_num,
                    args.group_num, (args.input_w, args.input_h),
                    min_points=32)
                rect_gg = rect_ggs[0]
                # batch_size == 1 when valid
                points_all = points_all.squeeze()

                # check pc_group
                if pc_group.shape[0] == 0:
                    print('No partial point clouds')
                    continue

                # get 2d grasp info (not grasp itself) for trainning
                grasp_info = np.zeros((0, 3), dtype=np.float32)
                g_thetas = rect_gg.thetas[None]
                g_ws = rect_gg.widths[None]
                g_ds = rect_gg.depths[None]
                cur_info = np.vstack([g_thetas, g_ws, g_ds])
                grasp_info = np.vstack([grasp_info, cur_info.T])
                grasp_info = torch.from_numpy(grasp_info).to(
                    dtype=torch.float32, device='cuda')

                # get data time
                torch.cuda.synchronize()
                time_data += time() - start
                start = time()

                # get gamma and beta classification result, padding for benchmark
                zero_pad_num = fixed_center_num - pc_group.shape[0]
                pc_group = torch.concat([
                    pc_group,
                    torch.zeros(zero_pad_num,
                                pc_group.shape[1],
                                pc_group.shape[2],
                                device='cuda')
                ])
                grasp_info = torch.concat([
                    grasp_info,
                    torch.zeros(zero_pad_num,
                                grasp_info.shape[1],
                                device='cuda')
                ])
                _, pred_view, offset = localnet(pc_group, grasp_info)
                valid_num = fixed_center_num - zero_pad_num
                pc_group = pc_group[:valid_num]
                pred_view = pred_view[:valid_num]
                offset = offset[:valid_num]

                # detect 6d grasp from 2d output and 6d output
                pred_grasp, pred_rect_gg = detect_6d_grasp_multi(
                    rect_gg,
                    pred_view,
                    offset,
                    valid_local_centers, (args.input_w, args.input_h),
                    anchors,
                    k=args.local_k)
                pred_grasp = torch.from_numpy(pred_grasp).to(
                    dtype=torch.float32, device='cuda')

                # get 6d time
                torch.cuda.synchronize()
                time_6d += time() - start

                # get nearest grasp labels
                gg_labels, _ = get_center_group_label(valid_local_centers,
                                                      all_grasp_labels,
                                                      args.local_grasp_num)
                # get center valid stats
                total_center_num += len(gg_labels)
                for gg in gg_labels:
                    valid_center_num += len(gg) > 0

                # get loss
                multi_cls_loss, offset_loss = compute_multicls_loss(
                    pred_view, offset, gg_labels, grasp_info, anchors, args)

                # get collision detect time
                start = time()

                # collision detect
                pred_grasp_from_rect = pred_rect_gg.to_6d_grasp_group(
                    depth=0.025)
                pred_gg, valid_mask = collision_detect(points_all,
                                                       pred_grasp_from_rect)
                pred_grasp = pred_grasp[valid_mask]

                # get collision detect time
                torch.cuda.synchronize()
                time_colli += time() - start

                # transfer grasp to (num, 4, 4)
                grasps = np.ones((len(pred_gg), 4, 4))
                grasps[:, :3, 3] = pred_gg.translations
                grasps[:, :3, :3] = pred_gg.rotations

                # regnet validation
                vgr, score, no_colli_idxs, grasp_nocoll_view_num, _, _ = eval_validate(
                    pc_path,
                    grasps,
                    camera_pose,
                    table_height=0.0,
                    depths=pred_gg.depths,
                    widths=pred_gg.widths,
                    colli_detect=False)

                if len(grasps) == 0:
                    print('no 6d grasp')
                else:
                    results['vgr'] += vgr
                    results['score'] += score
                    results['grasp_nocoll_view_num'] += grasp_nocoll_view_num

                # if batch_idx == vis_id:
                #     print('rect gg ==', len(pred_rect_gg))
                #     grasp_geo = pred_gg.to_open3d_geometry_list()
                #     cloud = points_all[:, :3].cpu().numpy()
                #     rgb = points_all[:, 3:6].cpu().numpy()
                #     vispc = o3d.geometry.PointCloud()
                #     vispc.points = o3d.utility.Vector3dVector(cloud)
                #     vispc.colors = o3d.utility.Vector3dVector(rgb)
                #     o3d.visualization.draw_geometries([vispc] + grasp_geo)

                # cal 6d match
                gg_ori_labels = get_ori_grasp_label(grasppaths)
                for scale_factor in eval_scale:
                    thre_dis = dis_criterion * scale_factor
                    thre_rot = rot_criterion * scale_factor
                    r_g, r_d, r_r = calculate_6d_match(pred_grasp,
                                                       gg_ori_labels,
                                                       threshold_dis=thre_dis,
                                                       threshold_rot=thre_rot)

                    results[f'grasp_{scale_factor}'] += r_g
                    results[f'trans_{thre_dis}'] += r_d
                    results[f'rot_{thre_rot}'] += r_r

                # cal coverage rate
                cover_cnt = calculate_coverage(pred_grasp[no_colli_idxs],
                                               gg_ori_labels)
                results['cover_cnt'] += cover_cnt
                results['label_cnt'] += len(gg_ori_labels)

            # tensorboard record
            results['loss'] += anchor_loss.item() + multi_cls_loss.item(
            ) + offset_loss.item()
            results['anchor_loss'] += anchor_loss.item()
            if epoch >= args.pre_epochs:
                results['multi_cls_loss'] += multi_cls_loss.item()
                results['offset_loss'] += offset_loss.item()
            for ln, l in anchor_losses.items():
                if ln not in results['losses']:
                    results['losses'][ln] = 0
                results['losses'][ln] += l.item()

    batch_idx += 1
    # time stat
    time_2d = time_2d / batch_idx * 1000
    time_data = time_data / batch_idx * 1000
    time_6d = time_6d / batch_idx * 1000
    time_colli = time_colli / batch_idx * 1000
    logging.info('Time stats:')
    logging.info(
        f'2d: {time_2d:.3f} ms  data: {time_data:.3f} ms  6d: {time_6d:.3f} ms  colli: {time_colli:.3f} ms'
    )

    # center stat
    if total_center_num > 0:
        logging.info(
            f'valid center == {valid_center_num / total_center_num:.2f}')

    # loss stat
    results['loss'] /= batch_idx
    results['anchor_loss'] /= batch_idx
    results['multi_cls_loss'] /= batch_idx
    results['offset_loss'] /= batch_idx
    for ln, l in anchor_losses.items():
        results['losses'][ln] /= batch_idx
    return results


def train(epoch, anchornet: nn.Module, localnet: nn.Module,
          train_data: torch.utils.data.DataLoader, optimizer: optim.AdamW,
          anchors: dict, args):
    results = {
        'loss': 0,
        'losses': {},
        'multi_cls_loss': 0,
        'offset_loss': 0,
        'anchor_loss': 0
    }
    valid_center_num, total_center_num = 0, 0

    optimizer.zero_grad()
    anchornet.train()
    localnet.train()

    if args.joint_trainning:
        train_data.dataset.unaug()
    else:
        if epoch >= args.pre_epochs:
            print('Attention: freeze anchor net!')
            anchornet.eval()
            for para in anchornet.parameters():
                para.requires_grad_(False)
            train_data.dataset.unaug()
        else:
            # extra aug for 2d net
            print('Extra augmentation for 2d network trainning!')
            train_data.dataset.setaug()

    # augment for trainning
    train_data.dataset.train()

    # log loss stat
    start = time()
    batch_idx = -1
    sum_local_loss = 0
    sum_offset_loss = 0
    sum_anchor_loss = 0
    sum_anchor_loss_d = {'loc_map_loss': 0, 'reg_loss': 0, 'cls_loss': 0}

    # for anchor shift
    cur_labels = torch.zeros((0, 8), dtype=torch.float32)

    data_start = time()
    data_time = 0
    for anchor_data, rgbs, depths, pc_path, grasppaths, camera_pose in train_data:
        if len(rgbs) < args.batch_size:
            continue
        data_time += time() - data_start
        batch_idx += 1

        # muli-step trainning for every epoch
        # get scene points
        points, _, _ = train_data.dataset.helper.to_scene_points(
            rgbs.cuda(), depths.cuda(), include_rgb=False)

        # get xyz maps
        xyzs = train_data.dataset.helper.to_xyz_maps(depths)
        # get labels
        all_grasp_labels = []
        for grasppath in grasppaths:
            all_grasp_labels.append(np.load(grasppath))

        # train anchornet first
        x, y, _, _, _ = anchor_data

        x = x.cuda(non_blocking=True)
        target = [yy.cuda(non_blocking=True) for yy in y]
        pred_2d, perpoint_features = anchornet(x)
        # cal loss
        anchor_lossd = compute_anchor_loss(pred_2d,
                                           target,
                                           loc_a=args.loc_a,
                                           reg_b=args.reg_b,
                                           cls_c=args.cls_c)
        anchor_losses = anchor_lossd['losses']
        anchor_loss = anchor_lossd['loss']

        # get loss stat
        if args.joint_trainning or epoch < args.pre_epochs:
            loss = anchor_loss
        else:
            loss = 0

        if epoch >= args.pre_epochs:
            # detect 2d grasp center
            loc_maps, theta_cls, theta_offset, depth_offset, width_offset = \
                    anchor_output_process(*pred_2d)

            # detect 2d grasp (x, y, theta)
            rect_ggs = []
            for i in range(args.batch_size):
                rect_gg = detect_2d_grasp(loc_maps[i],
                                          theta_cls[i],
                                          theta_offset[i],
                                          depth_offset[i],
                                          width_offset[i],
                                          ratio=args.ratio,
                                          anchor_k=args.anchor_k,
                                          anchor_w=args.anchor_w,
                                          anchor_z=args.anchor_z,
                                          mask_thre=0,
                                          center_num=args.center_num,
                                          grid_size=args.grid_size,
                                          grasp_nms=args.grid_size)
                rect_ggs.append(rect_gg)

            if len(rect_ggs) == 0:
                print('No 2d grasp found')
                continue

            # feature fusion using knn and max pooling
            points_all = feature_fusion(points, perpoint_features, xyzs)
            # using 2d grasp to crop point cloud
            pc_group, valid_local_centers = data_process(
                points_all, depths.cuda(), rect_ggs, args.center_num,
                args.group_num, (args.input_w, args.input_h))

            # get 2d grasp info (not grasp itself) for trainning
            grasp_info = np.zeros((0, 3), dtype=np.float32)
            for i in range(args.batch_size):
                g_thetas = rect_ggs[i].thetas[None]
                g_ws = rect_ggs[i].widths[None]
                g_ds = rect_ggs[i].depths[None]
                cur_info = np.vstack([g_thetas, g_ws, g_ds])
                grasp_info = np.vstack([grasp_info, cur_info.T])
            grasp_info = torch.from_numpy(grasp_info).to(dtype=torch.float32,
                                                         device='cuda')

            # check pc_group
            if pc_group.shape[0] == 0:
                print('No partial point clouds')
                continue

            # local net, padding for benchmark
            zero_pad_num = args.batch_size * args.center_num - pc_group.shape[0]
            pc_group = torch.concat([
                pc_group,
                torch.zeros(zero_pad_num,
                            pc_group.shape[1],
                            pc_group.shape[2],
                            device='cuda')
            ])
            grasp_info = torch.concat([
                grasp_info,
                torch.zeros(zero_pad_num, grasp_info.shape[1], device='cuda')
            ])
            _, pred_view, offset = localnet(pc_group, grasp_info)
            valid_num = args.batch_size * args.center_num - zero_pad_num
            pred_view = pred_view[:valid_num]
            offset = offset[:valid_num]

            # get nearest grasp labels
            gg_labels, total_labels = get_center_group_label(
                valid_local_centers, all_grasp_labels, args.local_grasp_num)

            # get center valid stats
            total_center_num += len(gg_labels)
            for gg in gg_labels:
                valid_center_num += len(gg) > 0

            # shift anchors only for first serveral epochs
            if epoch < args.shift_epoch:
                cur_labels = torch.cat([cur_labels, total_labels.cpu()], 0)
                if len(cur_labels) > 1e6:
                    shift_start = time()
                    old_gammas = anchors['gamma'].clone()
                    old_betas = anchors['beta'].clone()
                    anchors = shift_anchors(cur_labels, anchors)
                    # get shift error
                    error = (old_gammas - anchors['gamma']).abs().sum()
                    error += (old_betas - anchors['beta']).abs().sum()
                    logging.info(f'shift error == {error:.5f}')
                    logging.info(f'shift time == {time() - shift_start:.3f}')
                    cur_labels = torch.zeros((0, 8), dtype=torch.float32)
                    # stop when stable
                    # if error < 1e-2:
                    #     shift_epoch = 0

            # get loss
            multi_cls_loss, offset_loss = compute_multicls_loss(
                pred_view, offset, gg_labels, grasp_info, anchors, args)
            loss += multi_cls_loss + offset_loss

        # backward every step
        loss.backward()

        # step sum loss
        if batch_idx > 0 and batch_idx % args.step_cnt == 0:
            nn.utils.clip_grad.clip_grad_value_(anchornet.parameters(), 1)
            nn.utils.clip_grad.clip_grad_value_(localnet.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()

        # get accumulation loss (for log_batch_cnt)
        sum_anchor_loss += anchor_loss.item()
        if epoch >= args.pre_epochs:
            sum_local_loss += multi_cls_loss.item()
            sum_offset_loss += offset_loss.item()
        for key in anchor_losses:
            sum_anchor_loss_d[key] += anchor_losses[key].item()

        log_batch_cnt = 800 // args.batch_size
        if batch_idx > 0 and batch_idx % log_batch_cnt == 0:
            logging.info(
                f'{log_batch_cnt} batches using time: {time() - start:.2f} s  data time: {data_time:.2f} s'
            )
            data_time = 0
            start = time()
            # print loss stat
            log_anchor_loss(epoch, batch_idx,
                            sum_anchor_loss + sum_local_loss + sum_offset_loss,
                            sum_anchor_loss, sum_anchor_loss_d, log_batch_cnt)
            if epoch >= args.pre_epochs:
                logging.info(
                    f'multi_cls_loss: {sum_local_loss / log_batch_cnt:.4f}')
                logging.info(
                    f'offset_loss: {sum_offset_loss / log_batch_cnt:.4f}')
                logging.info(
                    f'valid center == {valid_center_num / total_center_num:.2f}'
                )
            # reset loss stat
            valid_center_num, total_center_num = 0, 0
            sum_local_loss = 0
            sum_anchor_loss = 0
            sum_offset_loss = 0
            sum_anchor_loss_d = {
                'loc_map_loss': 0,
                'reg_loss': 0,
                'cls_loss': 0
            }

        # train result update
        results['loss'] += anchor_loss.item()
        if epoch >= args.pre_epochs:
            results['loss'] += multi_cls_loss.item() + offset_loss.item()
        results['anchor_loss'] += anchor_loss.item()
        for key, value in anchor_losses.items():
            if key not in results['losses']:
                results['losses'][key] = 0
            results['losses'][key] += value.item()
        if epoch >= args.pre_epochs:
            results['multi_cls_loss'] += multi_cls_loss.item()
            results['offset_loss'] += offset_loss.item()

        data_start = time()

    # loss stat
    batch_idx += 1
    results['loss'] /= batch_idx
    results['anchor_loss'] /= batch_idx
    for key in results['losses']:
        results['losses'][key] /= batch_idx
    if epoch >= args.pre_epochs:
        results['multi_cls_loss'] /= batch_idx
        results['offset_loss'] /= batch_idx
    return results


def run():
    args = parse_args()

    # prepare for trainning
    tb, save_folder = prepare_torch_and_logger(args)

    # Load Dataset
    logging.info('Loading Dataset...')
    sceneIds = list(range(args.scene_l, args.scene_r))
    Dataset = AcronymPointDataset(args.all_points_num,
                                  args.dataset_path,
                                  noise=args.noise,
                                  sceneIds=sceneIds,
                                  sigma=args.sigma,
                                  ratio=args.ratio,
                                  anchor_k=args.anchor_k,
                                  anchor_z=args.anchor_z,
                                  anchor_w=args.anchor_w,
                                  grasp_count=args.grasp_count,
                                  output_size=(args.input_w, args.input_h),
                                  random_rotate=False,
                                  random_zoom=False)
    Val_Dataset = AcronymPointDataset(args.all_points_num,
                                      args.dataset_path,
                                      noise=args.noise,
                                      sceneIds=list(range(400, 500)),
                                      sigma=args.sigma,
                                      ratio=args.ratio,
                                      anchor_k=args.anchor_k,
                                      anchor_z=args.anchor_z,
                                      anchor_w=args.anchor_w,
                                      grasp_count=args.grasp_count,
                                      output_size=(args.input_w, args.input_h),
                                      random_rotate=False,
                                      random_zoom=False,
                                      anno_cnt=2)

    logging.info('Training size: {}'.format(len(Dataset)))
    logging.info('Validation size: {}'.format(len(Val_Dataset)))

    train_data = torch.utils.data.DataLoader(Dataset,
                                             batch_size=args.batch_size,
                                             num_workers=args.num_workers,
                                             shuffle=True)
    val_data = torch.utils.data.DataLoader(Val_Dataset,
                                           num_workers=1,
                                           batch_size=1)

    # Load the network
    logging.info('Loading Network...')
    input_channels = 1 * args.use_depth + 3 * args.use_rgb
    anchornet = AnchorGraspNet(ratio=args.ratio,
                               in_dim=input_channels,
                               anchor_k=args.anchor_k)
    localnet = PointMultiGraspNet(3, args.anchor_num**2)

    # load checkpoint
    basic_ranges = torch.linspace(-1, 1, args.anchor_num + 1).cuda()
    basic_anchors = (basic_ranges[1:] + basic_ranges[:-1]) / 2
    anchors = {'gamma': basic_anchors, 'beta': basic_anchors}
    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint)
        if 'gamma' in ckpt and len(ckpt['gamma']) == args.anchor_num:
            anchors['gamma'] = ckpt['gamma']
            anchors['beta'] = ckpt['beta']
            logging.info('Using saved anchors')
        print('Local heatmap model from checkpoint only!')
        anchornet.load_state_dict(ckpt['anchor'])
        # localnet.load_state_dict(ckpt['local'])

    # set optimizer
    params = itertools.chain(anchornet.parameters(), localnet.parameters())
    optimizer = get_optimizer(args, params)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 5, 0.1)

    # Decay Batchnorm momentum from 0.5 to 0.999
    # note: pytorch's BN momentum (default 0.1)= 1 - tensorflow's BN momentum
    BN_MOMENTUM_INIT = 0.5
    BN_MOMENTUM_MAX = 0.001
    bn_lbmd = lambda it: max(BN_MOMENTUM_INIT * 0.5**
                             (int(it / 2)), BN_MOMENTUM_MAX)
    bnm_scheduler = BNMomentumScheduler(anchornet,
                                        bn_lambda=bn_lbmd,
                                        last_epoch=-1)

    # get model architecture
    # print_model(args, input_channels, anchornet, save_folder)

    # multi gpu
    anchornet = nn.parallel.DataParallel(anchornet).cuda()
    localnet = nn.parallel.DataParallel(localnet).cuda()
    logging.info('Done')

    for epoch in range(args.epochs):
        logging.info('Beginning Epoch {:02d}'.format(epoch))
        train_results = train(epoch, anchornet, localnet, train_data,
                              optimizer, anchors, args)
        scheduler.step()
        bnm_scheduler.step()

        # Log training losses to tensorboard
        tb.add_scalar('train_loss/loss', train_results['loss'], epoch)
        tb.add_scalar('train_loss/anchor_loss', train_results['anchor_loss'],
                      epoch)
        for n, l in train_results['losses'].items():
            tb.add_scalar('train_loss/' + n, l, epoch)
        if epoch >= args.pre_epochs:
            tb.add_scalar('train_loss/multi_cls_loss',
                          train_results['multi_cls_loss'], epoch)
            tb.add_scalar('train_loss/offset_loss',
                          train_results['offset_loss'], epoch)

        # Run Validation
        logging.info('Validating...')
        val_results = validate(epoch, anchornet, localnet, val_data, anchors,
                               args)

        if epoch >= args.pre_epochs:
            log_match_result(val_results, dis_criterion, rot_criterion)

        log_and_save(args, tb, val_results, epoch, anchornet, localnet,
                     optimizer, anchors, save_folder)


if __name__ == '__main__':
    run()
