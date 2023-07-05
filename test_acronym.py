import logging
import random
from time import time

import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import torch.utils.data
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm

from acronym_eval_tools.evaluation import eval_validate
from dataset.acronym_dataset import AcronymPointDataset
from dataset.evaluation import (anchor_output_process, calculate_6d_match,
                                calculate_coverage,
                                calculate_guidance_accuracy,
                                calculate_iou_match, collision_detect,
                                detect_2d_grasp, detect_6d_grasp_multi)
from dataset.grasp import GraspGroup, RectGraspGroup
from dataset.pc_dataset_tools import (concat_feature_fusion, data_process,
                                      feature_fusion, get_center_group_label,
                                      get_ori_grasp_label)
from models.anchornet import AnchorGraspNet
from models.localgraspnet import PointMultiGraspNet
from models.losses import compute_multicls_loss
from train_utils import *

dis_criterion = 0.05
rot_criterion = 0.25


def validate(epoch, anchornet: nn.Module, localnet: nn.Module,
             val_data: torch.utils.data.DataLoader, anchors: dict, args):
    """Run validation.

    :param net: Network
    :param val_data: Validation Dataset
    :return: Successes, Failures and Losses
    """
    # network eval mode
    anchornet.eval()
    localnet.eval()
    # stop rot and zoom for validation
    val_data.dataset.eval()

    # using benchmark
    torch.backends.cudnn.benchmark = True

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
    for scale_factor in eval_scale:
        thre_dis = dis_criterion * scale_factor
        thre_rot = rot_criterion * scale_factor
        results[f'grasp_{scale_factor}'] = np.zeros((2, ))
        results[f'trans_{thre_dis}'] = np.zeros((2, ))
        results[f'rot_{thre_rot}'] = np.zeros((2, ))
    time_2d, time_data, time_6d, time_colli, time_input = 0, 0, 0, 0, 0

    batch_idx = -1
    vis_id = []
    with torch.no_grad():
        for anchor_data, rgb, depth, pc_path, grasppaths, camera_pose in tqdm(
                val_data):
            batch_idx += 1
            # get time
            start = time()
            # get scene points
            points, idxs, masks = val_data.dataset.helper.to_scene_points(
                rgb.cuda(), depth.cuda(), include_rgb=True)
            rgbs = points[..., 3:6]
            points = points[..., :3]
            # get xyz maps
            xyzs = val_data.dataset.helper.to_xyz_maps(depth.cuda())
            # get labels
            gg_ori_labels = get_ori_grasp_label(grasppaths)
            all_grasp_labels = []
            for grasppath in grasppaths:
                all_grasp_labels.append(np.load(grasppath))

            # get input time
            if batch_idx > 1:
                time_input += time() - start

            torch.cuda.synchronize()

            # get time
            start = time()

            # 2d prediction
            x, y, _, _, _ = anchor_data

            x = x.cuda(non_blocking=True)
            target = [yy.cuda(non_blocking=True) for yy in y]
            pred_2d, perpoint_features = anchornet(x)

            loc_map, cls_mask, theta_offset, depth_offset, width_offset = \
                anchor_output_process(*pred_2d, sigma=args.sigma)

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
                                      center_num=args.center_num,
                                      grid_size=args.grid_size,
                                      grasp_nms=args.grid_size,
                                      use_local_max=args.use_local_max)

            # get 2d time
            if batch_idx > 1:
                time_2d += time() - start

            # 2d bbox validation
            idx = anchor_data[2].numpy().squeeze()
            grasp_label = val_data.dataset.load_grasp_labels(idx)
            gt_rect_gg = RectGraspGroup()
            gt_rect_gg.load_from_dict(grasp_label)

            if batch_idx in vis_id:
                rgb_t = rgb.cpu().numpy().squeeze().transpose(2, 1, 0)
                plt.subplot(221)
                rect_rgb = gt_rect_gg.plot_rect_grasp_group(rgb_t, 0)
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

            # cal 2d iou/ guidance acc
            # rot = anchor_data[3].numpy().squeeze()
            # zoom_factor = anchor_data[4].numpy().squeeze()
            # gt_bbs = val_data.dataset.get_gtbb(gt_rect_gg, rot, zoom_factor)
            # for i in range(len(rect_gg)):
            # s = calculate_iou_match(rect_gg[i:i + 1], gt_bbs, thre=0.5)
            # s = calculate_guidance_accuracy(rect_gg[i], gt_bbs)
            # results['correct'] += s
            # results['total'] += 1

            # check 2d result
            if rect_gg.size == 0:
                print('No 2d grasp found')
                continue

            # get data time
            start = time()

            # feature fusion
            # points_all = concat_feature_fusion(points, perpoint_features, idxs,
            #                                    masks)
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
            grasp_info = torch.from_numpy(grasp_info).to(dtype=torch.float32,
                                                         device='cuda')

            # get data time
            if batch_idx > 1:
                time_data += time() - start
            start = time()

            # get gamma and beta classification result
            # padding for benchmark
            zero_pad_num = args.center_num - pc_group.shape[0]
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
            pc_group = pc_group[:args.center_num - zero_pad_num]
            pred_view = pred_view[:args.center_num - zero_pad_num]
            offset = offset[:args.center_num - zero_pad_num]

            # detect 6d grasp from 2d output and 6d output
            pred_grasp, pred_rect_gg = detect_6d_grasp_multi(
                rect_gg,
                pred_view,
                offset,
                valid_local_centers, (args.input_w, args.input_h),
                anchors,
                k=args.local_k)
            pred_grasp = torch.from_numpy(pred_grasp).to(dtype=torch.float32,
                                                         device='cuda')

            # get 6d time
            if batch_idx > 1:
                time_6d += time() - start

            # get collision detect time
            start = time()

            # collision detect
            pred_grasp_from_rect = pred_rect_gg.to_6d_grasp_group(depth=0.025)
            # pred_gg, valid_mask = collision_detect(points_all,
            #                                        pred_grasp_from_rect)
            # pred_grasp_after_detect = pred_grasp[valid_mask]

            # no colli
            pred_gg = pred_grasp_from_rect
            pred_grasp_after_detect = pred_grasp

            # sort and get best grasp
            _, score_idxs = torch.sort(pred_grasp_after_detect[:, 7],
                                       descending=True)
            real_top_num = int(args.top_num * len(pred_grasp_after_detect))
            pred_grasp_after_detect = pred_grasp_after_detect[
                score_idxs][:real_top_num]
            pred_gg = pred_gg.sort()[:real_top_num]

            # get collision detect time
            if batch_idx > 1:
                time_colli += time() - start

            # cal loss
            # anchor_lossd = compute_anchor_loss(pred_2d,
            #                                    target,
            #                                    loc_a=args.loc_a,
            #                                    reg_b=args.reg_b,
            #                                    cls_c=args.cls_c,
            #                                    use_CE_loss=args.use_CE_loss)
            # anchor_losses = anchor_lossd['losses']
            # anchor_loss = anchor_lossd['loss']

            # get nearest grasp labels
            # gg_labels = get_center_group_label(valid_local_centers,
            #                                    all_grasp_labels,
            #                                    args.local_grasp_num)
            # get loss
            # multi_cls_loss, offset_loss = compute_multicls_loss(
            #     pred_view, offset, gg_labels, grasp_info, anchors)

            # transfer grasp to (num, 4, 4)
            grasps = np.ones((len(pred_gg), 4, 4))
            grasps[:, :3, 3] = pred_gg.translations
            grasps[:, :3, :3] = pred_gg.rotations
            grasps[:, 3, 3] = pred_gg.scores

            # regnet validation
            vgr, score, no_colli_idxs, grasp_nocoll_view_num, grasp_nocoll_view, grasp_nocoll_scene = eval_validate(
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

            # cal 6d match
            for scale_factor in eval_scale:
                thre_dis = dis_criterion * scale_factor
                thre_rot = rot_criterion * scale_factor
                r_g, r_d, r_r = calculate_6d_match(pred_grasp_after_detect,
                                                   gg_ori_labels,
                                                   threshold_dis=thre_dis,
                                                   threshold_rot=thre_rot)

                results[f'grasp_{scale_factor}'] += r_g
                results[f'trans_{thre_dis}'] += r_d
                results[f'rot_{thre_rot}'] += r_r

            # cal coverage rate
            cover_cnt = calculate_coverage(
                pred_grasp_after_detect[no_colli_idxs], gg_ori_labels)
            results['cover_cnt'] += cover_cnt
            results['label_cnt'] += len(gg_ori_labels)

            if batch_idx in vis_id:
                print('pred gg ==', len(pred_gg))
                print(
                    f'score == {score / grasp_nocoll_view_num:.3f}  cover == {cover_cnt / len(gg_ori_labels)}'
                )
                pred_gg.depths = np.full(pred_gg.depths.shape, 0.025)
                grasp_geo = pred_gg.to_open3d_geometry_list()
                cloud = points_all[:, :3].cpu().numpy()
                rgb = rgbs.cpu().numpy().squeeze()
                vispc = o3d.geometry.PointCloud()
                vispc.points = o3d.utility.Vector3dVector(cloud)
                vispc.colors = o3d.utility.Vector3dVector(rgb)
                o3d.visualization.draw_geometries([vispc] + grasp_geo)

            if batch_idx > 0 and batch_idx % 100 == 0:
                cur_score = results['score'] / results['grasp_nocoll_view_num']
                cur_vgr = results['vgr'] / results['grasp_nocoll_view_num']
                cur_cover = results['cover_cnt'] / results['label_cnt']
                cur_ga = 0
                if results['total'] > 0:
                    cur_ga = results['correct'] / results['total']
                print(
                    f'cur: vgr == {cur_vgr:.3f} score == {cur_score:.3f} cover == {cur_cover.item():.3f} ga == {cur_ga:.3f}'
                )
                tmp_input = time_input / (batch_idx - 1) * 1000
                tmp_2d = time_2d / (batch_idx - 1) * 1000
                tmp_data = time_data / (batch_idx - 1) * 1000
                tmp_6d = time_6d / (batch_idx - 1) * 1000
                tmp_colli = time_colli / (batch_idx - 1) * 1000
                tmp_sum = tmp_2d + tmp_data + tmp_6d + tmp_colli
                print('Time stats:')
                print(f'input: {tmp_input:.3f} ms')
                print(f'total: {tmp_sum:.3f} ms')
                print(
                    f'2d: {tmp_2d:.3f} ms  data: {tmp_data:.3f} ms  6d: {tmp_6d:.3f} ms  colli: {tmp_colli:.3f} ms'
                )

            # tensorboard record
            # loss = anchor_loss + multi_cls_loss + offset_loss
            # results['loss'] += loss.item()
            # results['anchor_loss'] += anchor_loss.item()
            # results['multi_cls_loss'] += multi_cls_loss.item()
            # results['offset_loss'] += offset_loss.item()
            # for ln, l in anchor_losses.items():
            #     if ln not in results['losses']:
            #         results['losses'][ln] = 0
            #     results['losses'][ln] += l.item()

    batch_idx += 1
    # time stat
    time_input = time_input / batch_idx * 1000
    time_2d = time_2d / batch_idx * 1000
    time_data = time_data / batch_idx * 1000
    time_6d = time_6d / batch_idx * 1000
    time_colli = time_colli / batch_idx * 1000
    time_sum = time_2d + time_data + time_6d + time_colli
    logging.info('Time stats:')
    logging.info(f'total: {time_sum:.3f} ms')
    logging.info(
        f'2d: {time_2d:.3f} ms  data: {time_data:.3f} ms  6d: {time_6d:.3f} ms  colli: {time_colli:.3f} ms'
    )

    # loss stat
    # results['loss'] /= batch_idx
    # results['anchor_loss'] /= batch_idx
    # results['multi_cls_loss'] /= batch_idx
    # results['offset_loss'] /= batch_idx
    # for ln, l in anchor_losses.items():
    #     results['losses'][ln] /= batch_idx
    return results


def run():
    args = parse_args()

    # prepare for trainning
    tb, save_folder = prepare_torch_and_logger(args, mode='test')

    # Load Dataset
    logging.info('Loading Dataset...')
    Test_Dataset = AcronymPointDataset(
        args.all_points_num,
        args.dataset_path,
        noise=args.noise,
        sceneIds=list(range(args.scene_l, args.scene_r)),
        use_CE_loss=args.use_CE_loss,
        sigma=args.sigma,
        ratio=args.ratio,
        anchor_k=args.anchor_k,
        anchor_z=args.anchor_z,
        anchor_w=args.anchor_w,
        grasp_count=args.grasp_count,
        output_size=(args.input_w, args.input_h),
        random_rotate=False,
        random_zoom=False)

    logging.info('Test size: {}'.format(len(Test_Dataset)))

    test_data = torch.utils.data.DataLoader(Test_Dataset,
                                            batch_size=1,
                                            pin_memory=True,
                                            shuffle=True,
                                            num_workers=1)

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
        if 'gamma' in ckpt:
            anchors['gamma'] = ckpt['gamma']
            anchors['beta'] = ckpt['beta']
            logging.info('Using saved anchors')
            print(anchors['gamma'])
            print(anchors['beta'])
        anchornet.load_state_dict(ckpt['anchor'])
        localnet.load_state_dict(ckpt['local'])

    # get model architecture
    # print_model(args, input_channels, anchornet, save_folder)

    anchornet = anchornet.cuda()
    localnet = localnet.cuda()
    logging.info('Done')

    # Run Validation
    logging.info('Validating...')
    test_results = validate(0, anchornet, localnet, test_data, anchors, args)

    log_match_result(test_results, dis_criterion, rot_criterion)
    log_and_save(args, tb, test_results, 0, anchornet, localnet, None, anchors,
                 save_folder)


if __name__ == '__main__':
    run()
