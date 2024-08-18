import argparse
import datetime
import json
import logging
import os
import random
import sys

import numpy as np
import tensorboardX
import torch
import torch.multiprocessing as mp
import torch.optim as optim
from torchsummary import summary

from customgraspnetAPI import Grasp, GraspGroup

eval_scale = np.linspace(0.2, 1, 5)


def log_acc_str(name, T, F):
    T, F = int(T), int(F)
    if T + F == 0:
        return f'{name} 0/0 = 0'
    return f'{name} {T}/{T + F} = {T / (T + F):.3f}'


def parse_args():
    parser = argparse.ArgumentParser(description='Train network')
    # Network
    # 2d
    parser.add_argument('--resume', type=str, default=None, help='Model path')
    parser.add_argument('--input-h',
                        type=int,
                        default=360,
                        help='Input image size for the network')
    parser.add_argument('--input-w',
                        type=int,
                        default=640,
                        help='Input image size for the network')
    parser.add_argument('--use-depth',
                        type=int,
                        default=1,
                        help='Use Depth image for training (1/0)')
    parser.add_argument('--use-rgb',
                        type=int,
                        default=1,
                        help='Use RGB image for training (1/0)')
    parser.add_argument('--iou-threshold',
                        type=float,
                        default=0.25,
                        help='Threshold for IOU matching')

    # pc
    parser.add_argument(
        '--center-num',
        type=int,
        default=128,
        help='choose how many centers from 2d predicted heatmap')
    parser.add_argument('--group-num',
                        type=int,
                        default=512,
                        help='point num around one center of ball query')
    parser.add_argument('--anchor-num',
                        type=int,
                        default=7,
                        help='anchor num for gamma and beta')
    parser.add_argument('--local-grasp-num',
                        type=int,
                        default=500,
                        help='number of local grasps in local pointcloud')

    # Dataset
    parser.add_argument('--scene-l',
                        type=int,
                        default=0,
                        help='Scene id left range')
    parser.add_argument('--scene-r',
                        type=int,
                        default=100,
                        help='Scene id right range')
    parser.add_argument('--dataset-path',
                        type=str,
                        default=None,
                        help='Path to grasp dataset')
    parser.add_argument('--scene-path',
                        type=str,
                        default=None,
                        help='Path to scene dataset')
    parser.add_argument('--checkpoint',
                        type=str,
                        default=None,
                        help='Checkpoint path to load')
    parser.add_argument('--num-workers',
                        type=int,
                        default=4,
                        help='Dataset workers')

    # Anchor
    parser.add_argument('--ratio',
                        type=int,
                        default=8,
                        help='Down sample ratio')
    parser.add_argument('--grid-size',
                        type=int,
                        default=8,
                        help='2D center select grid size')
    parser.add_argument(
        '--anchor-k',
        type=int,
        default=6,
        help='The number of oriented anchor boxes with different angles')
    parser.add_argument('--anchor-w',
                        type=float,
                        default=50.0,
                        help='The default width of the anchor boxes')
    parser.add_argument('--anchor-z',
                        type=float,
                        default=20.0,
                        help='The default z of the anchor boxes')
    parser.add_argument('--grasp-count',
                        type=int,
                        default=5000,
                        help='The default grasp count of one image')
    parser.add_argument('--sigma',
                        type=int,
                        default=10,
                        help='Gaussian kernel sigma')
    parser.add_argument('--loc-a',
                        type=float,
                        default=1,
                        help='loss coef for loc map')
    parser.add_argument('--reg-b',
                        type=float,
                        default=1,
                        help='loss coef for regress')
    parser.add_argument('--cls-c',
                        type=float,
                        default=1,
                        help='loss coef for classify')
    parser.add_argument('--offset-d',
                        type=float,
                        default=5,
                        help='loss coef for grasp 3d offset')

    # Training
    parser.add_argument('--all-points-num',
                        type=int,
                        default=25600,
                        help='downsample scene points')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--shift-epoch',
                        type=int,
                        default=5,
                        help='Epoch num for anchor shifting')
    parser.add_argument(
        '--pre-epochs',
        type=int,
        default=-1,
        help='Pre training 2d epochs, will be 0 if joint-trainning')
    parser.add_argument('--joint-trainning',
                        action='store_true',
                        help='Whether to train 2d and 6d net together')
    parser.add_argument('--epochs',
                        type=int,
                        default=15,
                        help='Training epochs')
    parser.add_argument('--lr', type=float, default=3e-3, help='Learning rate')
    parser.add_argument('--optim',
                        type=str,
                        choices=['adam', 'adamw', 'sgd'],
                        help='Optmizer for the training. (adam, adamw or SGD)')
    parser.add_argument(
        '--step-cnt',
        type=int,
        default=1,
        help='Network batch step cnt (batch_size * step_cnt == real_batch_size)'
    )
    parser.add_argument('--noise', type=float, default=0.0, help='Depth noise')

    # grasp detection
    parser.add_argument('--heatmap-thres',
                        type=float,
                        default=0.01,
                        help='2D grasp generation heatmap_thres')
    parser.add_argument('--local-k',
                        type=int,
                        default=3,
                        help='Local anchor top-k selection')
    parser.add_argument('--local-thres',
                        type=float,
                        default=0.01,
                        help='6D grasp generation local multi_cls score thres')
    parser.add_argument('--rotation-num',
                        type=int,
                        default=1,
                        help='Local rotation num for 2D grasp')
    parser.add_argument('--top-num',
                        type=float,
                        default=1.0,
                        help='Grasp Detect Ratio Number')

    # Logging etc.
    parser.add_argument('--description',
                        type=str,
                        default='',
                        help='Training description')
    parser.add_argument('--save-freq',
                        type=int,
                        default=1,
                        help='Model save frequency')
    parser.add_argument('--logdir',
                        type=str,
                        default='./logs/',
                        help='Log directory')
    parser.add_argument('--random-seed',
                        type=int,
                        default=123,
                        help='Random seed')

    args = parser.parse_args()
    if args.joint_trainning:
        args.pre_epochs = 0
        print('Joint Trainning for the whole network')
    return args


def prepare_torch_and_logger(args, mode='train'):
    # multiprocess
    # mp.set_start_method('spawn')
    # set torch and gpu setting
    np.set_printoptions(precision=4, suppress=True)
    torch.set_printoptions(precision=4, sci_mode=False)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    else:
        raise RuntimeError('CUDA not available')

    # random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    # Set-up output directories
    net_desc = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
    net_desc = net_desc + '_' + args.description
    if mode == 'test':
        net_desc = 'test' + net_desc

    save_folder = os.path.join(args.logdir, net_desc)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    tb = tensorboardX.SummaryWriter(save_folder)

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

    return tb, save_folder


def get_optimizer(args, params):
    # get optimizer
    if args.optim.lower() == 'adam':
        optimizer = optim.Adam(params, lr=args.lr, weight_decay=1e-4)
    elif args.optim.lower() == 'adamw':
        optimizer = optim.AdamW(params, lr=args.lr, weight_decay=1e-2)
    elif args.optim.lower() == 'sgd':
        optimizer = optim.SGD(params, lr=args.lr, momentum=0.9)
    else:
        raise NotImplementedError('Optimizer {} is not implemented'.format(
            args.optim))
    return optimizer


def print_model(args, input_channels, model, save_folder):
    summary(model, (input_channels, args.input_w, args.input_h), device='cpu')
    with open(os.path.join(save_folder, 'arch.txt'), 'w') as f:
        sys.stdout = f
        summary(model, (input_channels, args.input_w, args.input_h),
                device='cpu')
        sys.stdout = sys.__stdout__


def log_match_result(results, dis_criterion, rot_criterion):
    for scale_factor in eval_scale:
        # get threshold from criterion and factor
        thre_dis = dis_criterion * scale_factor
        thre_rot = rot_criterion * scale_factor

        t_trans, f_trans = results[f'trans_{thre_dis}']
        t_rot, f_rot = results[f'rot_{thre_rot}']
        t_grasp, f_grasp = results[f'grasp_{scale_factor}']

        t_str = log_acc_str(f'trans_{thre_dis:.2f}', t_trans, f_trans)
        r_str = log_acc_str(f'rot_{thre_rot:.2f}', t_rot, f_rot)
        g_str = log_acc_str(f'grasp_{scale_factor:.2f}', t_grasp, f_grasp)

        logging.info(f'{t_str}  {r_str}  {g_str}')


def log_and_save(args,
                 tb,
                 results,
                 epoch,
                 anchornet,
                 localnet,
                 optimizer,
                 anchors,
                 save_folder,
                 mode='regnet'):
    # Log validation results to tensorboard
    # loss
    tb.add_scalar('val_loss/loss', results['loss'], epoch)
    tb.add_scalar('val_loss/anchor_loss', results['anchor_loss'], epoch)
    for n, l in results['losses'].items():
        tb.add_scalar('val_loss/' + n, l, epoch)

    logging.info('Validation Loss:')
    logging.info(f'test loss: {results["loss"]:.3f}')
    logging.info(f'anchor loss: {results["anchor_loss"]:.3f}')
    if 'loc_map_loss' in results['losses']:
        logging.info(
            f'loc: {results["losses"]["loc_map_loss"]:.3f}, reg: {results["losses"]["reg_loss"]:.3f}, cls: {results["losses"]["cls_loss"]:.3f}'
        )
    if epoch >= args.pre_epochs:
        tb.add_scalar('val_loss/multi_cls_loss', results['multi_cls_loss'],
                      epoch)
        tb.add_scalar('val_loss/offset_loss', results['offset_loss'], epoch)
        logging.info(f'multicls_loss: {results["multi_cls_loss"]:.3f}')
        logging.info(f'offset_loss: {results["offset_loss"]:.3f}')

    # coverage
    if epoch >= args.pre_epochs:
        cover_cnt = results['cover_cnt']
        label_cnt = results['label_cnt']
        tb.add_scalar('coverage', cover_cnt / label_cnt, epoch)
        logging.info(
            f'coverage rate: {cover_cnt} / {label_cnt} = {cover_cnt / label_cnt:.3f}'
        )

    # 2d iou
    if results['total'] > 0:
        iou = results['correct'] / results['total']
        tb.add_scalar('IOU', iou, epoch)
        logging.info(f'2d iou: {iou:.2f}')

    # regnet validation
    if epoch >= args.pre_epochs:
        if mode == 'regnet':
            view_num = results['grasp_nocoll_view_num']
            vgr = results['vgr']
            score = results['score']
            if view_num > 0:
                tb.add_scalar('collision_free_ratio', vgr / view_num, epoch)
                tb.add_scalar('score', score / view_num, epoch)

                logging.info('REGNet validation:')
                logging.info(f'vgr: {vgr} / {view_num} = {vgr / view_num:.3f}')
                logging.info(
                    f'score: {score:.3f} / {view_num} = {score / view_num:.3f}'
                )
            else:
                logging.info('No collision-free grasp')
        elif mode == 'graspnet':
            logging.info('please run test_graspnet.py for graspnet result')

    # Save best performing network
    if epoch % args.save_freq == 0 and optimizer is not None:
        if epoch < args.pre_epochs:
            torch.save(
                {
                    'anchor': anchornet.module.state_dict(),
                    'local': localnet.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'gamma': anchors['gamma'],
                    'beta': anchors['beta']
                }, os.path.join(save_folder, f'epoch_{epoch}_iou_{iou:.3f}'))
        elif mode == 'regnet':
            torch.save(
                {
                    'anchor': anchornet.module.state_dict(),
                    'local': localnet.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'gamma': anchors['gamma'],
                    'beta': anchors['beta']
                },
                os.path.join(
                    save_folder,
                    f'epoch_{epoch}_score_{score / view_num:.3f}_cover_{cover_cnt / label_cnt:.3f}'
                ))
        elif mode == 'graspnet':
            torch.save(
                {
                    'anchor': anchornet.module.state_dict(),
                    'local': localnet.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'gamma': anchors['gamma'],
                    'beta': anchors['beta']
                },
                os.path.join(
                    save_folder,
                    f'epoch_{epoch}_iou_{iou:.3f}_cover_{cover_cnt / label_cnt:.3f}'
                ))


def log_test_result(args, results, epoch, mode='regnet'):
    # Log validation results to tensorboard
    # loss
    logging.info('Test Loss:')
    logging.info(f'test loss: {results["loss"]:.3f}')
    logging.info(f'anchor loss: {results["anchor_loss"]:.3f}')
    logging.info(
        f'loc: {results["losses"]["loc_map_loss"]:.3f}, reg: {results["losses"]["reg_loss"]:.3f}, cls: {results["losses"]["cls_loss"]:.3f}'
    )
    if epoch >= args.pre_epochs:
        logging.info(f'multicls_loss: {results["multi_cls_loss"]:.3f}')

    # coverage
    if epoch >= args.pre_epochs:
        cover_cnt = results['cover_cnt']
        label_cnt = results['label_cnt']
        logging.info(
            f'coverage rate: {cover_cnt} / {label_cnt} = {cover_cnt / label_cnt:.3f}'
        )

    # 2d iou
    iou = results['correct'] / (results['correct'] + results['failed'])
    logging.info(f'2d iou: {iou:.2f}')

    # regnet validation
    if epoch >= args.pre_epochs:
        if mode == 'regnet':
            view_num = results['grasp_nocoll_view_num']
            vgr = results['vgr']
            score = results['score']
            if view_num > 0:
                logging.info('REGNet validation:')
                logging.info(f'vgr: {vgr} / {view_num} = {vgr / view_num:.3f}')
                logging.info(
                    f'score: {score:.3f} / {view_num} = {score / view_num:.3f}'
                )
            else:
                logging.info('No collision-free grasp')
        elif mode == 'graspnet':
            logging.info('please run test_graspnet.py for graspnet result')


def log_anchor_loss(epoch, batch_idx, loss, anchor_loss, anchor_losses,
                    batch_cnt):
    logging.info('Epoch: {}, Batch: {}, total_loss: {:0.4f}'.format(
        epoch, batch_idx, loss / batch_cnt))
    logging.info('anchor_loss: {:0.4f}'.format(anchor_loss / batch_cnt))
    logging.info(
        'loc_map_loss: {:0.4f}, reg_loss: {:0.4f}, cls_loss: {:0.4f}'.format(
            anchor_losses['loc_map_loss'] / batch_cnt,
            anchor_losses['reg_loss'] / batch_cnt,
            anchor_losses['cls_loss'] / batch_cnt))


def dump_grasp(epoch, batch_idx, pred_gg, scene_list, dump_dir='./pred'):
    gg = GraspGroup()
    for g in pred_gg:
        g = Grasp(1, g.width, 0.02, 0.02, g.rotation.reshape(9, ),
                  g.translation, -1)
        gg.add(g)

    # save grasps
    save_dir = os.path.join(dump_dir, f'epoch_{epoch}')
    save_dir = os.path.join(save_dir, scene_list[batch_idx])
    save_dir = os.path.join(save_dir, 'realsense')
    save_path = os.path.join(save_dir, str(batch_idx % 256).zfill(4) + '.npy')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    gg.save_npy(save_path)
