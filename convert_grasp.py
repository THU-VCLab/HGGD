import copy
import json
import math
import os
import random
import time
from multiprocessing import Pool

import cv2
import numpy as np
import open3d as o3d

from customgraspnetAPI import GraspNet
from util import (batch_framexy_depth_2_xyz, batch_rgbdxyz_2_rgbxy_depth,
                  get_batch_key_points, mask_grasp_group,
                  remove_invisible_grasp_points, to_grasp_group,
                  to_open3d_geometry_list, to_opencv_image,
                  to_rect_grasp_group)

graspnet_root = '/ssd/graspnet'  # ROOT PATH FOR GRASPNET
output_path = './realsense'
camera = 'realsense'
dataset = GraspNet(graspnet_root, camera=camera, split='train')

num_cpu = 1
skip = True
data_preload = False
vis_time = False
vis_2d = False
vis_6d = False
save_2d = False
save_6d = True
if save_2d and save_6d:
    raise Exception(
        'You can not save 2d dataset and 6d dataset simultaneously!')

threshold = 0  # threshold for rotations[:, 2, 0]
visnum_6d = 1000  # num of grasp to show 6d
visnum_2d = 0
np.random.seed(1)
sceneIds = list(range(0, 190))
annIds = range(0, 256)
# sceneIds = [0]
# annIds = [0]


def grasp2rect(_6d_grasp):

    k_points_6d, rect_grasp_group = to_rect_grasp_group(_6d_grasp, camera)

    dy = rect_grasp_group.center_points[:, 1] - rect_grasp_group.open_points[:,
                                                                             1]
    dx = rect_grasp_group.center_points[:, 0] - rect_grasp_group.open_points[:,
                                                                             0]
    thetas_rad = np.arctan2(dy, -dx)

    # batch speed up
    # cal gamma first
    tmp_gamma = (k_points_6d[:, 1, 2] -
                 k_points_6d[:, 0, 2]) * 2 / _6d_grasp.widths
    gammas_rad = np.arcsin(np.clip(tmp_gamma, -1, 1))
    # use gamma to cal beta
    tmp_beta = (k_points_6d[:, 2, 2] -
                k_points_6d[:, 3, 2]) / np.cos(gammas_rad) / _6d_grasp.heights
    betas_rad = np.arcsin(np.clip(tmp_beta, -1, 1))

    ## 在平移之后计算widths_2d
    widths_2d = np.sqrt(
        np.square(rect_grasp_group.center_points[:, 0] -
                  rect_grasp_group.open_points[:, 0]) +
        np.square(rect_grasp_group.center_points[:, 1] -
                  rect_grasp_group.open_points[:, 1])) * 2

    # for 6d dataset
    widths_2d /= np.cos(gammas_rad)
    rect_grasp_group.heights *= 2 / (
        1 + np.sqrt(1 - np.square(np.cos(gammas_rad) * np.sin(betas_rad))))

    return rect_grasp_group, thetas_rad, gammas_rad, betas_rad, widths_2d


def rect2grasp(rect_gg, widths_2d, center_z_depths, thetas_rad, gammas_rad,
               betas_rad):
    translations = np.array(
        batch_framexy_depth_2_xyz(rect_gg.center_points[:, 0],
                                  rect_gg.center_points[:, 1],
                                  center_z_depths / 1000, camera)).T
    center_points = rect_gg.center_points.copy()

    # centers和opens平移到图片中心
    if camera == 'kinect':
        center_points[:, 0] = 638.43
        center_points[:, 1] = 366.50
    elif camera == 'realsense':
        center_points[:, 0] = 651.32
        center_points[:, 1] = 349.62

    opens_x = rect_gg.center_points[:, 0] - widths_2d / 2 * np.cos(thetas_rad)
    opens_y = rect_gg.center_points[:, 1] + widths_2d / 2 * np.sin(thetas_rad)
    opens_2d = np.stack([opens_x, opens_y], axis=-1)
    rect_gg.open_points = opens_2d

    gg_from_rect = to_grasp_group(rect_gg, camera, center_z_depths)

    # batch matmul to speed up
    # rot gamma
    rots = gg_from_rect.rotation_matrices.copy()
    R = np.zeros((len(gg_from_rect), 3, 3))
    R[:, 0, 0] = np.cos(gammas_rad)
    R[:, 0, 1] = -np.sin(gammas_rad)
    R[:, 1, 0] = np.sin(gammas_rad)
    R[:, 1, 1] = np.cos(gammas_rad)
    R[:, 2, 2] = 1
    rots = np.einsum('ijk,ikN->ijN', rots, R)
    # rot beta
    R = np.zeros((len(gg_from_rect), 3, 3))
    R[:, 0, 0] = np.cos(betas_rad)
    R[:, 0, 2] = np.sin(betas_rad)
    R[:, 1, 1] = 1
    R[:, 2, 0] = -np.sin(betas_rad)
    R[:, 2, 2] = np.cos(betas_rad)
    rots = np.einsum('ijk,ikN->ijN', rots, R)
    gg_from_rect.rotation_matrices = rots

    # set depth to 0.02
    gg_from_rect.depths = np.ones(gg_from_rect.depths.shape) * 0.02
    gg_from_rect.translations = translations

    return gg_from_rect


def run(sceneIds):
    # convert to list
    if isinstance(sceneIds, int):
        sceneIds = [sceneIds]
    print(sceneIds)
    # preload all labels
    if data_preload:
        grasp_labels = dataset.loadGraspLabels(objIds=None)
        collision_labels = dataset.loadCollisionLabels(sceneIds=None)

    for scene in sceneIds:
        print(f'****scene {scene} started****')
        # skip existing file
        if save_2d:
            scenePath = os.path.join(output_path, 'planar_dataset',
                                     f'scene_{scene}')
        elif save_6d:
            scenePath = os.path.join(output_path, '6d_dataset',
                                     f'scene_{scene}')

        finished = True
        for ann in annIds:
            infoFile = os.path.join(scenePath, 'infos', f'{ann}_info.json')
            if not os.path.exists(infoFile):
                finished = False
                break

        if skip and finished:
            continue

        if not data_preload:
            object_list = []
            with open(
                    os.path.join(graspnet_root, 'scenes',
                                 'scene_%04d' % (scene, ),
                                 'object_id_list.txt'), 'r') as f:
                for line in f.readlines():
                    object_list.append(int(line))
            grasp_labels = dataset.loadGraspLabels(objIds=object_list)
            collision_labels = dataset.loadCollisionLabels(sceneIds=scene)

        for ann in annIds:
            if ann % 20 == 0:
                print(f'****scene {scene} ann {ann} started****')
            record = {}

            infoFile = os.path.join(scenePath, 'infos', f'{ann}_info.json')
            if skip and os.path.exists(infoFile):
                continue

            start = time.time()

            # get scene pc by models
            scenePCD = dataset.loadSceneModel(sceneId=scene,
                                              camera=camera,
                                              annId=ann,
                                              align=False)
            for i, pcd in enumerate(scenePCD):
                if i == 0:
                    scenePOINTS = np.array(pcd.points)
                else:
                    scenePOINTS = np.concatenate([scenePOINTS, pcd.points])
            sample_num = 40000
            idxs = random.sample(range(len(scenePOINTS)), sample_num)
            scenePOINTS_homo = np.concatenate(
                [scenePOINTS[idxs],
                 np.ones((sample_num, 1))], axis=1)  # (40000, 4)
            # free memory
            del scenePOINTS, scenePCD

            ## 从数据集加载6d grasps
            gg = dataset.loadGrasp(sceneId=scene,
                                   camera=camera,
                                   annId=ann,
                                   fric_coef_thresh=0.2,
                                   format='6d',
                                   grasp_labels=grasp_labels,
                                   collision_labels=collision_labels,
                                   vis_time=False)
            total_grasp_cnt = len(gg)

            gg_mask, _ = mask_grasp_group(gg, threshold=threshold)

            # free memory
            del gg

            # speed up
            # gg_mask = gg_mask.nms(translation_thresh=0.01,
            #                       rotation_thresh=30 * np.pi / 180)
            # gg_mask.grasp_group_array = np.copy(gg_mask.grasp_group_array)
            # gg_mask.grasp_group_array.flags.writeable = True
            # sample to speed up and save memory usage
            mask_num = len(gg_mask)
            print(mask_num)
            if len(gg_mask) > 5000:
                idxs = random.sample(range(mask_num), 5000)
                gg_mask = gg_mask[idxs]

            if vis_time:
                print('load and mask time ==', (time.time() - start))
                start = time.time()

            ## refine grasp width and grasp center using contact points
            # transform scenePOINTS to local grasp coodinates
            global_to_local = np.eye(4)[np.newaxis,
                                        ...].repeat(len(gg_mask),
                                                    0)  # (grasp_num, 4, 4)
            global_to_local[:, 0:3, 0:3] = gg_mask.rotation_matrices.transpose(
                0, 2, 1)
            global_to_local[:, 0:3, 3:4] = -np.matmul(
                gg_mask.rotation_matrices.transpose(0, 2, 1),
                np.expand_dims(gg_mask.translations, axis=2))
            local_clouds = np.einsum('ijk,kn->ijn', global_to_local,
                                     scenePOINTS_homo.transpose(1, 0))
            local_clouds = local_clouds.transpose(
                0, 2, 1)[..., :3]  # (grasp_num, 40000, 4)

            # crop points in grasp areas mask
            heights = gg_mask.heights[:, None]
            depths = gg_mask.depths[:, None]
            finger_length = 0.04
            widths = gg_mask.widths[:, None]
            # height mask
            mask1 = ((local_clouds[..., 2] > -heights / 2) &
                     (local_clouds[..., 2] < heights / 2))
            # left finger mask
            mask2 = ((local_clouds[..., 0] > depths - finger_length) &
                     (local_clouds[..., 0] < depths))
            mask3 = (local_clouds[..., 1] > -widths / 2)
            # right finger mask
            mask4 = (local_clouds[..., 1] < widths / 2)
            # check between points
            mask_between = (mask1 & mask2 & mask3 & mask4
                            )  # (grasp_num, 40000)
            # print(mask_between.shape)

            # get contact points
            local_areas_min = ((local_clouds[..., 1] - 10) *
                               mask_between).min(1) + 10  # min along y-axis
            local_areas_max = (
                (local_clouds[..., 1] + 10) * mask_between).max(1) - 10

            local_center = np.zeros(
                (local_clouds.shape[0], 4))  # (grasp_num, 4)
            local_center[:, 3] += 1
            local_center[:, 1] = (local_areas_min + local_areas_max) / 2
            # print(local_center[:, 1])
            mask_refine = abs(local_areas_min +
                              local_areas_max) >= 0.0  # (grasp_num,)
            # print(mask_refine)
            local_center[:, :3] = local_center[:, :3] * mask_refine[..., None]
            # print(local_center[:, 1])

            # print(gg_mask_nms.widths[4]/2, local_center[4, 1])
            width_refine = np.minimum(
                gg_mask.widths / 2 - local_center[:, 1],
                gg_mask.widths / 2 + local_center[:, 1]) * 2
            center_refine = np.einsum('ijk,ki->ij',
                                      np.linalg.inv(global_to_local),
                                      local_center.transpose(1, 0))

            gg_mask.translations = center_refine[..., :3]
            # not refine width
            # gg_mask.widths = width_refine
            if vis_time:
                print('refine time ==', (time.time() - start))
                start = time.time()

            # grasp nms
            gg_mask_nms = gg_mask.nms(translation_thresh=0.02,
                                      rotation_thresh=60 * np.pi / 180)
            # free memory
            del gg_mask
            # fix read-only array fault caused by grasp nms
            gg_mask_nms.grasp_group_array = np.copy(
                gg_mask_nms.grasp_group_array)
            gg_mask_nms.grasp_group_array.flags.writeable = True

            if vis_time:
                print('nms time ==', time.time() - start)
                start = time.time()

            ## 在平移之前记录下centers
            k_points = get_batch_key_points(gg_mask_nms.translations,
                                            gg_mask_nms.rotation_matrices,
                                            gg_mask_nms.widths)
            k_points = k_points.reshape([-1, 3])
            k_points = k_points.reshape([-1, 4, 3])
            centers_2d, center_z_depths = batch_rgbdxyz_2_rgbxy_depth(
                k_points[:, 0, :], camera)

            ## 将抓取平移到[0, 0, center_z_depths]再进行转换
            translations = copy.deepcopy(gg_mask_nms.translations)
            gg_mask_nms.translations = np.zeros((len(gg_mask_nms), 3))
            gg_mask_nms.translations[:, 2] = center_z_depths / 1000

            ## 6d转2d，并记录角度
            rect_grasp_group, thetas_rad, gammas_rad, betas_rad, widths_2d = grasp2rect(
                gg_mask_nms)

            ## 使用center_points、theta_rad和widths_2d_trans计算2D矩形框的open_points, upper_points
            opens_x = centers_2d[:, 0] - widths_2d / 2 * np.cos(thetas_rad)
            opens_y = centers_2d[:, 1] + widths_2d / 2 * np.sin(thetas_rad)
            opens_2d = np.stack([opens_x, opens_y], axis=-1)

            rect_grasp_group.center_points = centers_2d
            rect_grasp_group.open_points = opens_2d

            ## 2d转6d
            gg_from_rect = rect2grasp(rect_grasp_group, widths_2d,
                                      center_z_depths, thetas_rad, gammas_rad,
                                      betas_rad)

            ## 还原ground truth translations, 并计算6D grasps的widths和heights
            gg_mask_nms.translations = translations

            ## 过滤中心点在图片外的抓取
            valid_mask1 = np.min(centers_2d, axis=1) > 0
            valid_mask2 = centers_2d[:, 0] < 1280
            valid_mask3 = centers_2d[:, 1] < 720
            valid_mask = np.logical_and(
                np.logical_and(valid_mask1, valid_mask2), valid_mask3)

            # remove invisible grasp points
            scenecloud, _ = dataset.loadScenePointCloud(sceneId=scene,
                                                        camera=camera,
                                                        annId=ann,
                                                        format='numpy')
            idxs = random.sample(range(len(scenecloud)), 40000)
            cloud = scenecloud[idxs]
            # free memory
            del scenecloud
            visible_mask = remove_invisible_grasp_points(
                cloud, gg_from_rect.translations, th=0.01)
            gg_from_rect = gg_from_rect[visible_mask]
            gg_mask_nms = gg_mask_nms[visible_mask]

            print('grasp count ==', len(valid_mask))
            if vis_time:
                print('total ==', len(valid_mask), ' valid ==',
                      np.sum(valid_mask))
                print('convert time ==', time.time() - start)
                start = time.time()

            if save_2d:
                # print(f'*****Start ({scene}, {ann}) save 2D grasp dataset!******')
                if not os.path.exists(scenePath):
                    os.makedirs(os.path.join(scenePath, 'grasp_labels'))
                    os.makedirs(os.path.join(scenePath, 'infos'))
                annFile = os.path.join(scenePath, 'grasp_labels',
                                       f'{ann}_view.npz')
                np.savez(annFile,
                         centers_2d=centers_2d[valid_mask],
                         opens_2d=opens_2d[valid_mask],
                         thetas_rad=thetas_rad[valid_mask],
                         widths_2d=widths_2d[valid_mask],
                         heights_2d=rect_grasp_group.heights[valid_mask],
                         scores_from_6d=rect_grasp_group.scores[valid_mask],
                         object_ids=rect_grasp_group.object_ids[valid_mask],
                         center_z_depths=center_z_depths[valid_mask])

            if save_6d:
                # print(f'*****Start ({scene}, {ann}) save 6D grasp dataset!******')
                if not os.path.exists(scenePath):
                    os.makedirs(os.path.join(scenePath, 'grasp_labels'))
                    os.makedirs(os.path.join(scenePath, 'infos'))
                annFile = os.path.join(scenePath, 'grasp_labels',
                                       f'{ann}_view.npz')
                np.savez(annFile,
                         centers_2d=centers_2d[valid_mask],
                         opens_2d=opens_2d[valid_mask],
                         thetas_rad=thetas_rad[valid_mask],
                         gammas_rad=gammas_rad[valid_mask],
                         betas_rad=betas_rad[valid_mask],
                         widths_2d=widths_2d[valid_mask],
                         heights_2d=rect_grasp_group.heights[valid_mask],
                         scores_from_6d=rect_grasp_group.scores[valid_mask],
                         object_ids=rect_grasp_group.object_ids[valid_mask],
                         center_z_depths=center_z_depths[valid_mask])

            ## visualization
            if vis_2d:
                bgr = dataset.loadBGR(sceneId=scene, annId=ann, camera=camera)
                img = to_opencv_image(rect_grasp_group,
                                      bgr,
                                      shuffle=True,
                                      numGrasp=visnum_2d)
                cv2.imshow('rectangle grasps', img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            if vis_6d:
                # print(gg_mask_nms.translations)
                geometries = []
                geometries.append(
                    dataset.loadScenePointCloud(sceneId=scene,
                                                annId=ann,
                                                camera=camera))
                # cloud_scenepoints = o3d.geometry.PointCloud()
                # cloud_scenepoints.points = o3d.utility.Vector3dVector(
                #     scenePOINTS)
                # geometries.append(cloud_scenepoints)

                # geometries += scenePCD
                start = np.random.randint(low=0,
                                          high=max(
                                              1,
                                              len(gg_mask_nms) - visnum_6d))
                # geometries += to_open3d_geometry_list(gg_mask_nms[start:start +
                #                                                   visnum_6d],
                #                                       color=(0, 0, 255))
                geometries += to_open3d_geometry_list(
                    gg_from_rect[start:start + visnum_6d], color=(255, 0, 0))
                frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
                o3d.visualization.draw_geometries([*geometries, frame])

            translation_d = np.sqrt(
                np.square(gg_mask_nms.translations -
                          gg_from_rect.translations).sum()) / len(gg_mask_nms)
            rotation_d = np.sqrt(
                np.square(gg_mask_nms.rotation_matrices -
                          gg_from_rect.rotation_matrices).sum()) / len(
                              gg_mask_nms)

            width_d = np.sqrt(
                np.square(gg_mask_nms.widths -
                          gg_from_rect.widths).sum()) / len(gg_mask_nms)

            record['numgrasp'] = len(gg_mask_nms)
            record['translation_d'] = translation_d
            record['rotation_d'] = rotation_d
            record['width_d'] = width_d

            if save_2d or save_6d:
                infoFile = os.path.join(scenePath, 'infos', f'{ann}_info.json')
                with open(infoFile, 'w') as f:
                    json.dump(record, f)

            if vis_time:
                print('save time ==', time.time() - start)
                start = time.time()

            output_str = f'({scene}, {ann}) finished:\n'
            output_str += f'gg: {total_grasp_cnt}  gg_mask: {mask_num}  gg_mask_nms: {len(gg_mask_nms)}\n'
            output_str += 'pos dis: {:.6e} rot dis: {:.6e}'.format(
                translation_d, rotation_d)
            output_str += f' width dis: {width_d:.6e}'
            print(output_str)

        print(f'****scene {scene} finished****')


if __name__ == '__main__':
    print('len ==', len(sceneIds))
    with Pool(num_cpu) as pool:
        for scene_id in sceneIds:
            pool.apply_async(run, [scene_id])
        pool.close()
        pool.join()

    # for scene_id in sceneIds:
    #     run([scene_id])
