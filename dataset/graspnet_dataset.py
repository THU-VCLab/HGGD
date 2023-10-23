import logging
import os

import cupoch
import numpy as np
from torchvision.transforms import (ColorJitter, Compose, PILToTensor,
                                    ToPILImage)
from tqdm import tqdm

from .base_grasp_dataset import GraspDataset
from .config import camera
from .utils import PointCloudHelper


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print('Showing outliers (red) and inliers (gray): ')
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    cupoch.visualization.draw_geometries([inlier_cloud, outlier_cloud])


class GraspnetAnchorDataset(GraspDataset):

    def __init__(self,
                 labelroot,
                 graspnetroot,
                 sceneIds,
                 ratio,
                 anchor_k,
                 anchor_z,
                 anchor_w,
                 grasp_count,
                 sigma=10,
                 random_rotate=False,
                 random_zoom=False,
                 output_size=(640, 360),
                 include_rgb=True,
                 include_depth=True,
                 noise=0,
                 aug=False):
        logging.info('Using Graspnet dataset')
        # basic attributes
        self.trainning = True
        self.labelroot = labelroot
        self.graspnetroot = graspnetroot
        self.sceneIds = sceneIds
        self.random_rotate = random_rotate
        self.random_zoom = random_zoom
        self.include_rgb = include_rgb
        self.include_depth = include_depth
        # anchor size
        self.ratio = ratio
        self.anchor_k = anchor_k
        self.anchor_z = anchor_z
        self.anchor_w = anchor_w
        # grasp count
        self.grasp_count = grasp_count
        # gaussian kernel size
        self.sigma = sigma

        self.colorpath = []
        self.depthpath = []
        self.cameraposepath = []
        self.alignmatpath = []
        self.metapath = []
        self.grasppath = []
        self.sceneIds_str = ['scene_{}'.format(str(x)) for x in self.sceneIds]
        self.sceneIds_zfill = [
            'scene_{}'.format(str(x).zfill(4)) for x in self.sceneIds
        ]
        self.scenename = []
        self.frameid = []
        self.output_size = output_size
        self.noise = noise
        self.aug = None
        if aug:
            self.aug = Compose(
                [PILToTensor(),
                 ColorJitter(0.5, 0.5, 0.5, 0.3),
                 ToPILImage()])

        for x in tqdm(self.sceneIds_zfill, desc='Loading data path...'):
            self.cameraposepath.append(
                os.path.join(graspnetroot, 'scenes', x, camera,
                             'camera_poses.npy'))
            self.alignmatpath.append(
                os.path.join(graspnetroot, 'scenes', x, camera,
                             'cam0_wrt_table.npy'))
            for img_num in range(256):
                self.colorpath.append(
                    os.path.join(graspnetroot, 'scenes', x, camera, 'rgb',
                                 str(img_num).zfill(4) + '.png'))
                self.depthpath.append(
                    os.path.join(graspnetroot, 'scenes', x, camera, 'depth',
                                 str(img_num).zfill(4) + '.png'))
                self.metapath.append(
                    os.path.join(graspnetroot, 'scenes', x, camera, 'meta',
                                 str(img_num).zfill(4) + '.mat'))
                self.scenename.append(x.strip())
                self.frameid.append(img_num)

        for x in tqdm(self.sceneIds_str,
                      desc='Loading 6d grasp label path...'):
            for ann_num in range(256):
                self.grasppath.append(
                    os.path.join(labelroot, '6d_dataset', x, 'grasp_labels',
                                 '{}_view.npz'.format(ann_num)))


class GraspnetPointDataset(GraspnetAnchorDataset):

    def __init__(self, all_points_num, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # for pc convert
        self.all_points_num = all_points_num
        self.helper = PointCloudHelper(self.all_points_num)

    def get_camera_pose(self, index):
        camera_pose = np.load(self.cameraposepath[index // 256])
        align_mat = np.load(self.alignmatpath[index // 256])
        return align_mat @ camera_pose[index % 256]

    def __getitem__(self, index):
        # get anchor data
        anchor_data = super().__getitem__(index)

        # load image
        color_img = self.cur_rgb.astype(np.float32) / 255.0
        depth_img = self.cur_depth.astype(np.float32)

        # get grasp path
        grasp_path = self.grasppath[index]
        return anchor_data, color_img, depth_img, grasp_path
