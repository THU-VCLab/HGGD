import logging
import os

import numpy as np
import torch
from torchvision.transforms import (ColorJitter, Compose, GaussianBlur,
                                    PILToTensor, ToPILImage)
from tqdm import tqdm

from .base_grasp_dataset import GraspDataset
from .utils import PointCloudHelper


def gauss_noise_tensor(img):
    assert isinstance(img, torch.Tensor)
    dtype = img.dtype
    if not img.is_floating_point():
        img = img.to(torch.float32)
    sigma = 25.0
    out = img + np.random.uniform() * sigma * torch.randn_like(img)
    if out.dtype != dtype:
        out = out.to(dtype)
    return out


class AcronymAnchorDataset(GraspDataset):

    def __init__(self,
                 datapath,
                 sceneIds,
                 ratio,
                 anchor_k,
                 anchor_z,
                 anchor_w,
                 grasp_count,
                 sigma=5,
                 random_rotate=False,
                 random_zoom=False,
                 output_size=(640, 360),
                 include_rgb=True,
                 include_depth=True,
                 noise=0,
                 anno_cnt=50,
                 aug=True):
        logging.info('Using Acronym dataset')
        # basic attributes
        self.trainning = True
        self.datapath = datapath
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
        self.pcpath = []
        self.infopath = []
        self.grasppath = []
        self.sceneIds_zfill = [
            'scene_{}'.format(str(x).zfill(4)) for x in self.sceneIds
        ]
        self.scenename = []
        self.frameid = []
        self.output_size = output_size
        self.noise = noise
        self.aug = None
        self.anno_cnt = anno_cnt
        if aug:
            self.aug = Compose([
                PILToTensor(),
                ColorJitter(0.5, 0.5, 0.5, 0.3), gauss_noise_tensor,
                ToPILImage()
            ])

        for x in tqdm(self.sceneIds_zfill, desc='Loading data path...'):
            for img_num in range(self.anno_cnt):
                self.colorpath.append(
                    os.path.join(datapath, x, 'color',
                                 str(img_num).zfill(4) + '.png'))
                self.depthpath.append(
                    os.path.join(datapath, x, 'depth',
                                 str(img_num).zfill(4) + '.png'))
                self.pcpath.append(
                    os.path.join(datapath, x, 'pc',
                                 str(img_num).zfill(4) + '.npz'))
                self.infopath.append(
                    os.path.join(datapath, x, 'info',
                                 str(img_num).zfill(4) + '.npz'))
                self.scenename.append(x.strip())

                self.grasppath.append(
                    os.path.join(datapath, x, 'grasp',
                                 str(img_num).zfill(4) + '.npz'))
                self.frameid.append(img_num)


class AcronymPointDataset(AcronymAnchorDataset):

    def __init__(self, all_points_num, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # for pc convert
        self.all_points_num = all_points_num
        self.helper = PointCloudHelper(self.all_points_num)

    def __getitem__(self, index):
        # get 2d data
        anchor_data = super().__getitem__(index)

        # load image
        color_img = self.cur_rgb.astype(np.float32) / 255.0
        depth_img = self.cur_depth.astype(np.float32)

        color_img = torch.from_numpy(color_img)
        depth_img = torch.from_numpy(depth_img)

        camera_pose = np.load(self.infopath[index])['camera_pose']
        pc_path = self.pcpath[index]

        # get grasp path
        grasp_path = self.grasppath[index]

        return anchor_data, color_img, depth_img, pc_path, grasp_path, camera_pose
