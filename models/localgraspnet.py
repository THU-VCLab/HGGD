import math
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import knn_points

from .pointnet import PointNetfeat


class PointMultiGraspNet(nn.Module):

    def __init__(self, info_size, k_cls):
        super().__init__()
        self.k_cls = k_cls
        self.pointnet = PointNetfeat(feature_len=3)
        self.point_layer = nn.Sequential(nn.Linear(1024, 512),
                                         nn.LayerNorm(512), nn.Dropout(0.3),
                                         nn.ReLU(inplace=True))
        self.info_layer = nn.Linear(info_size, 32)
        self.anchor_mlp = nn.Sequential(nn.Linear(512 + 32, 256),
                                        nn.LayerNorm(256),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(256, k_cls))
        self.offset_mlp = nn.Sequential(nn.Linear(512 + 32, 256),
                                        nn.LayerNorm(256),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(256, k_cls * 3))

        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, points, info):
        # fuse pixel to points
        points = points.transpose(1, 2)
        # pointnet
        features = self.pointnet(points)
        # mlp
        point_features = self.point_layer(features)
        info_features = self.info_layer(info)
        x = torch.cat([point_features, info_features], 1)
        # get anchors and offset
        pred = self.anchor_mlp(x)
        offset = self.offset_mlp(x).view(-1, self.k_cls, 3)
        return features, pred, offset
