"""resnet in pytorch.

[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.     Deep Residual
Learning for Image Recognition     https://arxiv.org/abs/1512.03385v1
"""

import math

import torch
import torch.nn as nn
from torch.nn.functional import dropout, embedding


class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34."""

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=3,
                      stride=stride,
                      padding=1,
                      bias=False), nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels,
                      out_channels * BasicBlock.expansion,
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion))

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels * BasicBlock.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion))

    def forward(self, x):
        return nn.LeakyReLU(inplace=True)(self.residual_function(x) +
                                          self.shortcut(x))


class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers."""
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels,
                      out_channels,
                      stride=stride,
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels,
                      out_channels * BottleNeck.expansion,
                      kernel_size=1,
                      bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels * BottleNeck.expansion,
                          stride=stride,
                          kernel_size=1,
                          bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion))

    def forward(self, x):
        return nn.LeakyReLU(inplace=True)(self.residual_function(x) +
                                          self.shortcut(x))


class ResNet(nn.Module):

    def __init__(self, block, num_blocks, in_dim=3, planes=64):
        super().__init__()
        print('planes:', planes)
        # mul 4 for BottleNeck to get the same dim as BasicBlock
        self.in_channels = planes * 4 if block is BottleNeck else planes
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_dim,
                      out_channels=self.in_channels,
                      kernel_size=7,
                      stride=2,
                      padding=3,
                      bias=False), nn.BatchNorm2d(self.in_channels),
            nn.LeakyReLU(inplace=True))
        self.conv2_x = self._make_layer(block, planes * 2, num_blocks[0], 2)
        self.conv3_x = self._make_layer(block, planes * 4, num_blocks[1], 2)
        self.conv4_x = self._make_layer(block, planes * 8, num_blocks[2], 2)
        self.conv5_x = self._make_layer(block, planes * 16, num_blocks[3], 2)
        embeddings = planes * 16 * block.expansion  # * 1 for resnet<50, * 4 for resnet>=50
        print('embeddings:', embeddings)

    def _make_layer(self, block, out_channels, num_block, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the same
        as a neuron netowork layer, ex.

        conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_block - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        o1 = self.conv1(x)
        o2 = self.conv2_x(o1)
        o3 = self.conv3_x(o2)
        o4 = self.conv4_x(o3)
        o5 = self.conv5_x(o4)
        return [o1, o2, o3, o4, o5]


def resnet18(**kwargs):
    """return a ResNet 18 object."""
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    """return a ResNet 34 object."""
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    """return a ResNet 50 object."""
    return ResNet(BottleNeck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    """return a ResNet 101 object."""
    return ResNet(BottleNeck, [3, 4, 23, 3], **kwargs)


def resnet152(**kwargs):
    """return a ResNet 152 object."""
    return ResNet(BottleNeck, [3, 8, 36, 3], **kwargs)
