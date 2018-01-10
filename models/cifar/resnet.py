from __future__ import absolute_import

'''Resnet for cifar dataset.
Ported form
https://github.com/facebook/fb.resnet.torch
and
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
(c) YANG, Wei
'''
import math
import numpy as np
import torch, torchvision
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

__all__ = ['resnet']


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
    # return TConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
    #                padding=1, bias=False, meta_kernel_size=6)


class TConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=0, bias=False, meta_kernel_size=4, args=None, compression_ratio=1):
        super(TConv2d, self).__init__()

        def get_controller(
                scale=(1, (meta_kernel_size - 1) / (meta_kernel_size - 3)),
                translation=(
                        (meta_kernel_size - kernel_size) / (meta_kernel_size - 1),
                        (meta_kernel_size - kernel_size) / (meta_kernel_size - 1) + 2 / (meta_kernel_size - 1)
                ),
                # 2 / (meta_kernel_size - 1)
                theta=(0,
                       np.pi / 8, -np.pi / 8,
                       np.pi / 4, -np.pi / 4)):
            controller = []
            for sx in scale:
                for sy in scale:
                    for tx in translation:
                        for ty in translation:
                            for th in theta:
                                controller.append([sx * np.cos(th), -sx * np.sin(th), tx,
                                                   sy * np.sin(th), sy * np.cos(th), ty])
            print('len weight is ', len(controller))
            controller = np.stack(controller)
            controller = controller.reshape(-1, 2, 3)
            controller = np.ascontiguousarray(controller, np.float32)
            return controller

        controller = get_controller()
        len_controller = len(controller)
        assert out_channels % compression_ratio == 0
        self.meta_kernel_size = meta_kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.FloatTensor(
            out_channels // compression_ratio, in_channels, meta_kernel_size, meta_kernel_size
        ))
        self.reset_parameters()
        self.register_buffer('theta', torch.FloatTensor(controller).view(-1, 2, 3).cuda())
        self.reduce_conv = nn.Conv2d(out_channels * len_controller // compression_ratio, out_channels, 1)

    def reset_parameters(self):
        n = self.in_channels * self.kernel_size * self.kernel_size
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input):
        # bs, in_channels, h, w = input.size()
        weight_l = []
        for theta_ in Variable(self.theta, requires_grad=False):
            grid = F.affine_grid(theta_.expand(self.weight.size(0), 2, 3), self.weight.size())
            weight_l.append(F.grid_sample(self.weight, grid))
        weight_inst = torch.cat(weight_l)
        weight_inst = weight_inst[:, :, :self.kernel_size, :self.kernel_size].contiguous()
        out = F.conv2d(input, weight_inst, stride=self.stride, padding=self.padding)
        out = self.reduce_conv(out)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class LocNet(nn.Module):
    def __init__(self):
        super(LocNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(20, 6)
        self.fc1.bias.data = torch.FloatTensor([
            1, 0, 0, 0, 1, 0
        ])
        # self.fc1.bias.requires_grad=False

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(self.conv2(x))
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


class ResNet(nn.Module):
    def __init__(self, depth, num_classes=1000):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        block = Bottleneck if depth >= 44 else BasicBlock

        # self.loc_net=LocNet()

        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # param = self.loc_net(x)
        # param = param.view(-1, 2, 3)
        # grid = F.affine_grid(param, x.size())
        # x = F.grid_sample(x, grid)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # 32x32

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return ResNet(**kwargs)
