from __future__ import absolute_import

import math
import numpy as np
import torch, torchvision
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import init

__all__ = ['dcnn']


def _make_conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1,
               bias=False, with_relu=True):
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                     stride=stride, padding=padding, bias=bias)
    init.kaiming_normal(conv.weight, mode='fan_out')
    if bias:
        init.constant(conv.bias, 0)

    bn = nn.BatchNorm2d(out_planes)
    init.constant(bn.bias, 0)
    init.constant(bn.weight, 1)

    if with_relu:
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(conv, bn, relu)
    else:
        return nn.Sequential(conv, bn)


class DCNN(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(DCNN,self).__init__()
        self.base = nn.Sequential(
            _make_conv(3, 128),
            _make_conv(128, 128),
            nn.MaxPool2d(2),
            nn.Dropout2d(),

            _make_conv(128, 128),
            _make_conv(128, 128),
            nn.MaxPool2d(2),
            nn.Dropout2d(),

            _make_conv(128, 128),
            _make_conv(128, 128),
            nn.MaxPool2d(2),
            nn.Dropout2d(),

            _make_conv(128, 128),
            _make_conv(128, 128),
            nn.MaxPool2d(2),
            nn.Dropout2d(),

        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def dcnn(**kwargs):
    return DCNN(**kwargs)



