from __future__ import absolute_import

import math
import numpy as np
import torch, torchvision
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import init

__all__ = ['dcnn']


class TConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=0, bias=False, meta_kernel_size=6, compression_ratio=1):
        super(TConv2d, self).__init__()

        def get_controller(
                scale=(1,
                       ),
                translation=(0,
                        # 2 / (meta_kernel_size - 1)
                             ),
                theta=(0,
                       np.pi,
                       np.pi / 8, -np.pi / 8,
                       np.pi / 2, -np.pi / 2,
                       np.pi / 4, -np.pi / 4,
                       np.pi * 3 / 4, -np.pi * 3 / 4,
                       )):
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

        while out_channels % compression_ratio != 0:
            compression_ratio += 1
            if compression_ratio >= out_channels:
                compression_ratio = 1
                break
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


class DoubleConv2(nn.Module):
    def __init__(self, in_plates=2048, out_plates=512, meta_kernel_size=4, kernel_size=3, stride=2, padding=1, bias=False):
        super(DoubleConv2, self).__init__()
        self.in_plates, self.out_plates, self.zmeta, self.z, self.stride = in_plates, out_plates, meta_kernel_size, kernel_size, stride
        self.weight = nn.Parameter(torch.FloatTensor(out_plates, in_plates, meta_kernel_size, meta_kernel_size))
        self.reset_parameters()

        self.n_inst, self.n_inst_sqrt = (self.zmeta - self.z + 1) * (self.zmeta - self.z + 1), self.zmeta - self.z + 1

    def reset_parameters(self):
        n = self.out_plates * self.zmeta * self.zmeta
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input):
        self.h, self.w = input.size(2), input.size(3)
        weight = []
        for i in range(self.zmeta - self.z + 1):
            for j in range(self.zmeta - self.z + 1):
                weight.append(self.weight[:, :, i:i + self.z, j:j + self.z])
        n_inst = len(weight)
        n_inst_sqrt = int(math.sqrt(n_inst))
        weight_inst = torch.cat(weight)
        # self.weight_inst = weight_inst

        bs = input.size(0)
        out = F.conv2d(input, weight_inst, padding=1)
        # self.out_inst = out
        out = out.view(bs, n_inst_sqrt, n_inst_sqrt, self.out_plates, self.h, self.w)
        out = out.permute(0, 3, 4, 5, 1, 2).contiguous().view(bs, -1, n_inst_sqrt, n_inst_sqrt)
        out = F.avg_pool2d(out, (self.zmeta - self.z + 1, self.zmeta - self.z + 1))
        out = out.permute(0, 2, 3, 1).contiguous().view(bs, -1, self.h, self.w)
        # self.out=out
        # out = F.avg_pool2d(out, out.size()[2:])
        # out = out.view(out.size(0), -1)
        return out


def _make_conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1,
               bias=False, with_relu=True):
    conv = TConv2d(in_planes, out_planes, kernel_size=kernel_size,
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
        super(DCNN, self).__init__()
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
