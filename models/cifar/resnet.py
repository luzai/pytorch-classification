from __future__ import absolute_import
from lz import *

'''Resnet for cifar dataset.
Ported form
https://github.com/facebook/fb.resnet.torch
and
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
(c) YANG, Wei
'''
import torch.nn as nn
import math

__all__ = ['resnet', 'res_att1']

nn.Conv2d = wrapped_partial(nn.Conv2d, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


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


class ResNet(nn.Module):

    def __init__(self, depth, num_classes=1000, block_name='BasicBlock'):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        if block_name.lower() == 'basicblock':
            assert (depth - 2) % 6 == 0, 'When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202'
            n = (depth - 2) // 6
            block = BasicBlock
        elif block_name.lower() == 'bottleneck':
            assert (depth - 2) % 9 == 0, 'When use bottleneck, depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199'
            n = (depth - 2) // 9
            block = Bottleneck
        else:
            raise ValueError('block_name shoule be Basicblock or Bottleneck')

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
            if isinstance(m, nn.Conv2d.func):
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


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(Residual, self).__init__()
        self.downsample = downsample

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels // 4, 1)
        self.bn2 = nn.BatchNorm2d(out_channels // 4)
        self.conv2 = nn.Conv2d(out_channels // 4,
                               out_channels // 4, kernel_size=3, padding=1,
                               stride=1 if not downsample else 2,
                               )
        self.bn3 = nn.BatchNorm2d(out_channels // 4)
        self.conv3 = nn.Conv2d(out_channels // 4, out_channels, 1)
        if downsample:
            self.conv4 = nn.Conv2d(in_channels, out_channels, 1,
                                   stride=2)
        elif in_channels != out_channels:
            self.conv4 = nn.Conv2d(in_channels, out_channels, 1, stride=1)
        else:
            self.conv4 = None

    def forward(self, x):
        # todo
        if self.conv4 is not None:
            residual =x
            residual = self.conv4(residual)
        else:
            residual = x
            x = F.relu(self.bn1(x))
            x = self.conv1(x)
            x = F.relu(self.bn2(x))
            x = self.conv2(x)
            x = F.relu(self.bn3(x))
            x = self.conv3(x)


        return x + residual


class Attention(nn.Module):
    def __init__(self, in_channels, out_channels, unet_rep=1):
        super(Attention, self).__init__()
        self.residual1 = Residual(in_channels, in_channels)
        self.unet = nn.Sequential(Unet(in_channels, rep=unet_rep),
                                  nn.BatchNorm2d(in_channels), nn.ReLU(),
                                  nn.Conv2d(in_channels, in_channels, 1),
                                  nn.BatchNorm2d(in_channels), nn.ReLU(),
                                  nn.Conv2d(in_channels, in_channels, 1),
                                  nn.Sigmoid()
                                  )
        self.trunk = nn.Sequential(Residual(in_channels, in_channels),
                                   Residual(in_channels, in_channels))
        self.residual2 = Residual(in_channels, in_channels)

    def forward(self, x):
        x = self.residual1(x)
        x1 = self.trunk(x)
        x2 = self.unet(x)
        x3 = x1 * x2 + x1
        return self.residual2(x3)


class ResAtt1(nn.Module):
    def __init__(self, **kwargs):
        super(ResAtt1, self).__init__()

        self.conv1 = nn.Conv2d(3, 16,
                               kernel_size=3, stride=1, padding=1)

        self.residual1 = nn.Sequential(
            Residual(16, 32),
        )

        self.attention1 = nn.Sequential(
            Attention(32, 32, unet_rep=2),
            Attention(32, 32, unet_rep=2),
        )

        self.residual2 = nn.Sequential(
            Residual(32, 64, downsample=False),
        )
        self.attention2 = nn.Sequential(
            Attention(64, 64, unet_rep=2),
            Attention(64, 64, unet_rep=2),
        )
        self.residual3 = nn.Sequential(
            Residual(64, 128, downsample=True),
        )
        self.attention3 = nn.Sequential(
            Attention(128, 128, unet_rep=1),
            Attention(128, 128, unet_rep=1),
        )
        self.residual4 = nn.Sequential(
            Residual(128, 128, downsample=True),
            Residual(128, 128),
            Residual(128, 128),
        )
        self.fc1 = nn.Linear(128, 10)
        self.bn1 = nn.BatchNorm1d(128)
        self.init_model()

    def init_model(self):
        from torch.nn import init
        for m in self.modules():
            if isinstance(m, nn.Conv2d.func):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.residual1(x)
        x = self.attention1(x)
        x = self.residual2(x)
        x = self.attention2(x)
        x = self.residual3(x)
        x = self.attention3(x)
        x = self.residual4(x)
        x = F.avg_pool2d(x, x.size()[-2:])
        x = x.view(x.size(0), -1)
        x = self.bn1(x)
        x=F.relu(x)
        x = self.fc1(x)
        return x


class Unet(nn.Module):
    def __init__(self, nc, rep=1):
        super(Unet, self).__init__()
        # unet_block = UnetBlock(nc, innermost=True)
        # for i in range(rep - 1):
        #     unet_block = UnetBlock(nc, submodule=unet_block)
        if rep == 1:
            unet_block = UnetBlock(nc, innermost=True, outermost=True)
        elif rep == 2:
            unet_block = UnetBlock(nc, innermost=True)
            unet_block = UnetBlock(nc, outermost=True, submodule=unet_block)
        elif rep >= 3:
            raise NotImplementedError('wait')
        self.model = unet_block

    def forward(self, x):
        return self.model(x)


class UnetBlock(nn.Module):
    def __init__(self, nc,
                 submodule=None, innermost=False, outermost=False,
                 ):
        super(UnetBlock, self).__init__()
        downconv = Residual(nc, nc)
        upconv = Residual(nc, nc)
        down = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.innermost = innermost
        self.outermost = outermost
        if innermost:
            model = [down, downconv, upconv, up]
        else:
            model = [down, downconv, submodule, upconv, up]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return x + self.model(x)


def res_att1(**kwargs):
    return ResAtt1(**kwargs)


if __name__ == '__main__':
    net = ResAtt1().cuda()
    var = Variable(torch.rand(1, 3, 224, 224).cuda(), requires_grad=True)
    loss = net(var).sum()
