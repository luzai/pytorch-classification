from lz import *
from torchvision.models.resnet import conv3x3, model_urls, model_zoo, ResNet
from .se_resnet import SELayer


# class BasicBlock(nn.Module):
#     expansion = 1
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(BasicBlock, self).__init__()
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = nn.BatchNorm2d(planes)
#         if downsample is not None:
#             self.downsample = nn.Sequential(
#                 nn.Conv2d(downsample[0], downsample[1],
#                           kernel_size=1, stride=downsample[2], bias=False),
#                 nn.BatchNorm2d(downsample[1]),
#             )
#         else:
#             self.downsample = None
#         self.stride = stride
#
#     def forward(self, x):
#         residual = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         out += residual
#         out = self.relu(out)
#
#         return out


class SERIRBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(SERIRBasicBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv = []
        self.bn = []
        for i in range(4):
            self.conv += [conv3x3(inplanes // 2, planes // 2, stride)]
            self.bn += [nn.BatchNorm2d(planes // 2)]

        for i in range(4, 8):
            self.conv += [conv3x3(planes // 2, planes // 2)]
            self.bn += [nn.BatchNorm2d(planes // 2)]
        self.se1 = SELayer(planes // 2)
        self.se2 = SELayer(planes // 2)
        for i in range(8):
            setattr(self, 'conv' + str(i), self.conv[i])
            setattr(self, 'bn' + str(i), self.bn[i])

        if downsample is not None:
            self.downsample = nn.Sequential(
                nn.Conv2d(downsample[0] // 2, downsample[1] // 2,
                          kernel_size=1, stride=downsample[2], bias=False),
                nn.BatchNorm2d(downsample[1] // 2),
            )
        else:
            self.downsample = None
        # self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual, transient = x[:, :x.size(1) // 2, :, :].contiguous(), x[:, x.size(1) // 2:, :, :].contiguous()
        func = lambda ind, x: getattr(self, f'bn{ind}')(getattr(self, f'conv{ind}')(x))
        out = [func(ind, residual) for ind in range(2)] + \
              [func(ind, transient) for ind in range(2, 4)]
        if self.downsample:
            residual = self.downsample(residual)

        residual, transient = (residual + out[0] + out[2],
                               out[1] + out[3])
        residual, transient = map(self.relu, (residual, transient))

        func = lambda ind, x: getattr(self, f'bn{ind}')(getattr(self, f'conv{ind}')(x))
        out = [func(ind, residual) for ind in range(4, 6)] + \
              [func(ind, transient) for ind in range(6, 8)]
        out[2] = self.se1(out[2])
        out[3] = self.se2(out[3])
        residual, transient = (residual + out[0] + out[2],
                               out[1] + out[3])
        residual, transient = map(self.relu, (residual, transient))
        return torch.cat((residual, transient), dim=1).contiguous()


class SERIRResNet(nn.Module):
    __factory = {
        '18': [2, 2, 2, 2],
        '34': [3, 4, 6, 3],
        '50': [3, 4, 6, 3],
        '101': [3, 4, 23, 3],
        '152': [3, 8, 36, 3],
    }

    def _make_layer(self, block, planes, blocks, stride=1):
        if not isinstance(block, list):
            block = [block] * blocks
        downsample = None
        if stride != 1 or self.inplanes != planes * block[0].expansion:
            downsample = (self.inplanes, planes * block[0].expansion, stride)

        layers = []
        layers.append(block[0](self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block[0].expansion
        for i in range(1, blocks):
            layers.append(block[i](self.inplanes, planes))

        return nn.Sequential(*layers)

    def __init__(self, depth=50, pretrained=True,
                 block_name='SERIRBasicBlock',
                 block_name2='SERIRBasicBlock',
                 num_deform=3, num_classes =1000,
                 **kwargs):
        super(SERIRResNet, self).__init__()
        depth = str(depth)
        self.depth = depth
        self.inplanes = 64
        self.pretrained = pretrained

        # Construct base (pretrained) SERIRResNet
        if depth not in SERIRResNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        layers = SERIRResNet.__factory[depth]
        block = eval(block_name)
        block2 = eval(block_name2)
        self.out_planes = 512 * block2.expansion
        logging.info(f'out_planes is {self.out_planes}')
        self.conv1 = nn.Conv2d(3, 64,
                               kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        if num_deform > 3:
            self.layer3 = self._make_layer([block] * 3 + [block2] * (num_deform - 3), 256, layers[2],
                                           stride=2)
            self.layer4 = self._make_layer([block2] * 3, 512, layers[3], stride=2)
        else:
            self.layer3 = self._make_layer([block] * 6, 256, layers[2], stride=2)
            self.layer4 = self._make_layer([block2] * num_deform + [block] * (3 - num_deform), 512,
                                           layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        # if pretrained:
        #     logging.info('load resnet')
        #     load_state_dict(self, model_zoo.load_url(model_urls['resnet{}'.format(depth)]))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def serir_resnet50(num_classes=1000, **kwargs):
    # model = ResNet(SERIRBasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    model = SERIRResNet(depth=50, pretrained=True,
                        num_classes=1000, block_name='SERIRBasicBlock',
                        block_name2='SERIRBasicBlock',
                        num_deform=3, )
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    # logging.info('load resnet')
    # load_state_dict(model, model_zoo.load_url(model_urls['resnet{}'.format(50)]))
    return model
