from lz import *
from conf import conf


def cnt_param(weight):
    return sum(weight.numel())

global_compression_ratio = conf.get('comp', 1)


# move stack + reduce conv/avg pool
class TC1Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=False, meta_kernel_size=4,
                 compression_ratio=global_compression_ratio, mode='train'):
        super(TC1Conv, self).__init__()
        while out_channels % compression_ratio != 0:
            compression_ratio += 1
            if compression_ratio >= out_channels:
                compression_ratio = 1
                break
        assert out_channels % compression_ratio == 0
        self.meta_kernel_size = kernel_size + 1
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.mode = mode

        self.weight = nn.Parameter(torch.randn(
            out_channels // compression_ratio, in_channels, meta_kernel_size, meta_kernel_size
        ))
        len_controller = (self.meta_kernel_size - self.kernel_size + 1) ** 2
        self.reduce_conv = nn.Conv2d(out_channels * len_controller // compression_ratio,
                                     out_channels, kernel_size=1)

        self.reset_parameters()

    def get_weight_inst(self):
        weight_l = []

        for i in range(self.meta_kernel_size - self.kernel_size + 1):
            for j in range(self.meta_kernel_size - self.kernel_size + 1):
                weight_l.append(self.weight[:, :, i:i + self.kernel_size, j:j + self.kernel_size])
        weight_inst = torch.cat(weight_l).contiguous()
        return weight_inst

    def reset_parameters(self):
        def reset_w(weight):
            out_chl, in_chl, w, h = weight.size()
            n = in_chl * w * h
            stdv = 1. / math.sqrt(n)
            weight.data.uniform_(-stdv, stdv)

        reset_w(self.weight)
        reset_w(self.reduce_conv.weight)

    def forward(self, input):
        bs, chl, h, w = input.size()
        weight_inst = self.get_weight_inst()
        out = F.conv2d(input, weight_inst, stride=self.stride, padding=self.padding)
        bs, _, oh, ow = out.size()
        out = self.reduce_conv(out)
        # out = out.view(bs, len(weight_l), self.out_channels, oh, ow)
        # out = out.permute(0, 2, 3, 4, 1).contiguous().view(bs, -1, len(weight_l))
        # out = F.avg_pool1d(out, len(weight_l))
        # out = out.permute(0, 2, 1).contiguous().view(bs, -1, oh, ow)

        return out


# 1x1 conv generate weight
class C1Conv(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride=1, padding=0, bias=False, meta_kernel_size=4,
                 compression_ratio=global_compression_ratio, mode='train'):
        super(C1Conv, self).__init__()
        while out_channels % compression_ratio != 0:
            compression_ratio += 1
            if compression_ratio >= out_channels:
                compression_ratio = 1
                break
        assert out_channels % compression_ratio == 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.meta_kernel_size = kernel_size + 1
        self.mode = mode
        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(torch.FloatTensor(
            out_channels // compression_ratio, in_channels, kernel_size, kernel_size
        ))
        self.trans_weight = nn.Parameter(torch.FloatTensor(
            out_channels, out_channels, 1, 1
        ))
        self.reset_parameters()

    def reset_parameters(self):
        def reset_w(weight):
            out_chl, in_chl, w, h = weight.size()
            n = in_chl * w * h
            stdv = 1. / math.sqrt(n)
            weight.data.uniform_(-stdv, stdv)

        reset_w(self.weight)
        reset_w(self.trans_weight)

    def get_weight_inst(self):
        weight_inst = F.conv2d(
            self.weight.permute(1, 0, 2, 3).contiguous(),
            self.trans_weight
        ).permute(1, 0, 2, 3).contiguous()
        return weight_inst

    def forward(self, input):
        bs, chl, h, w = input.size()
        if self.mode == 'test' and hasattr(self, 'weight_inst'):
            weight_inst = self.weight_inst
        elif self.mode == 'test' and not hasattr(self, 'weight_inst'):
            weight_inst = self.get_weight_inst()
            self.weight_inst = weight_inst
        else:
            weight_inst = self.get_weight_inst()
        out = F.conv2d(input, weight_inst, stride=self.stride, padding=self.padding)
        bs, _, oh, ow = out.size()
        return out


# double 1x1 conv generate weight
class C1C1Conv(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride=1, padding=0, bias=False, meta_kernel_size=4,
                 compression_ratio=global_compression_ratio, mode='train'):
        super(C1C1Conv, self).__init__()
        while out_channels % compression_ratio != 0:
            compression_ratio += 1
            if compression_ratio >= out_channels:
                compression_ratio = 1
                break
        assert out_channels % compression_ratio == 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.meta_kernel_size = kernel_size + 1
        self.mode = mode
        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(torch.FloatTensor(
            out_channels // compression_ratio, in_channels, kernel_size, kernel_size
        ))
        self.trans_weight = nn.Parameter(torch.FloatTensor(
            out_channels, out_channels, 1, 1
        ))
        self.trans_weight2 = nn.Parameter(torch.FloatTensor(
            out_channels, out_channels, 1, 1
        ))
        self.reset_parameters()

    def reset_parameters(self):
        def reset_w(weight):
            out_chl, in_chl, w, h = weight.size()
            n = in_chl * w * h
            stdv = 1. / math.sqrt(n)
            weight.data.uniform_(-stdv, stdv)

        reset_w(self.weight)
        reset_w(self.trans_weight)
        reset_w(self.trans_weight2)

    def get_weight_inst(self):
        weight_inst = F.conv2d(
            self.weight.permute(1, 0, 2, 3).contiguous(),
            self.trans_weight
        ).permute(1, 0, 2, 3).contiguous()
        weight_inst = F.conv2d(
            weight_inst.permute(1, 0, 2, 3).contiguous(),
            self.trans_weight2
        ).permute(1, 0, 2, 3).contiguous()
        return weight_inst

    def forward(self, input):
        bs, chl, h, w = input.size()
        # if self.mode == 'test' and hasattr(self, 'weight_inst'):
        #     weight_inst = self.weight_inst
        # elif self.mode == 'test' and not hasattr(self, 'weight_inst'):
        #     weight_inst = self.get_weight_inst()
        #     self.weight_inst = weight_inst
        # else:
        weight_inst = self.get_weight_inst()

        out = F.conv2d(input, weight_inst, stride=self.stride, padding=self.padding)

        bs, _, oh, ow = out.size()
        return out


# zeropad + 1x1 conv generate weight
class ZPC1Conv(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride=1, padding=0, bias=False, meta_kernel_size=4,
                 compression_ratio=global_compression_ratio, mode='train'):
        super(ZPC1Conv, self).__init__()
        while out_channels % compression_ratio != 0:
            compression_ratio += 1
            if compression_ratio >= out_channels:
                compression_ratio = 1
                break
        assert out_channels % compression_ratio == 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.meta_kernel_size = kernel_size + 1
        self.mode = mode
        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(torch.randn(
            out_channels // compression_ratio, in_channels, kernel_size, kernel_size
        ))
        self.trans_weight = nn.Parameter(torch.randn(
            out_channels,
            out_channels // compression_ratio * len(range(0, 3, conf.get('meta_stride', 2))) ** 2,
            1, 1
        ))
        self.reset_parameters()

    def get_param(self):
        return cnt_param(self.weight),cnt_param(self.trans_weight)

    def reset_parameters(self):
        def reset_w(weight):
            out_chl, in_chl, w, h = weight.size()
            n = in_chl * w * h
            stdv = 1. / math.sqrt(n)
            weight.data.uniform_(-stdv, stdv)

        reset_w(self.weight)
        reset_w(self.trans_weight)

    def get_weight_inst(self):
        weight_meta = F.pad(self.weight, (1, 1, 1, 1), mode='constant', value=0)
        weight_l = []
        for i in range(0, 3, conf.get('meta_stride', 2)):
            for j in range(0, 3, conf.get('meta_stride', 2)):
                weight_l.append(weight_meta[:, :, i:i + 3, j:j + 3])
        weight_inst = torch.cat(weight_l).contiguous()
        weight_inst = F.conv2d(
            weight_inst.permute(1, 0, 2, 3).contiguous(),
            self.trans_weight
        ).permute(1, 0, 2, 3).contiguous()
        return weight_inst

    def forward(self, input):
        bs, chl, h, w = input.size()
        weight_inst = self.get_weight_inst()
        out = F.conv2d(input, weight_inst, stride=self.stride, padding=self.padding)
        bs, _, oh, ow = out.size()
        return out


if __name__ == '__main__':
    # torch.backends.benchmark=True

    demo1 = nn.Sequential(
        nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=1, padding=0),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True),

        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True),

        nn.Conv2d(512, 2048, 1),
        nn.BatchNorm2d(2048),
        nn.ReLU(inplace=True)
    ).cuda()

    demo2 = nn.Sequential(
        nn.Conv2d(2048, 512, 1),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True),

        nn.Conv2d(512, 2048, kernel_size=3, padding=1),
        nn.BatchNorm2d(2048),
        nn.ReLU(inplace=True),
    ).cuda()

    flops_sum = 0


    def flops(demo1):
        global flops_sum
        flops_sum = 0

        def flops_layers(self, input, output):
            global flops_sum
            # print('Inside ' + self.__class__.__name__ + ' forward')
            bs, chl, h, w = input[0].size()
            ochl, ichl, kh, kw = self.weight.size()
            # print(flops_sum)
            flops_sum += ichl * ochl * h * w * kh * kw
            # print(flops_sum)

        for key, module in demo1._modules.items():
            if 'Conv2d' in module.__class__.__name__:
                module.register_forward_hook(flops_layers)
            # if 'Linear' in module.__class__.__name__:
            #     module.register_forward_hook(flops_layers)

        _ = demo1(Variable(torch.randn(128, 2048, 14, 14), requires_grad=True).cuda())
        print(flops_sum / 10 ** 9, 'G')


    flops(demo1)
    flops(demo2)


    def forward_time(demo1):
        inp = Variable(torch.randn(128, 2048, 14, 14), requires_grad=False).cuda()
        for _ in range(10):
            demo1(inp)
        # start = time.time()
        start = time.clock()
        ntries = 100
        for _ in range(ntries):
            demo1(inp)
        # print(time.time()-start)
        print((time.clock() - start) / ntries)


    forward_time(demo1)
    forward_time(demo2)
