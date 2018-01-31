from lz import *

confs = [
    # edict(
    #     arch='resnet',
    #     depth=56,
    #     epochs=164,
    #     wd=1e-4, weight_decay=1e-4, lr=1e-1,
    #     checkpoint='work/res.56.1e-4.2',
    #     resume='',
    #     evaluate=False,
    #     conv_op='nn.Conv2d',
    #     bottleneck='Bottleneck',
    #     meta_stride=1,
    #     comp=1,
    # ),
edict(
        arch='resnet',
        depth=56,
        epochs=164,
        wd=5e-4, weight_decay=5e-4, lr=1e-1,
        checkpoint='work/res.56.1e-4.2',
        resume='',
        evaluate=False,
        conv_op='nn.Conv2d',
        bottleneck='Bottleneck',
        meta_stride=1,
        comp=1,
    ),
    # edict(
    #     arch='resnet',
    #     depth=56,
    #     epochs=164,
    #     checkpoint='work/res.56.1e-4.0.9999',
    #     resume='',
    #     evaluate=False,
    #     conv_op='nn.Conv2d',
    #     bottleneck='Bottleneck',
    #     meta_stride=1,
    #     comp=1,
    # ),

    # edict(
    #     arch='res_att1',
    #     depth=56,
    #     epochs=164,
    #     train_batch=128,
    #     test_batch=128,
    #     schedule=[81, 122],
    #     gamma=0.1,
    #     wd=1e-4,
    #     weight_decay=1e-4,
    #     lr=1e-2,
    #     checkpoint='work/res.att.deeper',
    #     resume='',
    #     evaluate=False,
    #     conv_op='nn.Conv2d',
    #     bottleneck='Bottleneck',
    #     meta_stride=1,
    #     comp=1,
    # ),

]
chs = -1
conf = confs[chs]
mkdir_p('work')
with open(conf.checkpoint + '.chs', 'w') as f:
    f.write('model {} use conf chs {} \n'.format(conf.checkpoint, chs))
