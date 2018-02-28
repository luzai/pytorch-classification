from lz import *

ngpu = 4

confs = [
    # edict(
    #     arch='se_resnet50',
    #     depth=50,
    #     schedule=[31, 61],
    #     gamma=0.1,
    #     data='/share/ILSVRC2012_imgdataset/',
    #     worker=32, epochs=90, train_batch=64*ngpu, test_batch=50*ngpu,
    #     checkpoint='work/se_res.cont',
    #     resume='work/se_res/checkpoint.pth.tar',
    #     # resume='work/se_res/model_best.pth.tar',
    #     evaluate=False,
    #     gpu_id='2,3'
    # )

    edict(
        arch='serir_resnet50',
        depth=50,
        schedule=[31, 61],
        gamma=0.1,
        data='/share/ILSVRC2012_imgdataset/',
        worker=32, epochs=90, train_batch=64*ngpu, test_batch=50*ngpu,
        checkpoint='work/serir_res.cont',
        resume='work/serir_res/checkpoint.pth.8',
        evaluate=False,
        gpu_id='2,3'
    )

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
    # edict(
    #     arch='resnet',
    #     depth=56,
    #     epochs=164,
    #     wd=5e-4, weight_decay=5e-4, lr=1e-1,
    #     checkpoint='work/res.56.1e-4',
    #     resume='',
    #     evaluate=False,
    #     conv_op='nn.Conv2d',
    #     bottleneck='Bottleneck',
    #     meta_stride=1,
    #     comp=1,
    # ),
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
# mkdir_p('work')
# with open(conf.checkpoint + '.chs', 'w') as f:
#     f.write('model {} use conf chs {} \n'.format(conf.checkpoint, chs))
