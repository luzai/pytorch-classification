from lz import *

conf = edict(
    arch='se_resnet50',
    depth=50,
    schedule=[31, 61],
    gamma=0.1,
    data='//share/ILSVRC2012_imgdataset/',
    worker=32, epochs=90, train_batch=128, test_batch=100,
    checkpoint='work/se_res.cont',
    resume='work/se_res/checkpoint.pth.tar',
    evaluate=False,
    gpu_id='2,3'
)

# conf = edict(
#     arch='resnet',
#     depth=56,
#     epochs=164,
#     wd=1e-4, weight_decay=1e-4, lr=1e-1,
#     checkpoint='work/res.56.1e-4.nest',
#     resume='',
#     evaluate=False,
# )
#
# conf = edict(
#     arch='res_att1',
#     depth=56,
#     epochs=164, train_batch=128, test_batch=128,
#     wd=1e-4, weight_decay=1e-4, lr=1e-1,
#     checkpoint='work/res.att.221.2',
#     resume='',
#     evaluate=False,
# )
#
# conf = edict(
#     arch='res_att1',
#     depth=56,
#     epochs=164,
#     wd=1e-4, weight_decay=1e-4, lr=1e-1,
#     checkpoint='work/res.att.222',
#     resume='',
#     evaluate=False,
# )
#
# conf = edict(
#     arch='res_att1',
#     depth=56,
#     epochs=164,
#     wd=1e-4, weight_decay=1e-4, lr=1e-1,
#     checkpoint='work/res.att.64',
#     resume='',
#     evaluate=False,
# )
#
#
# conf = edict(
#     arch='resnet',
#     depth=56,
#     epochs=164,
#     wd=1e-4, weight_decay=1e-4, lr=1e-1,
#     checkpoint='work/res.1e-4',
#     resume='',
#     evaluate=False,
# )
#
#
# conf = edict(
#     arch='resnet',
#     depth=56,
#     epochs=164,
#     wd=5e-4, weight_decay=5e-4, lr=1e-1,
#     checkpoint='work/res.5e-4',
#     resume='',
#     evaluate=False,
# )
