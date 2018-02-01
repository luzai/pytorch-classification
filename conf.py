from lz import *

conf = edict(
    arch='resnet',
    depth=56,
    epochs=164,
    wd=1e-4, weight_decay=1e-4, lr=1e-1,
    checkpoint='work/res.56.1e-4',
    resume='',
    evaluate=False,
)

conf = edict(
    arch='resnet',
    depth=56,
    epochs=164,
    wd=1e-4, weight_decay=1e-4, lr=1e-1,
    checkpoint='work/res.56.1e-4.nest',
    resume='',
    evaluate=False,
)

conf = edict(
    arch='res_att1',
    depth=56,
    epochs=164, train_batch=128, test_batch=128,
    wd=1e-4, weight_decay=1e-4, lr=1e-1,
    checkpoint='work/res.att.221.2',
    resume='',
    evaluate=False,
)

conf = edict(
    arch='res_att1',
    depth=56,
    epochs=164,
    wd=1e-4, weight_decay=1e-4, lr=1e-1,
    checkpoint='work/res.att.222',
    resume='',
    evaluate=False,
)

conf = edict(
    arch='res_att1',
    depth=56,
    epochs=164,
    wd=1e-4, weight_decay=1e-4, lr=1e-1,
    checkpoint='work/res.att.64',
    resume='',
    evaluate=False,
)
