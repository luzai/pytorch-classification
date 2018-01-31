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
    epochs=164,
    wd=1e-4, weight_decay=1e-4, lr=1e-2,
    checkpoint='work/res.att.221',
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
    checkpoint='work/res.att.221.deeper',
    resume='',
    evaluate=False,
)
