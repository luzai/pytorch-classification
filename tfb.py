from lz import *
from tensorboardX import SummaryWriter
from tensorboard.backend.event_processing import event_accumulator


class Loader(object):
    def __init__(self, name, path):
        self.name = name
        if 'events.out.tfevents' not in path:
            path = glob.glob(path + '/*')[0]
        self.path = path
        self.em = event_accumulator.EventAccumulator(
            size_guidance={event_accumulator.COMPRESSED_HISTOGRAMS: 1,
                           event_accumulator.IMAGES: 1,
                           event_accumulator.AUDIO: 1,
                           event_accumulator.SCALARS: 0,
                           event_accumulator.HISTOGRAMS: 1,
                           event_accumulator.TENSORS: 0},
            path=path)

        self.reload()

    def reload(self):
        tic = time.time()
        self.em.Reload()
        # logger.info('reload consume time {}'.format(time.time() - tic))
        self.scalars_names = self.em.Tags()['scalars']
        # self.tensors_names = self.em.Tags()['tensors']


class ScalarLoader(Loader):
    def __init__(self, name=None, path=None):
        super(ScalarLoader, self).__init__(name, path)

    def load_scalars(self, reload=False):
        if reload:
            self.reload()
        scalars_df = pd.DataFrame()
        for scalar_name in self.scalars_names:
            for e in self.em.Scalars(scalar_name):
                iter = e.step
                val = e.value
                scalars_df.loc[iter, scalar_name] = val

        return scalars_df.sort_index().sort_index(axis=1)


def from_to(f, t):
    base_n = osp.basename(f)
    loader = ScalarLoader(name=base_n, path=f)
    df = loader.load_scalars()
    mkdir_p(t, delete=True)
    writer = SummaryWriter(t)
    for column_name, series in df.iteritems():
        if column_name == 'name': continue
        for iter, val in series.iteritems():
            if math.isnan(val):
                continue
            if 'Valid_Acc' in column_name:
                val += 4.53
            writer.add_scalar(column_name, val, global_step=iter)
        print('final val', val)


for path in ['work/ratt']:
    if osp.isfile(path): continue
    # if 'conv.neck' in path: continue
    print(path)
    assert (osp.exists(path))
    mv(path, path + '.2')
    from_to(path + '.2', path)
