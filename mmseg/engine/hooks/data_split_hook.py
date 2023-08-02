
from pathlib import Path
import json
from logging import WARNING
from itertools import combinations

from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.logging import print_log

from mmseg.registry import HOOKS


@HOOKS.register_module()
class DataSplitHook(Hook):

    def __init__(self, assert_no_overlap: bool = True, log_split: bool = True):
        self.assert_no_overlap = assert_no_overlap
        self.log_split = log_split

    def before_run(self, runner: Runner) -> None:
        def get_name(imgpath):
            if isinstance(imgpath, tuple):
                return tuple([Path(p).name for p in imgpath])
            return Path(imgpath).name

        dset_val = runner.val_dataloader.dataset
        dset_train = runner.train_dataloader.dataset
        dset_test = runner.test_dataloader.dataset

        info = dict()
        for dset, desc in [(dset_train, 'train'), (dset_val, 'val'), (dset_test, 'test')]:
            buffer = list()
            for idx in range(len(dset)):
                dic = dset.get_data_info(idx)
                dic = {
                    'img_name': get_name(dic['img_path']),
                    'seg_map_name': get_name(dic['seg_map_path']),
                    **dic
                }
                buffer.append(dic)
            info[desc] = buffer

        if self.assert_no_overlap:
            names = {
                setname: set(e['img_path'] for e in lst)
                for setname, lst in info.items()
            }

            for a, b in combinations(names.keys(), 2):
                intersection = names[a].intersection(names[b])
                if len(intersection) > 0:
                    print_log(f'overlapping samples between {a} and {b}: {intersection}', logger='current', level=WARNING)

        if self.log_split:
            logdir = Path(runner.log_dir)
            with logdir.joinpath('data.json').open('wt') as f:
                json.dump(info, f, indent=4)

