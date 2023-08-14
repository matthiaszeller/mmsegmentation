


"""
raw = ' '.join([
    'PORT=29511',
    'sbatch',
    #'--begin=23:50:00',
    '-J',
    'hivit-fold1',
    './tools/dist_train_supercloud_1gpu.sh',
    'configs/hivit/hitvit-tiny-ade20k-pre_uperhead-channels-128_1x48-4k-amp_ivoct-polar-gray-512x512.py',
    '--amp',
    '--cfg-options',
    'model.init_cfg.checkpoint=work_dirs/hivit-tiny-itpn-in1k-pre_upernet_4xb16-80k_ade20k-512x512/iter_80000.pth',
    'train_dataloader.dataset.ann_file=splits/segmentation/train_fold2.txt',
    'val_dataloader.dataset.ann_file=splits/segmentation/val_fold2.txt',
    'test_dataloader.dataset.ann_file=splits/segmentation/val_fold2.txt',
    '--work-dir=work_dirs/hitvit-tiny-ade20k-pre_uperhead-channels-128_1x48-4k-amp_ivoct-polar-gray-512x512/fold2'
])
"""
import argparse
from copy import deepcopy
from functools import partial
from pathlib import Path


def gen_cmd_sbatch(script: str, config, jobname: str = None, split_train: str = None, split_val: str = None,
                   amp: bool = True, model_checkpoint: str = None, work_dir: str = None, port: int = None,
                   sbatch_args: str = None, cfg_options: list[str] = None):
    cmd = []

    if port is not None:
        cmd.append(f'PORT={port}')

    cmd.append('sbatch')
    if jobname is not None:
        cmd.extend(['-J', jobname])

    if sbatch_args is not None:
        cmd.append(sbatch_args)

    cmd.append(script)
    cmd.append(config)

    if amp:
        cmd.append('--amp')

    if cfg_options is None:
        cfg_options = []

    if model_checkpoint is not None:
        cfg_options.append(f'model.init_cfg.checkpoint={model_checkpoint}')

    if split_train is not None:
        cfg_options.append(f'train_dataloader.dataset.ann_file={split_train}')
    if split_val is not None:
        cfg_options.append(f'val_dataloader.dataset.ann_file={split_val}')
        cfg_options.append(f'test_dataloader.dataset.ann_file={split_val}')

    if work_dir is not None:
        cfg_options.append(f'work_dir={work_dir}')

    if cfg_options:
        cmd.append('--cfg-options')
        cmd.extend(cfg_options)


    return ' '.join(cmd)


def gen_cmd_sbatch_kfold(script: str, config: str, jobname: str = None,
                         split_format: str = 'splits/segmentation/{set}_fold{fold}.txt',
                         port: int = 29510, nfolds: int = 5, **kwargs):
    if split_format is None:
        split_format = 'splits/segmentation/{set}_fold{fold}.txt'

    jobname = 'fold{fold}' if jobname is None else jobname + '-fold{fold}'

    args = {
        'work_dir': str(Path('work_dirs') / Path(config).stem / 'fold{fold}'),
        'jobname': jobname,
        'split_train': partial(split_format.format, set='train'),
        'split_val': partial(split_format.format, set='val'),
    }

    cmd = []
    for i in range(1, nfolds + 1):
        args_fold = {
            k: v.format(fold=i) if isinstance(v, str) else v(fold=i)
            for k, v in args.items()
        }
        port_fold = port + i - 1
        cmd.append(
            gen_cmd_sbatch(script, config, port=port_fold, **args_fold, **deepcopy(kwargs))
        )

    return cmd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('script', type=str)
    parser.add_argument('config', type=str)
    parser.add_argument('--jobname', type=str, default=None)
    parser.add_argument('--split-format', type=str, default=None)
    parser.add_argument('--port', type=int, default=29510)
    parser.add_argument('--nfolds', type=int, default=5)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--model-checkpoint', type=str, default=None)
    parser.add_argument('--cfg-options', type=str, nargs='*')
    parser.add_argument('--sbatch-args', type=str, default=None, nargs='*')

    parser.add_argument('--run', help='Run the commands', action='store_true')

    args = parser.parse_args()
    run = args.run
    del args.run

    cmd = gen_cmd_sbatch_kfold(**vars(args))

    if run:
        import subprocess
        for c in cmd:
            print(c + '\n')
            subprocess.run(c, shell=True)
    else:
        print('\n\n'.join(cmd))
