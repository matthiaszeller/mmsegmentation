import json
from copy import deepcopy
from logging import INFO
from pathlib import Path
from typing import Optional

import torch
from mmengine import print_log
from mmengine.evaluator import Evaluator
from mmengine.hooks import Hook
from mmengine.hooks.hook import DATA_BATCH
from mmengine.runner import Runner

from mmseg.registry import HOOKS


@HOOKS.register_module()
class TrainEvalHook(Hook):

    def __init__(self, interval: int = 50):
        self.interval = interval
        self.evaluator: Evaluator = None

    def before_train(self, runner: Runner) -> None:
        self.evaluator = deepcopy(runner.val_evaluator)

    def after_train_iter(self,
                         runner: Runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs: Optional[dict] = None) -> None:
        if self.every_n_train_iters(runner, self.interval):
            # TODO a bit dirty
            print_log('evaluating training batch', logger='current', level=INFO)
            runner.model.eval()
            for inp, samples in zip(data_batch['inputs'], data_batch['data_samples']):
                batch = dict(inputs=[inp], data_samples=[samples])
                self.run_iter(runner, batch)

            metrics = self.evaluator.evaluate(len(data_batch['inputs']))
            metrics['iter'] = runner.iter
            self.write_metrics(runner, metrics)

            runner.model.train()

    @torch.no_grad()
    def run_iter(self, runner: Runner, data_batch: DATA_BATCH):
        outputs = runner.model.val_step(data_batch)
        self.evaluator.process(data_samples=outputs, data_batch=data_batch)

    def write_metrics(self, runner: Runner, metrics: dict):
        logdir = Path(runner.log_dir).joinpath('train_metrics.json')
        with logdir.open('a') as fh:
            buffer = json.dumps(metrics)
            fh.write(buffer + '\n')
