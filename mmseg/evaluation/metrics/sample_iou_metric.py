# Modified by Matthias Gilles Zeller
import os.path as osp
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence

import numpy as np
from mmengine.dist import is_main_process
from mmengine.logging import MMLogger, print_log
from mmengine.utils import mkdir_or_exist
from prettytable import PrettyTable

from mmseg.registry import METRICS
from mmengine.evaluator import BaseMetric
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score


@METRICS.register_module()
class SampleIoUMetric(BaseMetric):

    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.labels = None

    def _get_sample_metrics(self, pred_label, label):
        res = dict()
        for name, fun in [('IoU', jaccard_score), ('Precision', precision_score),
                          ('Recall', recall_score), ('F1', f1_score)]:
            buffer = fun(label, pred_label, average=None, zero_division=0, labels=self.labels)
            for value, class_name in zip(buffer, self.dataset_meta['classes']):
                res[f'{name}.{class_name}'] = value

        return res

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        self.labels = np.arange(len(self.dataset_meta['classes']))
        for ds in data_samples:
            pred = ds['pred_sem_seg']['data'].squeeze().cpu().view(-1)
            label = ds['gt_sem_seg']['data'].squeeze().cpu().to(pred).view(-1)
            self.results.append(self._get_sample_metrics(pred, label))

    def compute_metrics(self, results: list) -> dict:
        return {
            'samples': results
        }

