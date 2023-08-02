import torch
from mmengine.structures import PixelData

from .typing_utils import SampleList
from ..structures import SegDataSample
from copy import deepcopy


def unpack_data_samples(data_samples: SampleList) -> SampleList:
    """Unpack data samples.

    Args:
        data_samples (list[:obj:`SegDataSample`]): The seg data samples.
            It usually includes information such as `metainfo` and
            `gt_sem_seg`.

    Returns:
        list[:obj:`SegDataSample`]: The unpacked seg data samples.
    """
    keys = data_samples[0].keys()

    out = []
    for data_sample in data_samples:
        n = len(data_sample.metainfo['img_path'])
        for i in range(n):
            meta = deepcopy(data_sample.metainfo)
            meta['img_path'] = meta['img_path'][i]
            meta['seg_map_path'] = meta['seg_map_path'][i]
            sample = SegDataSample(metainfo=meta)
            for k in keys:
                in_pixeldata = getattr(data_sample, k)
                pixeldata = PixelData(metainfo=in_pixeldata.metainfo)
                pixeldata.data = in_pixeldata.data[i:i+1]
                setattr(sample, k, pixeldata)

            out.append(sample)

    return out
