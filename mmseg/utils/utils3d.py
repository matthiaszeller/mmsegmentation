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
    def get_img_path(img_paths, i):
        nums = [
            p.split('.')[0].split('_')[-1]
            for p in img_paths
        ]
        basename = img_paths[0].split('.')[0].rsplit('_', 1)[0]
        ext = img_paths[0].split('.')[-1]
        name = basename + '-' + '_'.join(nums) + '_' + nums[i] + '.' + ext
        return name

    keys = data_samples[0].keys()

    out = []
    for data_sample in data_samples:
        n = len(data_sample.metainfo['img_path'])
        for i in range(n):
            meta = deepcopy(data_sample.metainfo)
            meta['img_path'] = get_img_path(meta['img_path'], i)
            meta['seg_map_path'] = meta['seg_map_path'][i]
            sample = SegDataSample(metainfo=meta)
            for k in keys:
                in_pixeldata = getattr(data_sample, k)
                pixeldata = PixelData(metainfo=in_pixeldata.metainfo)
                pixeldata.data = in_pixeldata.data[i:i+1]
                setattr(sample, k, pixeldata)

            out.append(sample)

    return out
