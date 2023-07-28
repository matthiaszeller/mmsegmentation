# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from pathlib import Path

import mmengine
import mmengine.fileio as fileio
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class IVOCTDataset(BaseSegDataset):
    """Pascal VOC dataset.

    Args:
        ann_file (str): split txt file
    """

    METAINFO = dict(
        classes=('background', 'calcium'),
        palette=[[0, 0, 0], [255, 0, 0]]
    )

    def __init__(self,
                 ann_file,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            ann_file=ann_file,
            **kwargs)
        assert fileio.exists(self.data_prefix['img_path'],
                             self.backend_args) and osp.isfile(self.ann_file)


@DATASETS.register_module()
class IVOCTZipDataset(IVOCTDataset):
    """
    IVOCT dataset reading images from zip file, associated loader: ReadImageFromZipFile.
    Annotation file can contain additional info, in the form of key=value pairs, e.g.:

    0001_1.png group=1 otherkey=othervalue
    0001_2.png group=1 otherkey=othervalue
    0002_1.png group=2 otherkey=othervalue
    ...
    """

    def __init__(self,
                 data_root: str,
                 ann_file,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 **kwargs) -> None:

        super().__init__(
            ann_file=ann_file,
            data_root=data_root,
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            **kwargs)

    def load_data_list(self) -> list[dict]:
        data_list = []
        lines = mmengine.list_from_file(self.ann_file, backend_args=self.backend_args)

        img_dir = Path(self.data_prefix['img_path'])
        ann_dir = Path(self.data_prefix['seg_map_path'])
        for line in lines:
            line = line.strip()
            img_name, *dicstr = line.split(' ')
            dic = dict([kv.split('=') for kv in dicstr])

            basename, slicenum = img_name.split('_')
            slicenum = int(slicenum)

            zip_path = img_dir.joinpath(basename).with_suffix('.zip')
            img_path = Path(img_name).with_suffix(self.img_suffix)
            seg_map = ann_dir.joinpath(img_name).with_suffix(self.seg_map_suffix)

            data_info = dict(
                img_path=str(img_path),
                zip_path=str(zip_path),
                seg_map_path=str(seg_map),
                label_map=self.label_map,
                reduce_zero_label=self.reduce_zero_label,
                seg_fields=[],
                basename=basename,
                slicenum=slicenum,
                **dic
            )
            data_list.append(data_info)

        return data_list
