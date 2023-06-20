# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from collections import OrderedDict

import mmengine
import torch
from mmengine.runner import CheckpointLoader


def convert_cswin(ckpt):
    new_ckpt = OrderedDict()

    for k, v in ckpt.items():
        if 'stage1_conv_embed' in k:
            if '0.' in k:
                new_k = k.replace('0.', 'projection.')
            elif '2.' in k:
                new_k = k.replace('2.', 'norm.')
            else:
                raise ValueError('Unknown key {}'.format(k))

            new_v = v
        else:
            new_v = v
            new_k = k

        new_ckpt[new_k] = new_v

    return new_ckpt


def main():
    parser = argparse.ArgumentParser(
        description='Convert keys in official pretrained cswin models to'
        'MMSegmentation style.')
    parser.add_argument('src', help='src model path or url')
    # The dst path must be a full path of the new checkpoint.
    parser.add_argument('dst', help='save path')
    args = parser.parse_args()

    checkpoint = CheckpointLoader.load_checkpoint(args.src, map_location='cpu')
    checkpoint['state_dict'] = convert_cswin(checkpoint['state_dict'])

    mmengine.mkdir_or_exist(osp.dirname(args.dst))
    torch.save(checkpoint, args.dst)


if __name__ == '__main__':
    main()
