# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from ..utils import resize
from .decode_head_3d import BaseDecodeHeadPseudo3D
from .psp_head import PPM
from .uper_head import UPerHead


@MODELS.register_module()
class UPerHeadPseudo3D(BaseDecodeHeadPseudo3D):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, volume_depth: int = 4, pool_scales=(1, 2, 3, 6), **kwargs):
        super().__init__(volume_depth=volume_depth, input_transform='multiple_select', **kwargs)
        # PPM head, takes all channels into account
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        # The head per frame
        kwargs['in_channels'] = [(i // self.volume_depth) for i in self.in_channels]
        kwargs['in_channels'][-1] += self.channels_per_frame

        kwargs['channels'] = self.channels // self.volume_depth

        self.frame_head = UPerHead(
            pool_scales,
            **kwargs
        )

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        inputs = self._transform_inputs(inputs)

        # the PPM with all frames involved
        ppm_out = self.psp_forward(inputs)
        ppm_out = self._frames2samples(ppm_out)

        # rearrange frames
        inputs = [
            self._frames2samples(i) for i in inputs
        ]

        # concat coarser input stage with master ppm
        inputs[-1] = torch.cat((inputs[-1], ppm_out), dim=1)

        # segment each frame
        out = self.frame_head._forward_feature(inputs)
        return out

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output
