import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from ..utils import resize
from .uper_head import UPerHead
from .psp_head import PPM


@MODELS.register_module()
class UPerHeadPseudo3D(UPerHead):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, volume_depth: int = 4, pool_scales=(1, 2, 3, 6), **kwargs):
        assert kwargs['channels'] % volume_depth == 0, f'channels must be divisible by num_images ' \
                                                       f'in {self.__class__.__name__}'
        super().__init__(**kwargs)
        self.volume_depth = volume_depth
        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1] // self.volume_depth,  # applied separately to each image within volume
            self.channels // self.volume_depth,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        self.bottleneck = ConvModule(
            self.in_channels[-1] // self.volume_depth + len(pool_scales) * (self.channels // self.volume_depth),
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = ConvModule(
                in_channels // self.volume_depth,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = ConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def _transform_inputs(self, inputs):
        inputs = super()._transform_inputs(inputs)
        # pull volume depth dimension to batch size dimension
        inputs = [
            x.reshape(x.shape[0] * self.volume_depth, x.shape[1] // self.volume_depth, *x.shape[2:])
            for x in inputs
        ]
        return inputs
