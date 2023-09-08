# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from .decode_head_3d import BaseDecodeHeadPseudo3D
from .decode_head import BaseDecodeHead


@MODELS.register_module()
class FCNHeadPseudo3D(BaseDecodeHeadPseudo3D):
    """Fully Convolution Networks for Semantic Segmentation.

    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.

    Args:
        n_slice (int): Number of slices in the 3D volume.
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
        dilation (int): The dilation rate for convs in the head. Default: 1.
    """

    def __init__(self,
                 n_slice=4,
                 num_convs=2,
                 kernel_size=3,
                 concat_input=True,
                 dilation=1,
                 **kwargs):
        assert num_convs >= 0 and dilation > 0 and isinstance(dilation, int)

        self.volume_depth = n_slice
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        super().__init__(**kwargs)

        assert n_slice > 0 and isinstance(n_slice, int)
        assert self.channels % n_slice == 0, 'channels must be divisible by volume_depth'

        if num_convs == 0:
            assert self.in_channels == self.channels

        conv_padding = (kernel_size // 2) * dilation

        self.conv_full = ConvModule(
            self.in_channels,
            self.channels,
            kernel_size=kernel_size,
            padding=conv_padding,
            dilation=dilation,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )

        channel_per_frame = self.channels // n_slice
        convs = [
            ConvModule(
                channel_per_frame,
                channel_per_frame,
                kernel_size=kernel_size,
                padding=conv_padding,
                dilation=dilation,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            )
            for _ in range(num_convs)
        ]

        if num_convs == 0:
            self.convs_indep = nn.Identity()
        else:
            self.convs_indep = nn.Sequential(*convs)

        if self.concat_input:
            raise NotImplementedError
            self.conv_cat = ConvModule(
                self.in_channels + self.channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        # x is (N, C, H, W)
        x = self._transform_inputs(inputs)

        # feats is (N, C, H, W)
        feats = self.conv_full(x)
        # feats_frame is (N * volume_depth, C // volume_depth, H, W)
        feats_frame = self._frames2samples(feats)
        feats_frame = self.convs_indep(feats_frame)

        if self.concat_input:
            raise NotImplementedError
            feats = self.conv_cat(torch.cat([x, feats], dim=1))

        return feats_frame

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output


@MODELS.register_module()
class FCNHeadPseudo3DBASE(BaseDecodeHead):
    """Fully Convolution Networks for Semantic Segmentation.

    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.

    Args:
        volume_depth (int): Number of slices in the 3D volume.
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
        dilation (int): The dilation rate for convs in the head. Default: 1.
    """

    def __init__(self,
                 volume_depth=4,
                 num_convs=2,
                 kernel_size=3,
                 concat_input=True,
                 dilation=1,
                 **kwargs):
        assert num_convs >= 0 and dilation > 0 and isinstance(dilation, int)

        self.volume_depth = volume_depth
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        super().__init__(**kwargs)

        assert volume_depth > 0 and isinstance(volume_depth, int)
        assert self.channels % volume_depth == 0, 'channels must be divisible by volume_depth'

        if num_convs == 0:
            assert self.in_channels == self.channels

        conv_padding = (kernel_size // 2) * dilation

        self.conv_full = ConvModule(
            self.in_channels,
            self.channels,
            kernel_size=kernel_size,
            padding=conv_padding,
            dilation=dilation,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )

        channel_per_frame = self.channels // volume_depth
        convs = [
            ConvModule(
                channel_per_frame,
                channel_per_frame,
                kernel_size=kernel_size,
                padding=conv_padding,
                dilation=dilation,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            )
            for _ in range(num_convs)
        ]

        if num_convs == 0:
            self.convs_indep = nn.Identity()
        else:
            self.convs_indep = nn.Sequential(*convs)

        if self.concat_input:
            raise NotImplementedError
            self.conv_cat = ConvModule(
                self.in_channels + self.channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        # x is (N, C, H, W)
        x = self._transform_inputs(inputs)

        # feats is (N, C, H, W)
        feats = self.convs_full(x)
        # feats_frame is (N * volume_depth, C // volume_depth, H, W)
        feats_frame = feats.reshape(feats.shape[0] * self.volume_depth, self.channels // self.volume_depth, *feats.shape[2:])
        feats_frame = self.convs_indep(feats_frame)

        if self.concat_input:
            raise NotImplementedError
            feats = self.conv_cat(torch.cat([x, feats], dim=1))

        return feats_frame

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output
