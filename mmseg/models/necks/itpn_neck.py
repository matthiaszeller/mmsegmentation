"""
Mix of code from mmpretrain and original iptn codebase
"""
import math
import warnings
from functools import partial
from typing import List, Optional, Union, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import FFN
from mmengine import digit_version
from mmengine.model import BaseModule
from mmengine.runner import CheckpointLoader

from mmseg.registry import MODELS
from ..backbones.hivit import BlockWithRPE

# After pytorch v1.10.0, use torch.meshgrid without indexing
# will raise extra warning. For more details,
# refers to https://github.com/pytorch/pytorch/issues/50276
if digit_version(torch.__version__) >= digit_version('1.10.0'):
    torch_meshgrid = partial(torch.meshgrid, indexing='ij')
else:
    torch_meshgrid = torch.meshgrid


def build_2d_sincos_position_embedding(
        patches_resolution: Union[int, Sequence[int]],
        embed_dims: int,
        temperature: Optional[int] = 10000.,
        cls_token: Optional[bool] = False) -> torch.Tensor:
    """The function is to build position embedding for model to obtain the
    position information of the image patches.

    Args:
        patches_resolution (Union[int, Sequence[int]]): The resolution of each
            patch.
        embed_dims (int): The dimension of the embedding vector.
        temperature (int, optional): The temperature parameter. Defaults to
            10000.
        cls_token (bool, optional): Whether to concatenate class token.
            Defaults to False.

    Returns:
        torch.Tensor: The position embedding vector.
    """

    if isinstance(patches_resolution, int):
        patches_resolution = (patches_resolution, patches_resolution)

    h, w = patches_resolution
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w, grid_h = torch_meshgrid(grid_w, grid_h)
    assert embed_dims % 4 == 0, \
        'Embed dimension must be divisible by 4.'
    pos_dim = embed_dims // 4

    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1. / (temperature**omega)
    out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
    out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])

    pos_emb = torch.cat(
        [
            torch.sin(out_w),
            torch.cos(out_w),
            torch.sin(out_h),
            torch.cos(out_h)
        ],
        dim=1,
    )[None, :, :]

    if cls_token:
        cls_token_pe = torch.zeros([1, 1, embed_dims], dtype=torch.float32)
        pos_emb = torch.cat([cls_token_pe, pos_emb], dim=1)

    return pos_emb


def scaled_dot_product_attention_pyimpl(query,
                                        key,
                                        value,
                                        attn_mask=None,
                                        dropout_p=0.,
                                        scale=None,
                                        is_causal=False):
    scale = scale or query.size(-1)**0.5
    if is_causal and attn_mask is not None:
        attn_mask = torch.ones(
            query.size(-2), key.size(-2), dtype=torch.bool).tril(diagonal=0)
    if attn_mask is not None and attn_mask.dtype == torch.bool:
        attn_mask = attn_mask.masked_fill(not attn_mask, -float('inf'))

    attn_weight = query @ key.transpose(-2, -1) / scale
    if attn_mask is not None:
        attn_weight += attn_mask
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, True)
    return attn_weight @ value


if digit_version(torch.__version__) >= digit_version('2.0.0'):
    scaled_dot_product_attention = F.scaled_dot_product_attention
else:
    scaled_dot_product_attention = scaled_dot_product_attention_pyimpl


class LayerScale(nn.Module):
    """LayerScale layer.

    Args:
        dim (int): Dimension of input features.
        layer_scale_init_value (float or torch.Tensor): Init value of layer
            scale. Defaults to 1e-5.
        inplace (bool): inplace: can optionally do the
            operation in-place. Defaults to False.
        data_format (str): The input data format, could be 'channels_last'
             or 'channels_first', representing (B, C, H, W) and
             (B, N, C) format data respectively. Defaults to 'channels_last'.
    """

    def __init__(self,
                 dim: int,
                 layer_scale_init_value: Union[float, torch.Tensor] = 1e-5,
                 inplace: bool = False,
                 data_format: str = 'channels_last'):
        super().__init__()
        assert data_format in ('channels_last', 'channels_first'), \
            "'data_format' could only be channels_last or channels_first."
        self.inplace = inplace
        self.data_format = data_format
        self.weight = nn.Parameter(torch.ones(dim) * layer_scale_init_value)

    def forward(self, x):
        if self.data_format == 'channels_first':
            if self.inplace:
                return x.mul_(self.weight.view(-1, 1, 1))
            else:
                return x * self.weight.view(-1, 1, 1)
        return x.mul_(self.weight) if self.inplace else x * self.weight


class MultiheadAttention(BaseModule):
    """Multi-head Attention Module.

    This module implements multi-head attention that supports different input
    dims and embed dims. And it also supports a shortcut from ``value``, which
    is useful if input dims is not the same with embed dims.

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        input_dims (int, optional): The input dimension, and if None,
            use ``embed_dims``. Defaults to None.
        attn_drop (float): Dropout rate of the dropout layer after the
            attention calculation of query and key. Defaults to 0.
        proj_drop (float): Dropout rate of the dropout layer after the
            output projection. Defaults to 0.
        dropout_layer (dict): The dropout config before adding the shortcut.
            Defaults to ``dict(type='Dropout', drop_prob=0.)``.
        qkv_bias (bool): If True, add a learnable bias to q, k, v.
            Defaults to True.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        proj_bias (bool) If True, add a learnable bias to output projection.
            Defaults to True.
        v_shortcut (bool): Add a shortcut from value to output. It's usually
            used if ``input_dims`` is different from ``embed_dims``.
            Defaults to False.
        use_layer_scale (bool): Whether to use layer scale. Defaults to False.
        layer_scale_init_value (float or torch.Tensor): Init value of layer
            scale. Defaults to 0.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 input_dims=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 qkv_bias=True,
                 qk_scale=None,
                 proj_bias=True,
                 v_shortcut=False,
                 use_layer_scale=False,
                 layer_scale_init_value=0.,
                 init_cfg=None):
        super(MultiheadAttention, self).__init__(init_cfg=init_cfg)

        self.input_dims = input_dims or embed_dims
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.v_shortcut = v_shortcut

        self.head_dims = embed_dims // num_heads
        if qk_scale is not None:
            self.scaled_dot_product_attention = partial(
                scaled_dot_product_attention_pyimpl,
                scale=self.head_dims**-0.5)
        else:
            self.scaled_dot_product_attention = scaled_dot_product_attention

        self.qkv = nn.Linear(self.input_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(embed_dims, embed_dims, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.out_drop = build_dropout(dropout_layer)

        if use_layer_scale:
            warnings.warn('The `use_layer_scale` in `MultiheadAttention` will '
                          'be deprecated. Please use `layer_scale_init_value` '
                          'to control whether using layer scale or not.')

        if use_layer_scale or (layer_scale_init_value > 0):
            layer_scale_init_value = layer_scale_init_value or 1e-5
            self.gamma1 = LayerScale(
                embed_dims, layer_scale_init_value=layer_scale_init_value)
        else:
            self.gamma1 = nn.Identity()

    def forward(self, x):
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  self.head_dims).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_drop = self.attn_drop if self.training else 0.
        x = self.scaled_dot_product_attention(q, k, v, dropout_p=attn_drop)
        x = x.transpose(1, 2).reshape(B, N, self.embed_dims)

        x = self.proj(x)
        x = self.out_drop(self.gamma1(self.proj_drop(x)))

        if self.v_shortcut:
            x = v.squeeze(1) + x
        return x


class TransformerEncoderLayer(BaseModule):
    """Implements one encoder layer in Vision Transformer.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Default: 0.0.
        attn_drop_rate (float): The drop out rate for attention layer.
            Default: 0.0.
        drop_path_rate (float): stochastic depth rate. Default 0.0.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Default: 2.
        qkv_bias (bool): enable bias for qkv if True. Default: True
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default: True.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 num_fcs=2,
                 qkv_bias=True,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 batch_first=True,
                 attn_cfg=dict(),
                 ffn_cfg=dict(),
                 with_cp=False):
        super().__init__()

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, embed_dims, postfix=1)
        self.add_module(self.norm1_name, norm1)

        attn_cfg.update(
            dict(
                embed_dims=embed_dims,
                num_heads=num_heads,
                attn_drop=attn_drop_rate,
                proj_drop=drop_rate,
                batch_first=batch_first,
                bias=qkv_bias))

        self.build_attn(attn_cfg)

        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, embed_dims, postfix=2)
        self.add_module(self.norm2_name, norm2)

        ffn_cfg.update(
            dict(
                embed_dims=embed_dims,
                feedforward_channels=feedforward_channels,
                num_fcs=num_fcs,
                ffn_drop=drop_rate,
                dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate)
                if drop_path_rate > 0 else None,
                act_cfg=act_cfg))
        self.build_ffn(ffn_cfg)
        self.with_cp = with_cp

    def build_attn(self, attn_cfg):
        self.attn = MultiheadAttention(**attn_cfg)

    def build_ffn(self, ffn_cfg):
        self.ffn = FFN(**ffn_cfg)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x):

        def _inner_forward(x):
            x = self.attn(self.norm1(x), identity=x)
            x = self.ffn(self.norm2(x), identity=x)
            return x

        if self.with_cp and x.requires_grad:
            raise NotImplementedError
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


class PatchSplit(nn.Module):
    """The up-sample module used in neck (transformer pyramid network)

    Args:
        dim (int): the input dimension (channel number).
        fpn_dim (int): the fpn dimension (channel number).
        norm_cfg (dict): Config dict for normalization layer.
                Defaults to ``dict(type='LN')``.
    """

    def __init__(self, dim, fpn_dim, norm_cfg):
        super().__init__()
        _, self.norm = build_norm_layer(norm_cfg, dim)
        self.reduction = nn.Linear(dim, fpn_dim * 4, bias=False)
        self.fpn_dim = fpn_dim

    def forward(self, x):
        B, N, H, W, C = x.shape
        x = self.norm(x)
        x = self.reduction(x)
        x = x.reshape(B, N, H, W, 2, 2,
                      self.fpn_dim).permute(0, 1, 2, 4, 3, 5,
                                            6).reshape(B, N, 2 * H, 2 * W,
                                                       self.fpn_dim)
        return x


@MODELS.register_module()
class iTPNNeck(BaseModule):
    """The neck module of iTPN (transformer pyramid network).

    Args:
        num_patches (int): The number of total patches. Defaults to 196.
        patch_size (int): Image patch size. Defaults to 16.
        in_chans (int): The channel of input image. Defaults to 3.
        embed_dim (int): Encoder's embedding dimension. Defaults to 512.
        fpn_dim (int): The fpn dimension (channel number).
        fpn_depth (int): The layer number of feature pyramid.
        decoder_embed_dim (int): Decoder's embedding dimension.
            Defaults to 512.
        decoder_depth (int): The depth of decoder. Defaults to 8.
        decoder_num_heads (int): Number of attention heads of decoder.
            Defaults to 16.
        mlp_ratio (int): Ratio of mlp hidden dim to decoder's embedding dim.
            Defaults to 4.
        norm_cfg (dict): Normalization layer. Defaults to LayerNorm.
        reconstruction_type (str): The itpn supports 2 kinds of supervisions.
            Defaults to 'pixel'.
        num_outs (int): The output number of neck (transformer pyramid
            network). Defaults to 3.
        predict_feature_dim (int): The output dimension to supervision.
            Defaults to None.
        init_cfg (Union[List[dict], dict], optional): Initialization config
            dict. Defaults to None.
    """

    def __init__(self,
                 #num_patches: int = 196,
                 #patch_size: int = 16,
                 # in_chans: int = 3,
                 embed_dim: int = 512,
                 fpn_dim: int = 256,
                 fpn_depth: int = 2,
                 # decoder_embed_dim: int = 512,
                 # decoder_depth: int = 6,
                 # decoder_num_heads: int = 16,
                 mlp_ratio: int = 4,
                 norm_cfg: dict = dict(type='LN', eps=1e-6),
                 # reconstruction_type: str = 'pixel',
                 num_outs: int = 3,
                 qkv_bias: bool = True,
                 qk_scale: Optional[bool] = None,
                 drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0,
                 # predict_feature_dim: Optional[float] = None,
                 init_cfg: Optional[Union[List[dict], dict]] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        # assert reconstruction_type in ['pixel', 'clip'], \
        #     'iTPN method only support `pixel` and `clip`, ' \
        #     f'but got `{reconstruction_type}`.'
        # self.reconstruction_type = reconstruction_type
        self.num_outs = num_outs

        self.build_transformer_pyramid(
            num_outs=num_outs,
            embed_dim=embed_dim,
            fpn_dim=fpn_dim,
            fpn_depth=fpn_depth,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            rpe=False,
            norm_cfg=norm_cfg,
        )

    def build_transformer_pyramid(self,
                                  num_outs=3,
                                  embed_dim=512,
                                  fpn_dim=256,
                                  fpn_depth=2,
                                  mlp_ratio=4.0,
                                  qkv_bias=True,
                                  qk_scale=None,
                                  drop_rate=0.0,
                                  attn_drop_rate=0.0,
                                  rpe=False,
                                  norm_cfg=None):
        Hp = None
        mlvl_dims = {'4': embed_dim // 4, '8': embed_dim // 2, '16': embed_dim}
        if num_outs > 1:
            if embed_dim != fpn_dim:
                self.align_dim_16tofpn = nn.Linear(embed_dim, fpn_dim)
            else:
                self.align_dim_16tofpn = None
            self.fpn_modules = nn.ModuleList()
            self.fpn_modules.append(
                BlockWithRPE(
                    Hp,
                    fpn_dim,
                    0,
                    mlp_ratio,
                    qkv_bias,
                    qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=0.,
                    rpe=rpe,
                    norm_cfg=norm_cfg))
            self.fpn_modules.append(
                BlockWithRPE(
                    Hp,
                    fpn_dim,
                    0,
                    mlp_ratio,
                    qkv_bias,
                    qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=0.,
                    rpe=False,
                    norm_cfg=norm_cfg,
                ))

            self.align_dim_16to8 = nn.Linear(
                mlvl_dims['8'], fpn_dim, bias=False)
            self.split_16to8 = PatchSplit(mlvl_dims['16'], fpn_dim, norm_cfg)
            self.block_16to8 = nn.Sequential(*[
                BlockWithRPE(
                    Hp,
                    fpn_dim,
                    0,
                    mlp_ratio,
                    qkv_bias,
                    qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=0.,
                    rpe=rpe,
                    norm_cfg=norm_cfg,
                ) for _ in range(fpn_depth)
            ])

        if num_outs > 2:
            self.align_dim_8to4 = nn.Linear(
                mlvl_dims['4'], fpn_dim, bias=False)
            self.split_8to4 = PatchSplit(fpn_dim, fpn_dim, norm_cfg)
            self.block_8to4 = nn.Sequential(*[
                BlockWithRPE(
                    Hp,
                    fpn_dim,
                    0,
                    mlp_ratio,
                    qkv_bias,
                    qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=0.,
                    rpe=rpe,
                    norm_cfg=norm_cfg,
                ) for _ in range(fpn_depth)
            ])
            self.fpn_modules.append(
                BlockWithRPE(
                    Hp,
                    fpn_dim,
                    0,
                    mlp_ratio,
                    qkv_bias,
                    qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=0.,
                    rpe=rpe,
                    norm_cfg=norm_cfg))

    def init_weights(self) -> None:
        """Initialize position embedding and mask token of MAE decoder."""
        if self.init_cfg is None or 'checkpoint' not in self.init_cfg:
            super().init_weights()
            self.rescale_init_weight()
            return

        ckpt = CheckpointLoader.load_checkpoint(self.init_cfg['checkpoint'], map_location='cpu', logger='current')
        state_dict = {
            k[5:]: v
            for k, v in ckpt['state_dict'].items()
            if k.startswith('neck.')
        }
        self.load_state_dict(state_dict, strict=False)

    def rescale_init_weight(self) -> None:
        """Rescale the initialized weights."""

        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.fpn_modules):
            if isinstance(layer, BlockWithRPE):
                if layer.attn is not None:
                    rescale(layer.attn.proj.weight.data, layer_id + 1)
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        r"""Combine non-overlapped patches into images.

        Args:
            x (torch.Tensor): The shape is
                :math:`(B, L, \text{patch_size}^2 \times 3)`.

        Returns:
            torch.Tensor: The shape is :math:`(B, 3, H, W)`.
        """
        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.in_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.in_chans, h * p, h * p))
        return imgs

    @property
    def decoder_norm(self):
        """The normalization layer of decoder."""
        return getattr(self, self.decoder_norm_name)

    def forward_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        """The forward function.

        The process computes the visible patches' features vectors and the mask
        tokens to output feature vectors, which will be used for
        reconstruction.

        Args:
            x (torch.Tensor): hidden features, which is of shape
                    B x C x H x W.

        Returns:
            torch.Tensor:
        """
        features = x[:2]
        x = x[-1]
        B, L, _ = x.shape
        x = x[..., None, None, :]
        features += (x, )
        Hp = Wp = int(math.sqrt(L))

        outs = [x] if self.align_dim_16tofpn is None else [
            self.align_dim_16tofpn(x)
        ]
        if self.num_outs >= 2:
            x = self.block_16to8(
                self.split_16to8(x) + self.align_dim_16to8(features[1]))
            outs.append(x)
        if self.num_outs >= 3:
            x = self.block_8to4(
                self.split_8to4(x) + self.align_dim_8to4(features[0]))
            outs.append(x)

        # Code block: from mmpretrain iTPN
        # if self.num_outs > 3:
        #     outs = [
        #         out.reshape(B, Hp, Wp, *out.shape[-3:]).permute(
        #             0, 5, 1, 3, 2, 4).reshape(B, -1, Hp * out.shape[-3],
        #                                       Wp * out.shape[-2]).contiguous()
        #         for out in outs
        #     ]
        #     if self.num_outs >= 4:
        #         outs.insert(0, F.avg_pool2d(outs[0], kernel_size=2, stride=2))
        #     if self.num_outs >= 5:
        #         outs.insert(0, F.avg_pool2d(outs[0], kernel_size=2, stride=2))
        #
        # for i, out in enumerate(outs):
        #     out = self.fpn_modules[i](out)
        #     outs[i] = out

        # Code block: from original iTPN
        len_feat = len(features)
        for i, out in enumerate(outs):
            out = torch.cat([self.fpn_modules[i](out), features[len_feat - i - 1]], dim=-1)
            outs[i] = out

        # format tensors to image view
        outs = [
            out.reshape(B, Hp, Wp, *out.shape[-3:]).permute(0, 5, 1, 3, 2, 4).reshape(
                B, -1, Hp * out.shape[-3], Wp * out.shape[-2]).contiguous()
            for out in outs
        ]

        if self.num_outs >= 4:
            outs.insert(0, F.max_pool2d(outs[0], kernel_size=2, stride=2))
        if self.num_outs >= 5:
            outs.insert(0, F.max_pool2d(outs[0], kernel_size=2, stride=2))

        return tuple(outs)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        features = self.forward_features(x)
        features = tuple(reversed(features))
        return features

