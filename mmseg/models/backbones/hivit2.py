# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import logging
import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from mmengine import print_log
from mmengine.runner import CheckpointLoader
from scipy import interpolate
from timm.models.layers import to_2tuple
from timm.models.vision_transformer import DropPath, Mlp, trunc_normal_

from mmseg.registry import MODELS


class Attention(nn.Module):
    def __init__(self, input_size, dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., rpe=True):
        super().__init__()
        self.input_size = input_size
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * input_size - 1) * (2 * input_size - 1), num_heads)
        ) if rpe else None
        # if rpe:
        #     trunc_normal_(self.relative_position_bias_table, std=.02)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, rpe_index=None, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if rpe_index is not None:
            S = int(math.sqrt(rpe_index.size(-1)))
            relative_position_bias = self.relative_position_bias_table[rpe_index].view(-1, S, S, self.num_heads)
            relative_position_bias = relative_position_bias.permute(0, 3, 1, 2).contiguous()
            if N > S:
                relative_position_bias = F.pad(relative_position_bias, (N - S, 0, N - S, 0))
            attn = attn + relative_position_bias
        if mask is not None:
            mask = mask.bool()
            attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class BlockWithRPE(nn.Module):
    def __init__(self, input_size, dim, num_heads=0., mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., rpe=True,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        with_attn = num_heads > 0.

        self.norm1 = norm_layer(dim) if with_attn else None
        self.attn = Attention(
            input_size, dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, rpe=rpe,
        ) if with_attn else None

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, rpe_index=None, mask=None):
        if self.attn is not None:
            x = x + self.drop_path(self.attn(self.norm1(x), rpe_index, mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, inner_patches=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.inner_patches = inner_patches
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        conv_size = [size // inner_patches for size in patch_size]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=conv_size, stride=conv_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        patches_resolution = (H // self.patch_size[0], W // self.patch_size[1])
        num_patches = patches_resolution[0] * patches_resolution[1]
        x = self.proj(x).view(
            B, -1,
            patches_resolution[0], self.inner_patches,
            patches_resolution[1], self.inner_patches,
        ).permute(0, 2, 4, 3, 5, 1).reshape(B, num_patches, self.inner_patches, self.inner_patches, -1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerge(nn.Module):
    def __init__(self, dim, norm_layer):
        super().__init__()
        self.norm = norm_layer(dim * 4)
        self.reduction = nn.Linear(dim * 4, dim * 2, bias=False)

    def forward(self, x):
        x0 = x[..., 0::2, 0::2, :]
        x1 = x[..., 1::2, 0::2, :]
        x2 = x[..., 0::2, 1::2, :]
        x3 = x[..., 1::2, 1::2, :]

        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class BaseHiViT(nn.Module):
    def __init__(self, img_size=224, task_img_size=512, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=512, depths=(4, 4, 20), num_heads=8, stem_mlp_ratio=3., mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.0,
                 norm_layer=nn.LayerNorm, ape=True, rpe=True, patch_norm=True, use_checkpoint=False,
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.ape = ape
        self.rpe = rpe
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        self.num_main_blocks = depths[-1]

        embed_dim = embed_dim // 2 ** (self.num_layers - 1)
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=task_img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        Hp, Wp = self.patch_embed.patches_resolution
        assert Hp == Wp

        # absolute position embedding
        if ape:
            num_patches = (img_size // patch_size) ** 2
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, self.num_features)
            )
            trunc_normal_(self.absolute_pos_embed, std=.02)
        if rpe:
            coords_h = torch.arange(Hp)
            coords_w = torch.arange(Wp)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += Hp - 1
            relative_coords[:, :, 1] += Wp - 1
            relative_coords[:, :, 0] *= 2 * Wp - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer("relative_position_index", relative_position_index)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = iter(x.item() for x in
                   torch.linspace(0, drop_path_rate, sum(depths) + sum(depths[:-1])))  # stochastic depth decay rule

        # build blocks
        self.blocks = nn.ModuleList()
        for stage_depth in depths:
            is_main_stage = embed_dim == self.num_features
            nhead = num_heads if is_main_stage else 0
            ratio = mlp_ratio if is_main_stage else stem_mlp_ratio
            # every block not in main stage include two mlp blocks
            stage_depth = stage_depth if is_main_stage else stage_depth * 2
            for _ in range(stage_depth):
                self.blocks.append(
                    BlockWithRPE(
                        Hp, embed_dim, nhead, ratio, qkv_bias, qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=next(dpr),
                        rpe=rpe, norm_layer=norm_layer,
                    )
                )
            if not is_main_stage:
                self.blocks.append(
                    PatchMerge(embed_dim, norm_layer)
                )
                embed_dim *= 2

        self.fc_norm = norm_layer(self.num_features)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x, ids_keep=None, mask=None):
        B = x.shape[0]
        x = self.patch_embed(x)
        if ids_keep is not None:
            x = torch.gather(
                x, dim=1, index=ids_keep[:, :, None, None, None].expand(-1, -1, *x.shape[2:])
            )

        for blk in self.blocks[:-self.num_main_blocks]:
            x = checkpoint.checkpoint(blk, x) if self.use_checkpoint else blk(x)
        x = x[..., 0, 0, :]
        if self.ape:
            pos_embed = self.absolute_pos_embed
            if ids_keep is not None:
                pos_embed = torch.gather(
                    pos_embed.expand(B, -1, -1),
                    dim=1,
                    index=ids_keep[:, :, None].expand(-1, -1, pos_embed.shape[2]),
                )
            x += pos_embed
        x = self.pos_drop(x)

        rpe_index = None
        if self.rpe:
            if ids_keep is not None:
                B, L = ids_keep.shape
                rpe_index = self.relative_position_index
                rpe_index = torch.gather(
                    rpe_index[ids_keep, :], dim=-1, index=ids_keep[:, None, :].expand(-1, L, -1)
                ).reshape(B, -1)
            else:
                rpe_index = self.relative_position_index.view(-1)

        for blk in self.blocks[-self.num_main_blocks:]:
            x = checkpoint.checkpoint(blk, x, rpe_index, mask) if self.use_checkpoint else blk(x, rpe_index, mask)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = x.mean(dim=1)
        x = self.fc_norm(x)
        x = self.head(x)
        return x

    def get_num_layers(self) -> int:
        return len(self.blocks)


# INFO Model PARAMs 71.75M, FLOPs 15.92G with 224 input
def hivit_base(**kwargs):
    model = HiViT2(
        embed_dim=512, depths=[4, 4, 20], num_heads=8, stem_mlp_ratio=3., mlp_ratio=4.,
        rpe=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


@MODELS.register_module()
class HiViT2(BaseHiViT):
    def __init__(self,
                 img_size=224,
                 task_img_size=640,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=512,
                 depths=(4, 4, 20),
                 num_heads=8,
                 stem_mlp_ratio=3.,
                 mlp_ratio=4.,
                 drop_path_rate=0.1,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 ape=True, rpe=True,
                 patch_norm=True,
                 with_fpn=False,
                 out_indices=('H', 'M', 19, 19),
                 use_checkpoint=False,
                 init_cfg=None,
                 **kwargs):
        super(HiViT2, self).__init__(
            img_size=img_size,
            task_img_size=task_img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            stem_mlp_ratio=stem_mlp_ratio,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            ape=ape, rpe=rpe,
            patch_norm=patch_norm,
            **kwargs)

        assert not with_fpn or patch_size in (16,)
        self.init_cfg = init_cfg
        self.patch_size = patch_size
        self.with_fpn = with_fpn
        self.merge_indices = (depths[0] - 1, depths[0] + depths[1])
        self.out_indices = out_indices
        self.use_checkpoint = use_checkpoint

        del self.fc_norm, self.head, self.num_classes
        if with_fpn:
            self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                nn.BatchNorm2d(embed_dim),
                nn.GELU(),
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            ) if 'H' not in out_indices else nn.Identity()
            self.fpn2 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            ) if 'M' not in out_indices else nn.Identity()
            self.fpn3 = nn.Identity()
            self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            print_log('Build model without FPN.')

        self.init_weights()

    def init_weights(self):
        # apply weight init in case some keys are missing in state_dict
        self.apply(self._init_weights)

        if self.init_cfg is None:
            pass
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '

            logger = 'current'
            checkpoint = CheckpointLoader.load_checkpoint(
                self.init_cfg['checkpoint'], map_location='cpu', logger=logger
            )

            # get state_dict from checkpoint
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'module' in checkpoint:
                state_dict = checkpoint['module']
            else:
                state_dict = checkpoint
            # strip prefix of state_dict
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}

            # for MoBY, load model of online branch
            if any([k.startswith('encoder') for k in state_dict.keys()]):
                print_log('Remove the prefix of "encoder."', logger)
                state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}

            if "rel_pos_bias.relative_position_bias_table" in state_dict.keys():
                print_log("Expand the shared relative position embedding to each transformer block. ", logger)
                num_layers = self.get_num_layers()
                rel_pos_bias = state_dict["rel_pos_bias.relative_position_bias_table"]
                for i in range(num_layers):
                    state_dict["blocks.%d.attn.relative_position_bias_table" % i] = rel_pos_bias.clone()

            # reshape absolute position embedding for Swin
            if state_dict.get('absolute_pos_embed') is not None:
                absolute_pos_embed = state_dict['absolute_pos_embed']
                N1, L, C1 = absolute_pos_embed.size()
                N2, L2, C2 = self.absolute_pos_embed.size()
                if N1 != N2 or C1 != C2 or L != L2:
                    # TODO implement interpolation
                    print_log("Error in loading absolute_pos_embed, pass, TODO IMPLEMENT", logger, level=logging.ERROR)
                else:
                    state_dict['absolute_pos_embed'] = absolute_pos_embed

            all_keys = list(self.state_dict().keys())
            for key in all_keys:
                if "relative_position_bias_table" in key:
                    if key not in state_dict.keys():
                        continue

                    rel_pos_bias = state_dict[key]
                    src_num_pos, num_attn_heads = rel_pos_bias.size()
                    dst_num_pos, _ = self.state_dict()[key].size()
                    dst_patch_shape = self.patch_embed.patch_shape
                    if dst_patch_shape[0] != dst_patch_shape[1]:
                        raise NotImplementedError()
                    num_extra_tokens = dst_num_pos - (dst_patch_shape[0] * 2 - 1) * (dst_patch_shape[1] * 2 - 1)
                    src_size = int((src_num_pos - num_extra_tokens) ** 0.5)
                    dst_size = int((dst_num_pos - num_extra_tokens) ** 0.5)
                    if src_size != dst_size:
                        print_log("Position interpolate for %s from %dx%d to %dx%d" % (
                            key, src_size, src_size, dst_size, dst_size), logger)
                        extra_tokens = rel_pos_bias[-num_extra_tokens:, :]
                        rel_pos_bias = rel_pos_bias[:-num_extra_tokens, :]

                        def geometric_progression(a, r, n):
                            return a * (1.0 - r ** n) / (1.0 - r)

                        left, right = 1.01, 1.5
                        while right - left > 1e-6:
                            q = (left + right) / 2.0
                            gp = geometric_progression(1, q, src_size // 2)
                            if gp > dst_size // 2:
                                right = q
                            else:
                                left = q

                        # if q > 1.13492:
                        #     q = 1.13492

                        dis = []
                        cur = 1
                        for i in range(src_size // 2):
                            dis.append(cur)
                            cur += q ** (i + 1)

                        r_ids = [-_ for _ in reversed(dis)]

                        x = r_ids + [0] + dis
                        y = r_ids + [0] + dis

                        t = dst_size // 2.0
                        dx = np.arange(-t, t + 0.1, 1.0)
                        dy = np.arange(-t, t + 0.1, 1.0)
                        print_log("x = {}".format(x), logger)
                        print_log("dx = {}".format(dx), logger)

                        all_rel_pos_bias = []

                        for i in range(num_attn_heads):
                            z = rel_pos_bias[:, i].view(src_size, src_size).float().numpy()
                            f = interpolate.interp2d(x, y, z, kind='cubic')
                            all_rel_pos_bias.append(
                                torch.Tensor(f(dx, dy)).contiguous().view(-1, 1).to(rel_pos_bias.device))

                        rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)
                        new_rel_pos_bias = torch.cat((rel_pos_bias, extra_tokens), dim=0)
                        state_dict[key] = new_rel_pos_bias

            if 'pos_embed' in state_dict:
                pos_embed_checkpoint = state_dict['pos_embed']
                embedding_size = pos_embed_checkpoint.shape[-1]
                num_patches = self.patch_embed.num_patches
                num_extra_tokens = self.absolute_pos_embed.shape[-2] - num_patches
                # height (== width) for the checkpoint position embedding
                orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
                # height (== width) for the new position embedding
                new_size = int(num_patches ** 0.5)
                # class_token and dist_token are kept unchanged
                if orig_size != new_size:
                    print_log("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size),
                              logger)
                    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                    # only the position tokens are interpolated
                    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                    pos_tokens = torch.nn.functional.interpolate(
                        pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                    new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                    state_dict['pos_embed'] = new_pos_embed

            # interpolate position bias table if needed
            relative_position_bias_table_keys = [k for k in self.state_dict().keys() if
                                                 "relative_position_bias_table" in k]
            for k in relative_position_bias_table_keys:
                if k not in state_dict.keys():
                    continue

                table_pretrained = state_dict[k]
                table_current = self.state_dict()[k]
                L1, nH1 = table_pretrained.size()
                L2, nH2 = table_current.size()
                if nH1 != nH2:
                    print_log(f"Error in loading {k}, pass", logger, level=logging.WARNING)
                else:
                    if L1 != L2:
                        raise ValueError("This part should not be excuted. Please check if geo interpolation work!!")
                        # S1 = int(L1 ** 0.5)
                        # S2 = int(L2 ** 0.5)
                        # table_pretrained_resized = F.interpolate(
                        #      table_pretrained.permute(1, 0).view(1, nH1, S1, S1),
                        #      size=(S2, S2), mode='bicubic')
                        # state_dict[table_key] = table_pretrained_resized.view(nH2, L2).permute(1, 0)
            if 'relative_position_index' in state_dict:
                state_dict.pop('relative_position_index')
            # load state_dict
            self.load_state_dict(state_dict, strict=False)

    def interpolate_pos_encoding(self, x, h, w):
        npatch = x.shape[1]
        N = self.absolute_pos_embed.shape[1]
        if npatch == N and w == h:
            return self.absolute_pos_embed
        patch_pos_embed = self.absolute_pos_embed
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(h0 / math.sqrt(N), w0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(h0) == patch_pos_embed.shape[-2] and int(w0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed

    def prepare_tokens(self, x, mask=None):
        B, _, H, W = x.shape
        Hp, Wp = H // self.patch_size, W // self.patch_size
        features = []

        x = self.patch_embed(x)
        for i, blk in enumerate(self.blocks[:-self.num_main_blocks]):
            x = checkpoint.checkpoint(blk, x) if self.use_checkpoint else blk(x)
            if i == self.merge_indices[0] and 'H' in self.out_indices:
                xp = x.reshape(B, Hp, Wp, 4, 4, -1).permute(
                    0, 5, 1, 3, 2, 4
                ).reshape(B, -1, Hp * 4, Wp * 4).contiguous()
                for _ in range(self.out_indices.count('H')):
                    features.append(xp)
            if i == self.merge_indices[1] and 'M' in self.out_indices:
                xp = x.reshape(B, Hp, Wp, 2, 2, -1).permute(
                    0, 5, 1, 3, 2, 4
                ).reshape(B, -1, Hp * 2, Wp * 2).contiguous()
                for _ in range(self.out_indices.count('M')):
                    features.append(xp)
        x = x[..., 0, 0, :]
        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, H, W)
        return self.pos_drop(x), features

    def forward(self, x):
        inputs = x
        B, _, H, W = x.shape
        Hp, Wp = H // self.patch_size, W // self.patch_size
        x, features = self.prepare_tokens(x)
        rpe_index = self.relative_position_index.view(-1) if self.rpe else None

        for i, blk in enumerate(self.blocks[-self.num_main_blocks:]):
            x = checkpoint.checkpoint(blk, x, rpe_index) if self.use_checkpoint else blk(x, rpe_index)
            if i in self.out_indices:
                xp = x.permute(0, 2, 1).reshape(B, -1, Hp, Wp).contiguous()
                for _ in range(self.out_indices.count(i)):
                    features.append(xp)

        if self.with_fpn:
            ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
            for i in range(len(features)):
                features[i] = ops[i](features[i])

        return tuple(features)
