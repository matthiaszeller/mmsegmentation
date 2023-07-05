
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from mmcv.cnn import build_norm_layer
from mmengine.logging import print_log
from mmengine.model import BaseModule
from mmengine.runner import CheckpointLoader
from timm.models.layers import trunc_normal_

from mmseg.registry import MODELS
from .cswin import img2windows, windows2img, CSWinBlock
from ..utils import PatchEmbed


# class Feedforward(nn.Module):
#     def __init__(self, in_features, hidden_features, drop=0.):
#         super().__init__()
#         self.net=nn.Sequential(
#                         nn.Linear(in_features, hidden_features),
#                         nn.GELU(),
#                         nn.Linear(hidden_features, in_features),
#                         nn.Dropout(drop)
#         )

#     def forward(self, x):
#         x = self.net(x)
#         return x

# class Mlp(nn.Module):
#     def __init__(self, dim, num_patch, token_dim, channel_dim, drop=0.):
#         super().__init__()
#         self.token_mixer = nn.Sequential(
#             nn.LayerNorm(dim),
#             Rearrange('b n d -> b d n'),
#             Feedforward(num_patch, token_dim, drop),
#             Rearrange('b d n -> b n d')
#         )
#         self.channel_mixer = nn.Sequential(
#             nn.LayerNorm(dim),
#             Feedforward(dim, channel_dim, drop)
#         )
#     def forward(self, x):
#         x = x + self.token_mixer(x)
#         x = x + self.channel_mixer(x) 
#         return x

class LePEAttention(nn.Module):
    def __init__(self, dim, resolution, idx, split_size=7, dim_out=None, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        """Not supported now, since we have cls_tokens now.....
        """
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.resolution = resolution
        self.split_size = split_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.idx = idx
        if idx == -1:
            H_sp, W_sp = self.resolution, self.resolution
        elif idx == 0:
            H_sp, W_sp = self.resolution, self.split_size
        elif idx == 1:
            W_sp, H_sp = self.resolution, self.split_size
        else:
            print("ERROR MODE", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp

        self.H_sp_ = self.H_sp
        self.W_sp_ = self.W_sp

        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.get_v1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)

        #         self.get_v = nn.Conv2d(dim, dim, kernel_size=1, stride=1)

        #         self.get_v = nn.Sequential(
        #                         nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,groups=dim),
        #                         nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        #         )
        #         self.get_v = nn.Conv2d(dim//2, dim//2, kernel_size=3, stride=1, padding=1,groups=dim//2)
        #         self.get_v1 = nn.Conv2d(dim//2, dim//2, kernel_size=3, stride=1, padding=1)

        self.attn_drop = nn.Dropout(attn_drop)

    def im2cswin(self, x):
        B, C, H, W = x.shape
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp * self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def get_rpe(self, x, func, func1):
        B, C, H, W = x.shape
        H_sp, W_sp = self.H_sp, self.W_sp
        x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp)  ### B', C, H', W'
        B, C, H, W = x.shape

        #         xs = torch.chunk(x, 2, 1)

        #         rpe = func(x) ### B', C, H', W'
        rpe = func1(x + func(x))

        #         rpe0 = func(xs[0])
        #         rpe0 = nn.functional.gelu(self.ln(rpe0))
        #         rpe = self.get_v1(xs[1]+rpe0)
        #         rpe = nn.functional.gelu(self.ln(rpe))
        #         rpe = torch.cat((rpe0, rpe),1)
        rpe = rpe.reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp).permute(0, 1, 3, 2).contiguous()

        x = x.reshape(-1, self.num_heads, C // self.num_heads, self.H_sp * self.W_sp).permute(0, 1, 3, 2).contiguous()
        return x, rpe

    def forward(self, temp):
        """
        x: B N C
        mask: B N N
        """
        B, _, C, H, W = temp.shape

        idx = self.idx
        if idx == -1:
            H_sp, W_sp = H, W
        elif idx == 0:
            H_sp, W_sp = H, self.split_size
        elif idx == 1:
            H_sp, W_sp = self.split_size, W
        else:
            print("ERROR MODE in forward", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp

        ### padding for split window
        H_pad = (self.H_sp - H % self.H_sp) % self.H_sp
        W_pad = (self.W_sp - W % self.W_sp) % self.W_sp
        top_pad = H_pad // 2
        down_pad = H_pad - top_pad
        left_pad = W_pad // 2
        right_pad = W_pad - left_pad
        H_ = H + H_pad
        W_ = W + W_pad

        qkv = F.pad(temp, (left_pad, right_pad, top_pad, down_pad))  ### B,3,C,H',W'
        qkv = qkv.permute(1, 0, 2, 3, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self.im2cswin(q)
        k = self.im2cswin(k)
        v, rpe = self.get_rpe(v, self.get_v, self.get_v1)

        ### Local attention
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N

        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)

        attn = self.attn_drop(attn)

        x = (attn @ v) + rpe
        x = x.transpose(1, 2).reshape(-1, self.H_sp * self.W_sp, C)  # B head N N @ B head N C

        ### Window2Img
        x = windows2img(x, self.H_sp, self.W_sp, H_, W_)  # B H_ W_ C
        x = x[:, top_pad:H + top_pad, left_pad:W + left_pad, :]
        x = x.reshape(B, -1, C)

        return x


# class Merge_Block(nn.Module):
#     def __init__(self, dim, dim_out, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.conv = nn.Conv2d(dim, dim_out, 3, 2, 1)
#         self.norm = norm_layer(dim_out)

#     def forward(self, x, H, W):
#         B, new_HW, C = x.shape
#         x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
#         x = self.conv(x)
#         B, C, H, W = x.shape
#         x = x.view(B, C, -1).transpose(-2, -1).contiguous()
#         x = self.norm(x)

#         return x, H, W

# def add_avgmax_pool2d(x, output_size=(1,1)):
#     x_avg = F.adaptive_avg_pool2d(x, output_size)
#     x_max = F.adaptive_max_pool2d(x, output_size)
#     return 0.3 * x_avg + 0.7 * x_max


# MSRD
class Merge_Block(nn.Module):
    def __init__(self, s, dim, dim_out, h, norm_cfg=dict(type='LN')):
        super().__init__()
        self.s = s
        assert dim_out % self.s == 0
        #         self.h = 512//(2**((dim//64)+1)) if dim != 256 else 32
        self.h = h

        self.inconv = nn.Sequential(
            nn.Conv2d(dim, dim_out, 1, 1, 0),
            nn.LayerNorm([dim_out, self.h, self.h]),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(dim_out // self.s, dim_out // self.s, 3, 1, 1),
            #             nn.Conv2d(dim_out//4, dim_out//4, 3, 1, 1, groups=dim_out//4),#d
            nn.LayerNorm([dim_out // self.s, self.h, self.h]),
        )
        self.ghconv = nn.Sequential(
            nn.Conv2d(dim_out // self.s, dim_out // self.s, 1, 1, 0),
            nn.LayerNorm([dim_out // self.s, self.h, self.h]),
        )

        self.outconv = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, 1, 1, 0),
            #             nn.Conv2d(dim_out, dim_out, 1, 1, 0, groups=dim_out),#e
            nn.LayerNorm([dim_out, self.h, self.h]),
        )
        self.pool = nn.MaxPool2d(3, 2, 1)

        self.norm = build_norm_layer(norm_cfg, dim_out)[1]

    def forward(self, x, H, W):
        B, new_HW, C = x.shape
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)

        x = F.gelu(self.inconv(x))

        xs = torch.chunk(x, self.s, 1)
        ys = []
        ys.append(xs[0])
        #         ys.append(F.gelu(self.conv(xs[1])))
        # 2
        #         ys.append(F.gelu(self.conv(xs[1])+ys[0]))
        #         ys.append(F.gelu(self.conv(xs[2])+xs[1]))
        #         ys.append(F.gelu(self.conv(xs[3])+xs[2]))
        # 3
        #         ys.append(F.gelu(self.conv(xs[1])+ys[0]))
        #         ys.append(F.gelu(self.conv(xs[2])+xs[1]+xs[0]))
        #         ys.append(F.gelu(self.conv(xs[3])+xs[2]+xs[1]+xs[0]))
        # 4
        #         ys.append(F.gelu(self.conv(xs[1])+self.ghconv(ys[0])))
        #         ys.append(F.gelu(self.conv(xs[2])+self.ghconv(xs[1])+self.ghconv(xs[0])))
        #         ys.append(F.gelu(self.conv(xs[3])+self.ghconv(xs[2])+self.ghconv(xs[1])+self.ghconv(xs[0])))
        # 1
        #         ys.append(F.gelu(self.conv(xs[1])+self.ghconv(ys[0])))
        #         ys.append(F.gelu(self.conv(xs[2])+self.ghconv(ys[1])))
        #         ys.append(F.gelu(self.conv(xs[3])+self.ghconv(ys[2])))

        for i in range(1, self.s):
            ys.append(F.gelu(self.conv(xs[i]) + self.ghconv(ys[i - 1])))
        # 5
        #         ys.append(F.gelu(self.conv(xs[1])+self.ghconv(xs[0])))
        #         ys.append(F.gelu(self.conv(xs[2])+self.ghconv(self.conv(xs[1]))+self.ghconv(xs[0])))
        #         ys.append(F.gelu(self.conv(xs[3])+self.ghconv(self.conv(xs[2]))+self.ghconv(self.conv(xs[1]))+self.ghconv(xs[0])))
        #         ys.append(xs[0])
        #         ys.append(F.gelu(self.conv(xs[1])))
        #         ys.append(F.gelu(self.conv(xs[2]+ys[1])))
        #         ys.append(F.gelu(self.conv(xs[3]+ys[2])))
        ys = torch.cat(ys, 1)
        ys = self.outconv(ys)
        x = F.gelu(ys + x)
        x = self.pool(x)
        #         x = add_avgmax_pool2d(x,(x.shape[2]//2,x.shape[3]//2))
        B, C, H, W = x.shape
        x = x.view(B, C, -1).transpose(-2, -1).contiguous()
        x = self.norm(x)

        return x, H, W


# class Merge_Block(nn.Module):
#     def __init__(self, dim, dim_out, H, norm_layer=nn.LayerNorm):
#         super().__init__()
#         assert dim_out % 2 == 0
#         self.H=H
#         self.W=H
#         self.inconv = nn.Sequential(
#             nn.Conv2d(dim, dim_out, 1, 1, 0),
#             nn.LayerNorm([dim_out,self.H,self.W]),
#             nn.GELU()
#         )
#         self.conv = nn.Sequential(
#             nn.Conv2d(dim_out//2, dim_out//2, 3, 1, 1),
#             nn.LayerNorm([dim_out//2,self.H, self.W]),
#         )

#         self.outconv = nn.Sequential(
#             nn.Conv2d(dim_out, dim_out, 1, 1, 0),
#             nn.LayerNorm([dim_out,self.H,self.W]),
#         )
#         self.pool = nn.MaxPool2d(3, 2, 1)
#         self.norm = norm_layer(dim_out)

#     def forward(self, x, H, W):
#         B, new_HW, C = x.shape
#         x = x.transpose(-2, -1).contiguous().view(B, C, H, W)

#         x = self.inconv(x)
#         xs = torch.chunk(x, 2, 1)
#         ys = []
#         ys.append(xs[0])
#         ys.append(F.gelu(self.conv(xs[1]+ys[0])))
#         ys = torch.cat(ys, 1)
#         ys = self.outconv(ys)
#         x = F.gelu(ys+x)
#         x = self.pool(x)
#         B, C, H, W = x.shape
#         x = x.view(B, C, -1).transpose(-2, -1).contiguous()
#         x = self.norm(x)

#         return x, H, W

# class Merge_Block(nn.Module):
#     def __init__(self, dim, dim_out, norm_layer=nn.LayerNorm):
#         super().__init__()
#         assert dim_out % 4 == 0
#         self.h = 512//(2**((dim//64)+1)) if dim != 256 else 32
#         self.inconv = nn.Sequential(
#             nn.Conv2d(dim, dim_out, 1, 1, 0),
#             nn.LayerNorm([dim_out,self.h,self.h]),
#             nn.GELU()
#         )
#         self.conv = nn.Sequential(
#             nn.Conv2d(dim_out, dim_out, 3, 2, 1),
#             nn.LayerNorm([dim_out,self.h//2, self.h//2]),
#             nn.GELU()
#         )

#         self.outconv = nn.Sequential(
#             nn.Conv2d(dim_out, dim_out, 1, 1, 0),
#             nn.LayerNorm([dim_out,self.h//2,self.h//2]),
#             nn.GELU()
#         )

#         self.pool = nn.AdaptiveAvgPool2d((self.h//2, self.h//2))
#         self.norm = norm_layer(dim_out)

#     def forward(self, x, H, W):
#         B, new_HW, C = x.shape
#         x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
#         x = self.inconv(x)
#         x = self.conv(x)
#         x = self.outconv(x)
#         B, C, H, W = x.shape
#         x = x.view(B, C, -1).transpose(-2, -1).contiguous()
#         x = self.norm(x)

#         return x, H, W

@MODELS.register_module()
class CSWinTransformer2(BaseModule):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 img_size=224,
                 patch_size=4,
                 in_channels=3,
                 embed_dim=64,
                 depth=(1, 2, 21, 1),
                 split_size=(1, 2, 5, 5),
                 s=(4, 4, 4),
                 num_heads=(1, 2, 4, 8),
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 hybrid_backbone=None,
                 norm_cfg=dict(type='LN'),
                 use_chk=False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        heads = num_heads
        self.use_chk = use_chk
        # Overlapping convolutional embeddings
        # produces patches with (H, W) dimensions divided by 4
        self.stage1_conv_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dim,
            conv_type='Conv2d',
            kernel_size=7,
            stride=4,
            # in mmseg transformers, padding is now 'corner'
            padding=2,
            norm_cfg=norm_cfg, # TODO patch_norm param
            init_cfg=None
        )

        self.norm1 = build_norm_layer(norm_cfg, embed_dim)[1]

        curr_dim = embed_dim
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, np.sum(depth))]  # stochastic depth decay rule
        self.stage1 = nn.ModuleList([
            CSWinBlock(
                dim=curr_dim, num_heads=heads[0], patches_resolution=224 // 4, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[0],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_cfg=norm_cfg)
            for i in range(depth[0])])

        H1, W1 = img_size // 4, img_size // 4
        self.merge1 = Merge_Block(s[0], curr_dim, curr_dim * (heads[1] // heads[0]), H1)
        #         self.merge1 = Merge_Block(curr_dim, curr_dim*(heads[1]//heads[0]), img_size//4)
        curr_dim = curr_dim * (heads[1] // heads[0])
        self.norm2 = build_norm_layer(norm_cfg, curr_dim)[1]
        self.stage2 = nn.ModuleList(
            [CSWinBlock(
                dim=curr_dim, num_heads=heads[1], patches_resolution=224 // 8, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[1],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:1]) + i], norm_cfg=norm_cfg)
                for i in range(depth[1])])

        H2, W2 = img_size // 8, img_size // 8
        self.merge2 = Merge_Block(s[1], curr_dim, curr_dim * (heads[2] // heads[1]), H2)
        #         self.merge2 = Merge_Block(curr_dim, curr_dim*(heads[2]//heads[1]), img_size//8)
        curr_dim = curr_dim * (heads[2] // heads[1])
        self.norm3 = build_norm_layer(norm_cfg, curr_dim)[1]
        temp_stage3 = []
        temp_stage3.extend(
            [CSWinBlock(
                dim=curr_dim, num_heads=heads[2], patches_resolution=224 // 16, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[2],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:2]) + i], norm_cfg=norm_cfg)
                for i in range(depth[2])])

        self.stage3 = nn.ModuleList(temp_stage3)

        H3, W3 = img_size // 16, img_size // 16
        self.merge3 = Merge_Block(s[2], curr_dim, curr_dim * (heads[3] // heads[2]), H3)
        #         self.merge3 = Merge_Block(curr_dim, curr_dim*(heads[3]//heads[2]), img_size//16)
        curr_dim = curr_dim * (heads[3] // heads[2])
        self.stage4 = nn.ModuleList(
            [CSWinBlock(
                dim=curr_dim, num_heads=heads[3], patches_resolution=img_size // 32, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[-1],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:-1]) + i], norm_cfg=norm_cfg, last_stage=True)
                for i in range(depth[-1])])

        self.norm4 = build_norm_layer(norm_cfg, curr_dim)[1]

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)

        if self.init_cfg is None:
            print_log(f'No pre-trained weights for {self.__class__.__name__}, '
                      f'training start from scratch')

        else:
            assert 'checkpoint' in self.init_cfg, 'specify `pretrained` in `init_cfg`'
            state_dict = CheckpointLoader.load_checkpoint(
                self.init_cfg['checkpoint'], logger=None, map_location='cpu')

            if 'state_dict' in state_dict:
                _state_dict = state_dict['state_dict']
            else:
                _state_dict = state_dict

            state_dict = OrderedDict()
            # strip prefix of backbone
            for k, v in _state_dict.items():
                if k.startswith('backbone.'):
                    state_dict[k[9:]] = v
                else:
                    state_dict[k] = v

            # strip prefix of state_dict
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}

            # skip "for MoBY, load model of online branch" of mmcv_custom.checkpoint.load_checkpoint

            # reshape absolute position embedding -- copy-paste of swin, same in mmcv_custom (up to contiguous())
            if state_dict.get('absolute_pos_embed') is not None:
                absolute_pos_embed = state_dict['absolute_pos_embed']
                N1, L, C1 = absolute_pos_embed.size()
                N2, C2, H, W = self.absolute_pos_embed.size()
                if N1 != N2 or C1 != C2 or L != H * W:
                    print_log('Error in loading absolute_pos_embed, pass')
                else:
                    state_dict['absolute_pos_embed'] = absolute_pos_embed.view(
                        N2, H, W, C2).permute(0, 3, 1, 2).contiguous()

            # interpolate position bias table if needed -- copy-paste of swin, same as mmcv_custom up to contiguous()
            relative_position_bias_table_keys = [
                k for k in state_dict.keys()
                if 'relative_position_bias_table' in k
            ]
            for table_key in relative_position_bias_table_keys:
                table_pretrained = state_dict[table_key]
                table_current = self.state_dict()[table_key]
                L1, nH1 = table_pretrained.size()
                L2, nH2 = table_current.size()
                if nH1 != nH2:
                    print_log(f'Error in loading {table_key}, pass')
                elif L1 != L2:
                    S1 = int(L1 ** 0.5)
                    S2 = int(L2 ** 0.5)
                    table_pretrained_resized = F.interpolate(
                        table_pretrained.permute(1, 0).reshape(1, nH1, S1, S1),
                        size=(S2, S2),
                        mode='bicubic')
                    state_dict[table_key] = table_pretrained_resized.view(
                        nH2, L2).permute(1, 0).contiguous()

            # load state_dict
            self.load_state_dict(state_dict, strict=False)

    def save_out(self, x, norm, H, W):
        x = norm(x)
        B, N, C = x.shape
        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return x

    def forward_features(self, x):
        x, (H, W) = self.stage1_conv_embed(x)

        out = []
        for blk in self.stage1:
            blk.H = H
            blk.W = W
            if self.use_chk:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        out.append(self.save_out(x, self.norm1, H, W))

        for pre, blocks, norm in zip([self.merge1, self.merge2, self.merge3],
                                     [self.stage2, self.stage3, self.stage4],
                                     [self.norm2, self.norm3, self.norm4]):

            x, H, W = pre(x, H, W)
            for blk in blocks:
                blk.H = H
                blk.W = W
                if self.use_chk:
                    x = checkpoint.checkpoint(blk, x)
                else:
                    x = blk(x)

            out.append(self.save_out(x, norm, H, W))

        return tuple(out)

    def forward(self, x):
        x = self.forward_features(x)
        return x
