# Copyright (c) Shanghai AI Lab. All rights reserved.
import logging
import math
import warnings
from typing import Sequence

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from einops import rearrange
from mmcv.cnn.bricks.transformer import PatchEmbed
from mmcv.runner import load_checkpoint
from mmcv.runner.base_module import BaseModule, ModuleList
from mmdet.models import BACKBONES
from mmdet.utils import get_root_logger

from mmdet_custom.models.backbones.base.channel_attn import ECA
from mmdet_custom.models.backbones.base.scan import s_hw, s_wh, sr_hw, sr_wh, s_rhw, s_wrh, sr_rhw, sr_wrh
from mmdet_custom.models.backbones.base.shift import MVCShift
from mmdet_custom.models.backbones.base.wkv import RUN_CUDA
from mmdet_custom.models.utils import DropPath, resize_pos_embed

logger = logging.getLogger(__name__)


class VRWKV_SpatialMix(BaseModule):
    def __init__(self, n_embd, n_layer, layer_id, init_mode='fancy', key_norm=False, with_cp=False):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.device = None
        self.num_experts = 4
        assert n_embd % self.num_experts == 0, 'n_embd must be divisible by 4'
        self.attn_sz = n_embd // self.num_experts

        self.mvc_shift = MVCShift(n_embd)

        self._init_weights(init_mode)

        self.key = nn.Linear(n_embd, self.attn_sz, bias=False)
        self.value = nn.Linear(n_embd, self.attn_sz, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        if key_norm:
            self.key_norm = [nn.LayerNorm(self.attn_sz) for _ in range(self.num_experts)]
        else:
            self.key_norm = None

        self.output = nn.Linear(n_embd, n_embd, bias=False)

        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

        self.with_cp = with_cp

    def _init_weights(self, init_mode):
        multi_dim = self.attn_sz
        if init_mode == 'fancy':
            with torch.no_grad():  # fancy init
                ratio_0_to_1 = (self.layer_id / (self.n_layer - 1))  # 0 to 1
                ratio_1_to_almost0 = (1.0 - (self.layer_id / self.n_layer))  # 1 to ~0

                # fancy time_decay
                decay_speed = torch.ones(multi_dim)
                for h in range(multi_dim):
                    decay_speed[h] = -5 + 8 * (h / (multi_dim - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
                self.spatial_decay = nn.Parameter(decay_speed)

                # fancy time_first
                zigzag = (torch.tensor([(i + 1) % 3 - 1 for i in range(multi_dim)]) * 0.5)
                self.spatial_first = nn.Parameter(torch.ones(multi_dim) * math.log(0.3) + zigzag)
        elif init_mode == 'local':
            self.spatial_decay = nn.Parameter(torch.ones(multi_dim))
            self.spatial_first = nn.Parameter(torch.ones(multi_dim))
        elif init_mode == 'global':
            self.spatial_decay = nn.Parameter(torch.zeros(multi_dim))
            self.spatial_first = nn.Parameter(torch.zeros(multi_dim))
        else:
            raise NotImplementedError

    def jit_func(self, x, patch_resolution):
        h, w = patch_resolution
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        x = self.mvc_shift(x)
        x = rearrange(x, "b c h w -> b (h w) c")

        # Use xk, xv, xr to produce k, v, r
        k = self.key(x)
        v = self.value(x)
        r = self.receptance(x)
        sr = torch.sigmoid(r)

        return sr, k, v

    def forward(self, x, patch_resolution=None):
        def _inner_forward(x):
            B, T, C = x.size()
            self.device = x.device

            h, w = patch_resolution
            sr, k, v = self.jit_func(x, patch_resolution)

            k = rearrange(k, "b (h w) c -> b c h w", h=h, w=w)
            v = rearrange(v, "b (h w) c -> b c h w", h=h, w=w)

            scan_func = [s_hw, s_wh, s_rhw, s_wrh]
            re_scan_func = [sr_hw, sr_wh, sr_rhw, sr_wrh]

            ks = torch.cat([scan_func[i](k) for i in range(self.num_experts)], dim=2)  # b (h w) (c e)
            vs = torch.cat([scan_func[i](v) for i in range(self.num_experts)], dim=2)  # b (h w) (c e)

            spatial_decay = self.spatial_decay.repeat(self.num_experts) / T
            spatial_first = self.spatial_first.repeat(self.num_experts) / T

            expert_output = RUN_CUDA(B, T, C, spatial_decay, spatial_first, ks, vs)
            expert_outputs = torch.split(expert_output, self.attn_sz, dim=2)  # (b (h w) c) * e
            expert_outputs = [rearrange(re_scan_func[i](expert_outputs[i], h, w), "b c h w -> b (h w) c")
                              for i in range(self.num_experts)]

            if self.key_norm is not None:
                expert_outputs = [self.key_norm[i](expert_outputs[i]) for i in range(self.num_experts)]

            expert_outputs = torch.cat(expert_outputs, dim=2)  # b (h w) (c e)
            expert_outputs = expert_outputs * sr

            x = self.output(expert_outputs)
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


class VRWKV_ChannelMix(BaseModule):
    def __init__(self, n_embd, n_layer, layer_id, hidden_rate=2, init_mode='fancy',
                 key_norm=False, with_cp=False):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.with_cp = with_cp
        self._init_weights(init_mode)

        self.mvc_shift = MVCShift(n_embd)

        self.channel_attn = ECA(n_embd)

        hidden_sz = hidden_rate * n_embd
        self.key = nn.Linear(n_embd, hidden_sz, bias=False)
        if key_norm:
            self.key_norm = nn.LayerNorm(hidden_sz)
        else:
            self.key_norm = None
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)

        self.value.scale_init = 0
        self.receptance.scale_init = 0

    def _init_weights(self, init_mode):
        pass

    def forward(self, x, patch_resolution=None):
        def _inner_forward(x):
            h, w = patch_resolution
            x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
            x = self.mvc_shift(x)
            x = rearrange(x, "b c h w -> b (h w) c")

            k = self.key(x)
            k = torch.square(torch.relu(k))
            if self.key_norm is not None:
                k = self.key_norm(k)
            kv = self.value(k)
            x = torch.sigmoid(self.receptance(x)) * kv

            x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
            x = self.channel_attn(x)
            x = rearrange(x, "b c h w -> b (h w) c")
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


class Block(BaseModule):
    def __init__(self, n_embd, n_layer, layer_id, drop_path=0., hidden_rate=2,
                 init_mode='fancy', init_values=None, post_norm=False, key_norm=False, with_cp=False):
        super().__init__()
        self.layer_id = layer_id
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(n_embd)

        self.att = VRWKV_SpatialMix(n_embd, n_layer, layer_id, init_mode, key_norm=key_norm, with_cp=with_cp)

        self.ffn = VRWKV_ChannelMix(n_embd, n_layer, layer_id, hidden_rate, init_mode, key_norm=key_norm,
                                    with_cp=with_cp)
        self.layer_scale = (init_values is not None)
        self.post_norm = post_norm
        if self.layer_scale:
            self.gamma1 = nn.Parameter(init_values * torch.ones(n_embd), requires_grad=True)
            self.gamma2 = nn.Parameter(init_values * torch.ones(n_embd), requires_grad=True)
        self.with_cp = with_cp

    def forward(self, x, patch_resolution=None):
        def _inner_forward(x):
            if self.layer_id == 0:
                x = self.ln0(x)
            if self.post_norm:
                if self.layer_scale:
                    x = x + self.drop_path(self.gamma1 * self.ln1(self.att(x, patch_resolution)))
                    x = x + self.drop_path(self.gamma2 * self.ln2(self.ffn(x, patch_resolution)))
                else:
                    x = x + self.drop_path(self.ln1(self.att(x, patch_resolution)))
                    x = x + self.drop_path(self.ln2(self.ffn(x, patch_resolution)))
            else:
                if self.layer_scale:
                    x = x + self.drop_path(self.gamma1 * self.att(self.ln1(x), patch_resolution))
                    x = x + self.drop_path(self.gamma2 * self.ffn(self.ln2(x), patch_resolution))
                else:
                    x = x + self.drop_path(self.att(self.ln1(x), patch_resolution))
                    x = x + self.drop_path(self.ffn(self.ln2(x), patch_resolution))
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


@BACKBONES.register_module()
class VVRWKV(BaseModule):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 out_indices=-1,
                 drop_rate=0.,
                 embed_dims=256,
                 depth=12,
                 drop_path_rate=0.,
                 init_mode='fancy',
                 post_norm=False,
                 key_norm=False,
                 init_values=None,
                 hidden_rate=2,
                 final_norm=True,
                 interpolate_mode='bicubic',
                 pretrained=None,
                 with_cp=False,
                 init_cfg=None):
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            self.init_cfg = init_cfg
        else:
            raise TypeError('pretrained must be a str or None')
        super().__init__(self.init_cfg)
        self.embed_dims = embed_dims
        self.num_extra_tokens = 0
        self.num_layers = depth
        self.drop_path_rate = drop_path_rate
        logger = get_root_logger()
        logger.info(f'layer_scale: {init_values is not None}')

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            input_size=img_size,
            embed_dims=self.embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=patch_size,
            bias=True)

        self.patch_resolution = self.patch_embed.init_out_size
        num_patches = self.patch_resolution[0] * self.patch_resolution[1]

        # Set position embedding
        self.interpolate_mode = interpolate_mode
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, self.embed_dims))

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = self.num_layers + index
            assert 0 <= out_indices[i] <= self.num_layers, \
                f'Invalid out_indices {index}'
        self.out_indices = out_indices
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.layers = ModuleList()
        for i in range(self.num_layers):
            self.layers.append(Block(
                n_embd=embed_dims,
                n_layer=depth,
                layer_id=i,
                hidden_rate=hidden_rate,
                drop_path=dpr[i],
                init_mode=init_mode,
                post_norm=post_norm,
                key_norm=key_norm,
                init_values=init_values,
                with_cp=with_cp
            ))

        self.final_norm = final_norm
        if final_norm:
            self.ln1 = nn.LayerNorm(self.embed_dims)

    def init_weights(self):
        logger = get_root_logger()
        if self.init_cfg is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            load_checkpoint(self,
                            self.init_cfg['checkpoint'], map_location='cpu', logger=logger, strict=False,
                            revise_keys=[(r'^backbone.', '')])
            logger.warn(f'Load pre-trained model for '
                        f'{self.__class__.__name__} from original repo')

    def forward(self, x):
        B = x.shape[0]
        x, patch_resolution = self.patch_embed(x)

        x = x + resize_pos_embed(
            self.pos_embed,
            self.patch_resolution,
            patch_resolution,
            mode=self.interpolate_mode,
            num_extra_tokens=self.num_extra_tokens)

        x = self.drop_after_pos(x)

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x, patch_resolution)

            if i == len(self.layers) - 1 and self.final_norm:
                x = self.ln1(x)

            if i in self.out_indices:
                B, _, C = x.shape
                patch_token = x.reshape(B, *patch_resolution, C)
                patch_token = patch_token.permute(0, 3, 1, 2)

                out = patch_token
                outs.append(out)
        return tuple(outs)


if __name__ == "__main__":
    model = VVRWKV(
        img_size=224,
        patch_size=16,
        in_channels=3,
        out_indices=[2, 5, 8, 11],
        drop_rate=0.,
        embed_dims=192,
        depth=12,
        pretrained="checkpoints/latest.pth"
    ).cuda()

    from thop import profile, clever_format

    x = torch.randn(1, 3, 224, 224).cuda()
    # out = model(x)
    flops, params = profile(model, inputs=(x,))
    print(f"flops: {flops}, params: {params}")
    flops, params = clever_format([flops, params])
    print(f"flops: {flops}, params: {params}")
