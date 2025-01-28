import logging
import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from einops import rearrange
from mmcv.cnn.bricks.transformer import PatchEmbed
from mmcv.runner.base_module import BaseModule
from mmseg.models import BACKBONES

from mmseg_custom.models.backbones.base.down_sample import Down_wt
from mmseg_custom.models.backbones.base.scan import s_hw, s_wh, sr_hw, sr_wh, s_rhw, s_wrh, sr_rhw, sr_wrh
from mmseg_custom.models.backbones.base.shift import MVCShift
from mmseg_custom.models.backbones.base.vvrwkv import Block, ECA
from mmseg_custom.models.backbones.base.wkv import RUN_CUDA
from mmseg_custom.models.utils import DropPath, resize_pos_embed

logger = logging.getLogger(__name__)


class VRWKV_SpatialMix(BaseModule):
    def __init__(self, n_embd, n_layer, layer_id, init_mode='fancy', key_norm=False, with_cp=False):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.device = None
        self.attn_sz = n_embd

        # self.uw_shift = UWShift(n_features=n_embd, kernel_size=7)
        # self.omni_shift = OmniShift(dim=n_embd)
        self.mvc_shift = MVCShift(n_embd)

        self.num_experts = 4
        self.gate = nn.Conv2d(n_embd, self.num_experts, 1)

        self._init_weights(init_mode)

        self.key = nn.Linear(n_embd, self.attn_sz, bias=False)
        self.value = nn.Linear(n_embd, self.attn_sz, bias=False)
        self.receptance = nn.Linear(n_embd, self.attn_sz * self.num_experts, bias=False)
        if key_norm:
            self.key_norm = nn.LayerNorm(self.attn_sz)
        else:
            self.key_norm = None
        self.output = nn.Linear(self.attn_sz, n_embd, bias=False)

        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

        self.with_cp = with_cp

    def _init_weights(self, init_mode):
        multi_dim = self.n_embd
        if init_mode == 'fancy':
            with torch.no_grad():  # fancy init
                ratio_0_to_1 = (self.layer_id / (self.n_layer - 1))  # 0 to 1

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
            score = F.softmax(
                self.gate(rearrange(x, "b (h w) c -> b c h w", h=h, w=w)), dim=1
            )  # b e h w
            score = score.unsqueeze(1)  # b 1 e h w

            sr, k, v = self.jit_func(x, patch_resolution)

            k = rearrange(k, "b (h w) c -> b c h w", h=h, w=w)
            v = rearrange(v, "b (h w) c -> b c h w", h=h, w=w)

            scan_func = [s_hw, s_wh, s_rhw, s_wrh]
            re_scan_func = [sr_hw, sr_wh, sr_rhw, sr_wrh]

            ks = torch.cat(
                [scan_func[i](k) for i in range(self.num_experts)], dim=2
            )  # b (h w) (c e)
            vs = torch.cat(
                [scan_func[i](v) for i in range(self.num_experts)], dim=2
            )  # b (h w) (c e)

            spatial_decay = self.spatial_decay.repeat(self.num_experts) / T
            spatial_first = self.spatial_first.repeat(self.num_experts) / T

            expert_output = RUN_CUDA(B, T, C * self.num_experts, spatial_decay, spatial_first, ks, vs)
            expert_outputs = [
                expert_output[:, :, i * self.attn_sz: (i + 1) * self.attn_sz]
                for i in range(self.num_experts)
            ]  # (b (h w) c) * e
            expert_outputs = [rearrange(re_scan_func[i](expert_outputs[i], h, w), "b c h w -> b (h w) c")
                              for i in range(self.num_experts)]

            if self.key_norm is not None:
                expert_outputs = [self.key_norm(eo) for eo in expert_outputs]

            expert_outputs = torch.cat(expert_outputs, dim=2)  # b (h w) (c e)
            expert_outputs = expert_outputs * sr
            expert_outputs = rearrange(expert_outputs, "b (h w) (c e) -> b c e h w",
                                       h=h, w=w, c=self.attn_sz, e=self.num_experts)  # b c e h w
            prediction = torch.sum(expert_outputs * score, dim=2)

            x = rearrange(prediction, "b c h w -> b (h w) c")
            x = self.output(x)
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


class VRWKV_ChannelMix(BaseModule):
    def __init__(self, n_embd, n_layer, layer_id, hidden_rate=4, init_mode='fancy',
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
    def __init__(self, n_embd, n_layer, layer_id, drop_path=0., hidden_rate=4,
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


class Layer(BaseModule):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, patch_resolution=None):
        for block in self.blocks:
            x = block(x, patch_resolution)
        return x


class InteractionBlock(BaseModule):
    def __init__(self, in_feats: list[int], interpolate_mode="bilinear"):
        super().__init__()
        self.in_feats = in_feats
        self.cat_feats = sum(in_feats)
        self.interpolate_mode = interpolate_mode

        self.global_conv = nn.Conv2d(self.cat_feats, self.cat_feats, 1, bias=False)

        self.values = nn.ModuleList(nn.Conv2d(self.cat_feats, feat, 1, bias=False) for feat in in_feats)
        self.keys = nn.ModuleList(nn.Conv2d(feat, feat, 1, bias=False) for feat in in_feats)
        self.receptances = nn.ModuleList(nn.Conv2d(feat, feat, 1, bias=False) for feat in in_feats)
        self.outputs = nn.ModuleList(nn.Conv2d(feat, feat, 1, bias=False) for feat in in_feats)

        self.spatial_decays = nn.ParameterList(nn.Parameter(torch.ones(feat)) for feat in in_feats)
        self.spatial_firsts = nn.ParameterList(nn.Parameter(torch.ones(feat)) for feat in in_feats)

    def forward(self, xs):
        """
        xs: list (B C H W) tensor
        """
        max_h, max_w = max(x.shape[2] for x in xs), max(x.shape[3] for x in xs)

        # fuse all features
        aligned_xs = []
        for x in xs:
            aligned_x = F.interpolate(x, size=(max_h, max_w), mode=self.interpolate_mode)
            aligned_xs.append(aligned_x)
        global_x = torch.cat(aligned_xs, dim=1)
        global_x = self.global_conv(global_x)

        outputs = []
        for i, x in enumerate(xs):
            b, c, h, w = x.shape
            g_x = global_x if h == max_h and w == max_w \
                else F.interpolate(global_x, size=(h, w), mode=self.interpolate_mode)
            k = rearrange(self.keys[i](x), "b c h w -> b (h w) c")
            v = rearrange(self.values[i](g_x), "b c h w -> b (h w) c")
            wkv = RUN_CUDA(b, h * w, c, self.spatial_decays[i], self.spatial_firsts[i], k, v)
            wkv = rearrange(wkv, "b (h w) c -> b c h w", h=h, w=w)
            sr = torch.sigmoid(self.receptances[i](x))
            rwkv = sr * wkv
            output = self.outputs[i](rwkv)
            x = x + output
            outputs.append(x)
        return outputs


@BACKBONES.register_module()
class RSRWKV(BaseModule):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 out_indices=-1,
                 drop_rate=0.,
                 embed_dims=192,
                 layer_depth=3,
                 drop_path_rate=0.,
                 init_mode='fancy',
                 post_norm=False,
                 key_norm=False,
                 init_values=None,
                 hidden_rate=2,
                 final_norm=True,
                 interpolate_mode='bicubic',
                 with_cp=False,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.num_extra_tokens = 0
        self.num_layers = 4
        self.layer_depth = layer_depth
        self.drop_path_rate = drop_path_rate

        if isinstance(embed_dims, int):
            embed_dims = [embed_dims] * self.num_layers
        assert isinstance(embed_dims, Sequence), \
            f'"embed_dims" must by a sequence or int, ' \
            f'get {type(embed_dims)} instead.'
        assert len(embed_dims) == self.num_layers, \
            f'"embed_dims" must have {self.num_layers} elements, ' \
            f'get {len(embed_dims)} instead.'
        self.embed_dims = embed_dims

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            input_size=img_size,
            embed_dims=self.embed_dims[0],
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=patch_size,
            bias=True)

        self.patch_resolution = self.patch_embed.init_out_size
        num_patches = self.patch_resolution[0] * self.patch_resolution[1]

        # Set position embedding
        self.interpolate_mode = interpolate_mode
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dims[0]))

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

        all_layers = self.num_layers * self.layer_depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, all_layers)]

        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            blocks = []
            for j in range(self.layer_depth):
                idx = i * self.layer_depth + j
                blocks.append(Block(
                    n_embd=embed_dims[i],
                    n_layer=all_layers,
                    layer_id=idx,
                    hidden_rate=hidden_rate,
                    drop_path=dpr[idx],
                    init_mode=init_mode,
                    post_norm=post_norm,
                    key_norm=key_norm,
                    init_values=init_values,
                    with_cp=with_cp
                ))
            self.layers.append(Layer(blocks))

        self.down_samples = nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.down_samples.append(Down_wt(self.embed_dims[i], self.embed_dims[i + 1]))

        self.global_interaction = InteractionBlock(
            [dim for i, dim in enumerate(self.embed_dims) if i in self.out_indices])

        self.final_norm = final_norm
        if final_norm:
            self.ln1 = nn.LayerNorm(self.embed_dims[-1])

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

            if i != len(self.layers) - 1:
                x = rearrange(x, 'b (h w) c -> b c h w', h=patch_resolution[0], w=patch_resolution[1])
                x = self.down_samples[i](x)
                patch_resolution = tuple(x.shape[2:])
                x = rearrange(x, 'b c h w -> b (h w) c')

        outs = self.global_interaction(outs)
        return tuple(outs)


if __name__ == "__main__":
    model = RSRWKV(
        img_size=224,
        patch_size=16,
        embed_dims=192,
        layer_depth=3,
        out_indices=[0, 1, 2, 3]
    ).cuda()
    print(f"params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    x = torch.randn(1, 3, 224, 224).cuda()
    out = model(x)
    print([o.shape for o in out])
