"""
RT-DETRv4: Painlessly Furthering Real-Time Object Detection with Vision Foundation Models
Copyright (c) 2025 The RT-DETRv4 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_activation
from .p2_dbf import P2DBFBridge

from ..core import register

import logging
_logger = logging.getLogger(__name__)

__all__ = ['HybridEncoder']


class ConvNormLayer_fuse(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, g=1, padding=None, bias=False, act=None):
        super().__init__()
        padding = (kernel_size-1)//2 if padding is None else padding
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            groups=g,
            padding=padding,
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)
        self.ch_in, self.ch_out, self.kernel_size, self.stride, self.g, self.padding, self.bias = \
            ch_in, ch_out, kernel_size, stride, g, padding, bias

    def forward(self, x):
        if hasattr(self, 'conv_bn_fused'):
            y = self.conv_bn_fused(x)
        else:
            y = self.norm(self.conv(x))
        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv_bn_fused'):
            self.conv_bn_fused = nn.Conv2d(
                self.ch_in,
                self.ch_out,
                self.kernel_size,
                self.stride,
                groups=self.g,
                padding=self.padding,
                bias=True)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv_bn_fused.weight.data = kernel
        self.conv_bn_fused.bias.data = bias
        self.__delattr__('conv')
        self.__delattr__('norm')

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor()

        return kernel3x3, bias3x3

    def _fuse_bn_tensor(self):
        kernel = self.conv.weight
        running_mean = self.norm.running_mean
        running_var = self.norm.running_var
        gamma = self.norm.weight
        beta = self.norm.bias
        eps = self.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, g=1, padding=None, bias=False, act=None):
        super().__init__()
        padding = (kernel_size-1)//2 if padding is None else padding
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            groups=g,
            padding=padding,
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


# TODO, add activation for cv1 following YOLOv10
# self.cv1 = Conv(c1, c2, 1, 1)
# self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)
class SCDown(nn.Module):
    def __init__(self, c1, c2, k, s, act=None):
        super().__init__()
        self.cv1 = ConvNormLayer_fuse(c1, c2, 1, 1)
        self.cv2 = ConvNormLayer_fuse(c2, c2, k, s, c2)

    def forward(self, x):
        return self.cv2(self.cv1(x))


class VGGBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu'):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)

        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        self.__delattr__('conv1')
        self.__delattr__('conv2')

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)

        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class CSPLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=False,
                 act="silu",
                 bottletype=VGGBlock):
        super(CSPLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer_fuse(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer_fuse(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[
            bottletype(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer_fuse(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_2 = self.conv2(x)
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        return self.conv3(x_1 + x_2)

class RepNCSPELAN4(nn.Module):
    # csp-elan
    def __init__(self, c1, c2, c3, c4, n=3,
                 bias=False,
                 act="silu"):
        super().__init__()
        self.c = c3//2
        self.cv1 = ConvNormLayer_fuse(c1, c3, 1, 1, bias=bias, act=act)
        self.cv2 = nn.Sequential(CSPLayer(c3//2, c4, n, 1, bias=bias, act=act, bottletype=VGGBlock), ConvNormLayer_fuse(c4, c4, 3, 1, bias=bias, act=act))
        self.cv3 = nn.Sequential(CSPLayer(c4, c4, n, 1, bias=bias, act=act, bottletype=VGGBlock), ConvNormLayer_fuse(c4, c4, 3, 1, bias=bias, act=act))
        self.cv4 = ConvNormLayer_fuse(c3+(2*c4), c2, 1, 1, bias=bias, act=act)

    def forward_chunk(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


# transformer
class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False):
        super().__init__()
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output


class SELite(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        channels = int(channels)
        reduction = max(1, int(reduction))
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = F.adaptive_avg_pool2d(x, output_size=1)
        scale = F.silu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale


class ECALite(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5):
        super().__init__()
        kernel_size = int(kernel_size)
        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be > 0, got {kernel_size}")
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        scale = F.adaptive_avg_pool2d(x, output_size=1).reshape(b, 1, c)
        scale = self.conv(scale)
        scale = torch.sigmoid(scale).reshape(b, c, 1, 1)
        return x * scale


class ACAFDeformAttnLite(nn.Module):
    """
    Lightweight deformable-attention cross-scale refinement block.
    Input:  x_low, x_high in BCHW (same shape)
    Output: refined x_high in BCHW
    """

    def __init__(
        self,
        channels: int,
        *,
        reduction: int = 8,
        num_heads: int = 8,
        num_points: int = 4,
        channel_attn: str = "se",
        channel_attn_reduction: int = 16,
        offset_scale: float = 2.0,
        padding_mode: str = "zeros",
        align_corners: bool = True,
        act: str = "silu",
    ) -> None:
        super().__init__()
        channels = int(channels)
        reduction = max(1, int(reduction))
        num_heads = int(num_heads)
        num_points = int(num_points)
        if channels <= 0:
            raise ValueError(f"channels must be > 0, got {channels}")
        if num_heads <= 0 or channels % num_heads != 0:
            raise ValueError(f"num_heads must be > 0 and divide channels ({channels}), got {num_heads}")
        if num_points <= 0:
            raise ValueError(f"num_points must be > 0, got {num_points}")
        padding_mode = str(padding_mode).strip().lower()
        if padding_mode not in {"zeros", "border", "reflection"}:
            raise ValueError(f"Unsupported padding_mode={padding_mode}")

        self.channels = channels
        self.num_heads = num_heads
        self.num_points = num_points
        self.head_dim = channels // num_heads
        self.offset_scale = float(offset_scale)
        self.padding_mode = padding_mode
        self.align_corners = bool(align_corners)

        hidden = max(16, channels // reduction)
        self.cond_proj = nn.Sequential(
            ConvNormLayer_fuse(2 * channels, hidden, kernel_size=1, stride=1, act=act),
            ConvNormLayer_fuse(hidden, hidden, kernel_size=3, stride=1, g=hidden, act=act),
        )
        self.offset_head = nn.Conv2d(hidden, 2 * num_heads * num_points, kernel_size=3, stride=1, padding=1, bias=True)
        self.attn_head = nn.Conv2d(hidden, num_heads * num_points, kernel_size=3, stride=1, padding=1, bias=True)

        # Residual delta for stability: output starts as identity when res_scale=0.
        self.delta_proj = ConvNormLayer_fuse(2 * channels, channels, kernel_size=1, stride=1, act=act)
        self.res_scale = nn.Parameter(torch.zeros(1, dtype=torch.float32))

        attn_mode = str(channel_attn).strip().lower()
        if attn_mode in {"", "none", "off", "disable", "disabled"}:
            self.channel_attn = nn.Identity()
        elif attn_mode in {"se", "selite", "se_lite"}:
            self.channel_attn = SELite(channels, reduction=channel_attn_reduction)
        elif attn_mode in {"eca", "ecalite", "eca_lite"}:
            self.channel_attn = ECALite(channels, kernel_size=5)
        else:
            raise ValueError(f"Unsupported channel_attn={channel_attn}")

        self._init_identity()

    def _init_identity(self):
        # Keep deformable branch close to interpolation at startup.
        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)
        nn.init.zeros_(self.attn_head.weight)
        nn.init.zeros_(self.attn_head.bias)

    @staticmethod
    def _build_base_grid(h: int, w: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, int(h), dtype=dtype, device=device),
            torch.linspace(-1.0, 1.0, int(w), dtype=dtype, device=device),
            indexing="ij",
        )
        return torch.stack((xx, yy), dim=-1).unsqueeze(0)  # [1, H, W, 2], (x, y)

    def forward(self, x_low: torch.Tensor, x_high: torch.Tensor) -> torch.Tensor:
        if x_low.shape != x_high.shape:
            raise ValueError(f"ACAFDeformAttnLite expects same shapes, got low={x_low.shape} high={x_high.shape}")
        if x_low.ndim != 4:
            raise ValueError(f"ACAFDeformAttnLite expects BCHW tensors, got {x_low.shape}")
        b, c, h, w = x_low.shape
        if c != self.channels:
            raise ValueError(f"ACAFDeformAttnLite expects C={self.channels}, got {c}")

        cond = self.cond_proj(torch.cat([x_low, x_high], dim=1))
        offsets = self.offset_head(cond).reshape(b, self.num_heads, self.num_points, 2, h, w)
        attn = self.attn_head(cond).reshape(b, self.num_heads, self.num_points, h, w)
        attn = torch.softmax(attn, dim=2)

        x_group = x_high.reshape(b, self.num_heads, self.head_dim, h, w)
        x_group = x_group.reshape(b * self.num_heads, self.head_dim, h, w)
        base_grid = self._build_base_grid(h, w, device=x_group.device, dtype=x_group.dtype)

        if self.align_corners:
            norm_x = 2.0 / max(w - 1, 1)
            norm_y = 2.0 / max(h - 1, 1)
        else:
            norm_x = 2.0 / max(w, 1)
            norm_y = 2.0 / max(h, 1)

        agg = x_group.new_zeros((b * self.num_heads, self.head_dim, h, w))
        for point_i in range(self.num_points):
            off = torch.tanh(offsets[:, :, point_i])  # [B, HN, 2, H, W]
            off_x = off[:, :, 0] * (self.offset_scale * norm_x)
            off_y = off[:, :, 1] * (self.offset_scale * norm_y)
            grid = torch.stack((off_x, off_y), dim=-1).reshape(b * self.num_heads, h, w, 2)
            grid = base_grid + grid

            sampled = F.grid_sample(
                x_group,
                grid,
                mode="bilinear",
                padding_mode=self.padding_mode,
                align_corners=self.align_corners,
            )
            weight = attn[:, :, point_i].reshape(b * self.num_heads, 1, h, w)
            agg = agg + sampled * weight

        deform_feat = agg.reshape(b, c, h, w)
        deform_feat = self.channel_attn(deform_feat)

        delta = self.delta_proj(torch.cat([x_low, deform_feat], dim=1))
        return x_high + self.res_scale * delta


# Backward-compatible alias.
DCCFFDeformAttnLite = ACAFDeformAttnLite


@register()
class HybridEncoder(nn.Module):
    __share__ = ['eval_spatial_size', ]

    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 nhead=8,
                 dim_feedforward = 1024,
                 dropout=0.0,
                 enc_act='gelu',
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 eval_spatial_size=None,
                 version='dfine',
                 distill_teacher_dim=0,
                 acaf_enabled=None,
                 acaf_apply_topdown=None,
                 acaf_apply_bottomup=None,
                 acaf_reduction=None,
                 acaf_num_heads=None,
                 acaf_num_points=None,
                 acaf_channel_attn=None,
                 acaf_channel_attn_reduction=None,
                 acaf_offset_scale=None,
                 acaf_padding_mode=None,
                 acaf_align_corners=None,
                 acaf_act=None,
                 dccff_enabled=False,
                 dccff_apply_topdown=True,
                 dccff_apply_bottomup=False,
                 dccff_reduction=8,
                 dccff_num_heads=8,
                 dccff_num_points=4,
                 dccff_channel_attn='se',
                 dccff_channel_attn_reduction=16,
                 dccff_offset_scale=2.0,
                 dccff_padding_mode='zeros',
                 dccff_align_corners=True,
                 dccff_act='silu',
                 p2_dbf_enabled=False,
                 p2_dbf_use_internal_only=True,
                 p2_dbf_from_idx=0,
                 p2_dbf_to_idx=1,
                 p2_dbf_spd_factor=2,
                 p2_dbf_res_scale_init=0.1,
                 mkb_enabled=True,
                 mkb_kernel_size=7,
                 mkb_use_global_branch=True,
                 mkb_expand_ratio=0.5,
                 ):
        super().__init__()
        self.in_channels = list(in_channels)
        self.feat_strides = list(feat_strides)
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = list(use_encoder_idx)
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size
        self.distill_teacher_dim = distill_teacher_dim

        # Canonical ACAF params with backward-compatible DCCFF fallbacks.
        acaf_enabled = dccff_enabled if acaf_enabled is None else acaf_enabled
        acaf_apply_topdown = dccff_apply_topdown if acaf_apply_topdown is None else acaf_apply_topdown
        acaf_apply_bottomup = dccff_apply_bottomup if acaf_apply_bottomup is None else acaf_apply_bottomup
        acaf_reduction = dccff_reduction if acaf_reduction is None else acaf_reduction
        acaf_num_heads = dccff_num_heads if acaf_num_heads is None else acaf_num_heads
        acaf_num_points = dccff_num_points if acaf_num_points is None else acaf_num_points
        acaf_channel_attn = dccff_channel_attn if acaf_channel_attn is None else acaf_channel_attn
        acaf_channel_attn_reduction = dccff_channel_attn_reduction if acaf_channel_attn_reduction is None else acaf_channel_attn_reduction
        acaf_offset_scale = dccff_offset_scale if acaf_offset_scale is None else acaf_offset_scale
        acaf_padding_mode = dccff_padding_mode if acaf_padding_mode is None else acaf_padding_mode
        acaf_align_corners = dccff_align_corners if acaf_align_corners is None else acaf_align_corners
        acaf_act = dccff_act if acaf_act is None else acaf_act

        self.acaf_enabled = bool(acaf_enabled)
        self.acaf_apply_topdown = self.acaf_enabled and bool(acaf_apply_topdown)
        self.acaf_apply_bottomup = self.acaf_enabled and bool(acaf_apply_bottomup)

        # Backward-compatible field names used by existing configs/logics.
        self.dccff_enabled = self.acaf_enabled
        self.dccff_apply_topdown = self.acaf_apply_topdown
        self.dccff_apply_bottomup = self.acaf_apply_bottomup

        self.p2_dbf_enabled = bool(p2_dbf_enabled)
        self.p2_dbf_use_internal_only = bool(p2_dbf_use_internal_only)
        self.p2_dbf_from_idx = int(p2_dbf_from_idx)
        self.p2_dbf_to_idx = int(p2_dbf_to_idx)
        self.p2_dbf_drop_lowest_output = False

        out_channels = [hidden_dim for _ in range(len(self.in_channels))]
        out_strides = list(self.feat_strides)
        if self.p2_dbf_enabled and self.p2_dbf_use_internal_only and len(out_channels) >= 4 and self.p2_dbf_from_idx == 0 and self.p2_dbf_to_idx == 1:
            # Internal P2 bridge keeps decoder interface at 3 levels (P3-P5).
            self.p2_dbf_drop_lowest_output = True
            out_channels = out_channels[1:]
            out_strides = out_strides[1:]
        self.out_channels = out_channels
        self.out_strides = out_strides

        assert len(self.use_encoder_idx) > 0, "use_encoder_idx must specify at least one encoder output"
        # target AIFI output F5 for distillation
        self.encoder_idx_for_distillation = self.use_encoder_idx[-1]

        # channel projection
        self.input_proj = nn.ModuleList()
        for in_channel in self.in_channels:
            proj = nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False)),
                    ('norm', nn.BatchNorm2d(hidden_dim))
                ]))

            self.input_proj.append(proj)

        # encoder transformer
        encoder_layer = TransformerEncoderLayer(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=enc_act
            )

        self.encoder = nn.ModuleList([
            TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers) for _ in range(len(self.use_encoder_idx))
        ])

        # feature_projector
        self.feature_projector = None
        if self.distill_teacher_dim > 0:
            self.feature_projector = nn.Sequential(
                    nn.Linear(hidden_dim, self.distill_teacher_dim),
                    # nn.GELU(),
                )

        self.p2_dbf_bridge = None
        if self.p2_dbf_enabled:
            if not (0 <= self.p2_dbf_from_idx < len(self.in_channels) and 0 <= self.p2_dbf_to_idx < len(self.in_channels)):
                _logger.warning(
                    "P2-DBF disabled due to invalid indices: from_idx=%s to_idx=%s num_levels=%s",
                    self.p2_dbf_from_idx,
                    self.p2_dbf_to_idx,
                    len(self.in_channels),
                )
                self.p2_dbf_enabled = False
            else:
                self.p2_dbf_bridge = P2DBFBridge(
                    hidden_dim,
                    spd_factor=p2_dbf_spd_factor,
                    use_mkb=mkb_enabled,
                    mkb_kernel_size=mkb_kernel_size,
                    mkb_use_global_branch=mkb_use_global_branch,
                    mkb_expand_ratio=mkb_expand_ratio,
                    res_scale_init=p2_dbf_res_scale_init,
                    act=act,
                )

        # top-down fpn
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(self.in_channels) - 1, 0, -1):
            # TODO, add activation for those lateral convs
            if version == 'dfine':
                self.lateral_convs.append(ConvNormLayer_fuse(hidden_dim, hidden_dim, 1, 1))
            else:
                self.lateral_convs.append(ConvNormLayer_fuse(hidden_dim, hidden_dim, 1, 1, act=act))
            self.fpn_blocks.append(
                RepNCSPELAN4(hidden_dim * 2, hidden_dim, hidden_dim * 2, round(expansion * hidden_dim // 2), round(3 * depth_mult), act=act) \
                if version == 'dfine' else CSPLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion, bottletype=VGGBlock)
            )

        # bottom-up pan
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(self.in_channels) - 1):
            self.downsample_convs.append(
                nn.Sequential(SCDown(hidden_dim, hidden_dim, 3, 2, act=act)) \
                if version == 'dfine' else ConvNormLayer_fuse(hidden_dim, hidden_dim, 3, 2, act=act)
            )
            self.pan_blocks.append(
                RepNCSPELAN4(hidden_dim * 2, hidden_dim, hidden_dim * 2, round(expansion * hidden_dim // 2), round(3 * depth_mult), act=act) \
                if version == 'dfine' else CSPLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion, bottletype=VGGBlock)
            )

        self.topdown_acaf = nn.ModuleList()
        if self.acaf_apply_topdown:
            for _ in range(len(self.in_channels) - 1):
                self.topdown_acaf.append(
                    ACAFDeformAttnLite(
                        hidden_dim,
                        reduction=acaf_reduction,
                        num_heads=acaf_num_heads,
                        num_points=acaf_num_points,
                        channel_attn=acaf_channel_attn,
                        channel_attn_reduction=acaf_channel_attn_reduction,
                        offset_scale=acaf_offset_scale,
                        padding_mode=acaf_padding_mode,
                        align_corners=acaf_align_corners,
                        act=acaf_act,
                    )
                )

        self.bottomup_acaf = nn.ModuleList()
        if self.acaf_apply_bottomup:
            for _ in range(len(self.in_channels) - 1):
                self.bottomup_acaf.append(
                    ACAFDeformAttnLite(
                        hidden_dim,
                        reduction=acaf_reduction,
                        num_heads=acaf_num_heads,
                        num_points=acaf_num_points,
                        channel_attn=acaf_channel_attn,
                        channel_attn_reduction=acaf_channel_attn_reduction,
                        offset_scale=acaf_offset_scale,
                        padding_mode=acaf_padding_mode,
                        align_corners=acaf_align_corners,
                        act=acaf_act,
                    )
                )

        # Backward-compatible aliases for existing references.
        self.topdown_dccff = self.topdown_acaf
        self.bottomup_dccff = self.bottomup_acaf

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride, self.eval_spatial_size[0] // stride,
                    self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)
                # self.register_buffer(f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        """
        """
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def forward(self, feats):
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        if self.p2_dbf_bridge is not None:
            src_idx = self.p2_dbf_from_idx
            dst_idx = self.p2_dbf_to_idx
            proj_feats[dst_idx] = self.p2_dbf_bridge(proj_feats[src_idx], proj_feats[dst_idx])

        distill_student_output = None

        # encoder
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                # flatten [B, C, H, W] to [B, HxW, C]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)

                memory :torch.Tensor = self.encoder[i](src_flatten, src_mask=None, pos_embed=pos_embed)

                # Reshape back to [B, C, H, W] for subsequent FPN/PAN layers
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()

                # Apply feature projector to F5
                if self.training and self.feature_projector is not None and enc_ind == self.encoder_idx_for_distillation:
                    # _logger.info(f"[HybridEncoder] Feature size: {h}x{w}")
                    distill_student_output = self.feature_projector(proj_feats[enc_ind].permute(0, 2, 3, 1)).permute(0, 3, 1, 2) # [B, distill_teacher_dim, H, W]


        # broadcasting and fusion
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            block_idx = len(self.in_channels) - 1 - idx
            feat_heigh = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_heigh = self.lateral_convs[block_idx](feat_heigh)
            inner_outs[0] = feat_heigh
            upsample_feat = F.interpolate(feat_heigh, scale_factor=2., mode='nearest')
            if self.acaf_apply_topdown:
                upsample_feat = self.topdown_acaf[block_idx](feat_low, upsample_feat)
            inner_out = self.fpn_blocks[block_idx](torch.concat([upsample_feat, feat_low], dim=1))
            inner_outs.insert(0, inner_out)

        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            if self.acaf_apply_bottomup:
                feat_height = self.bottomup_acaf[idx](downsample_feat, feat_height)
            out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_height], dim=1))
            outs.append(out)

        if self.p2_dbf_drop_lowest_output:
            outs = outs[1:]

        if self.training and distill_student_output is not None:
            return outs, distill_student_output
        return outs
