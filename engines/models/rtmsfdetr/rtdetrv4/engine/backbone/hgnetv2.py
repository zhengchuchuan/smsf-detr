"""
reference
- https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py

Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import os
from collections.abc import Iterable
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import FrozenBatchNorm2d
from .deform_align import DeformableAlign2D
from .eemsa import EEMSA
from .group_deform_align import GroupwiseDeformableAlign2D, ProjectedGroupwiseDeformableAlign2D
from .ms_band_sep import MSBandSeparatedStemAlign
from .stem_cf_interactive import StemCFInteractive2D
from ..core import register
import logging

# Constants for initialization
kaiming_normal_ = nn.init.kaiming_normal_
zeros_ = nn.init.zeros_
ones_ = nn.init.ones_

__all__ = ['HGNetv2']


def _parse_eemsa_locations(value: Any) -> set[str]:
    if value is None:
        return {"stage1", "stage2", "stage3", "stage4"}
    if isinstance(value, str):
        items = [s.strip() for s in value.replace(";", ",").split(",") if s.strip()]
    elif isinstance(value, (list, tuple, set)):
        items = [str(s).strip() for s in value]
    elif isinstance(value, Iterable) and not isinstance(value, (str, bytes, Mapping)):
        items = [str(s).strip() for s in value]
    else:
        items = [str(value).strip()]

    out: set[str] = set()
    for item in items:
        key = item.lower()
        if key in {"stem", "after_stem", "post_stem"}:
            out.add("stem")
        elif key in {"stage1", "stage_1", "c2", "after_stage1"}:
            out.add("stage1")
        elif key in {"stage2", "stage_2", "c3", "after_stage2"}:
            out.add("stage2")
        elif key in {"stage3", "stage_3", "c4", "after_stage3"}:
            out.add("stage3")
        elif key in {"stage4", "stage_4", "c5", "after_stage4"}:
            out.add("stage4")
        else:
            raise ValueError(
                f"Unsupported EEMSA location: {item}. "
                "Expected one of stem/stage1/stage2/stage3/stage4 "
                "(aliases: c2/c3/c4/c5)."
            )
    return out


def _cfg_value(cfg: Mapping[str, Any], key: str, default: Any) -> Any:
    value = cfg.get(key, default)
    return default if value is None else value


def _normalize_stage_idx(key: Any) -> int:
    if isinstance(key, str):
        kk = key.strip().lower()
        if kk in {"c2", "p2", "stage1", "stage_1"}:
            return 0
        if kk in {"c3", "p3", "stage2", "stage_2"}:
            return 1
        if kk in {"c4", "p4", "stage3", "stage_3"}:
            return 2
        if kk in {"c5", "p5", "stage4", "stage_4"}:
            return 3
        return int(kk)
    return int(key)


class LearnableAffineBlock(nn.Module):
    def __init__(
            self,
            scale_value=1.0,
            bias_value=0.0
    ):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([scale_value]), requires_grad=True)
        self.bias = nn.Parameter(torch.tensor([bias_value]), requires_grad=True)

    def forward(self, x):
        return self.scale * x + self.bias


class ConvBNAct(nn.Module):
    def __init__(
            self,
            in_chs,
            out_chs,
            kernel_size,
            stride=1,
            groups=1,
            padding='',
            use_act=True,
            use_lab=False
    ):
        super().__init__()
        self.use_act = use_act
        self.use_lab = use_lab
        if padding == 'same':
            self.conv = nn.Sequential(
                nn.ZeroPad2d([0, 1, 0, 1]),
                nn.Conv2d(
                    in_chs,
                    out_chs,
                    kernel_size,
                    stride,
                    groups=groups,
                    bias=False
                )
            )
        else:
            self.conv = nn.Conv2d(
                in_chs,
                out_chs,
                kernel_size,
                stride,
                padding=(kernel_size - 1) // 2,
                groups=groups,
                bias=False
            )
        self.bn = nn.BatchNorm2d(out_chs)
        if self.use_act:
            self.act = nn.ReLU()
        else:
            self.act = nn.Identity()
        if self.use_act and self.use_lab:
            self.lab = LearnableAffineBlock()
        else:
            self.lab = nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.lab(x)
        return x


class LightConvBNAct(nn.Module):
    def __init__(
            self,
            in_chs,
            out_chs,
            kernel_size,
            groups=1,
            use_lab=False,
    ):
        super().__init__()
        self.conv1 = ConvBNAct(
            in_chs,
            out_chs,
            kernel_size=1,
            use_act=False,
            use_lab=use_lab,
        )
        self.conv2 = ConvBNAct(
            out_chs,
            out_chs,
            kernel_size=kernel_size,
            groups=out_chs,
            use_act=True,
            use_lab=use_lab,
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class StemBlock(nn.Module):
    # for HGNetv2
    def __init__(self, in_chs, mid_chs, out_chs, use_lab=False):
        super().__init__()
        self.stem1 = ConvBNAct(
            in_chs,
            mid_chs,
            kernel_size=3,
            stride=2,
            use_lab=use_lab,
        )
        self.stem2a = ConvBNAct(
            mid_chs,
            mid_chs // 2,
            kernel_size=2,
            stride=1,
            use_lab=use_lab,
        )
        self.stem2b = ConvBNAct(
            mid_chs // 2,
            mid_chs,
            kernel_size=2,
            stride=1,
            use_lab=use_lab,
        )
        self.stem3 = ConvBNAct(
            mid_chs * 2,
            mid_chs,
            kernel_size=3,
            stride=2,
            use_lab=use_lab,
        )
        self.stem4 = ConvBNAct(
            mid_chs,
            out_chs,
            kernel_size=1,
            stride=1,
            use_lab=use_lab,
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, ceil_mode=True)

    def forward(self, x):
        x = self.stem1(x)
        x = F.pad(x, (0, 1, 0, 1))
        x2 = self.stem2a(x)
        x2 = F.pad(x2, (0, 1, 0, 1))
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class EseModule(nn.Module):
    def __init__(self, chs):
        super().__init__()
        self.conv = nn.Conv2d(
            chs,
            chs,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        x = x.mean((2, 3), keepdim=True)
        x = self.conv(x)
        x = self.sigmoid(x)
        return torch.mul(identity, x)


class HG_Block(nn.Module):
    def __init__(
            self,
            in_chs,
            mid_chs,
            out_chs,
            layer_num,
            kernel_size=3,
            residual=False,
            light_block=False,
            use_lab=False,
            agg='ese',
            drop_path=0.,
    ):
        super().__init__()
        self.residual = residual

        self.layers = nn.ModuleList()
        for i in range(layer_num):
            if light_block:
                self.layers.append(
                    LightConvBNAct(
                        in_chs if i == 0 else mid_chs,
                        mid_chs,
                        kernel_size=kernel_size,
                        use_lab=use_lab,
                    )
                )
            else:
                self.layers.append(
                    ConvBNAct(
                        in_chs if i == 0 else mid_chs,
                        mid_chs,
                        kernel_size=kernel_size,
                        stride=1,
                        use_lab=use_lab,
                    )
                )

        # feature aggregation
        total_chs = in_chs + layer_num * mid_chs
        if agg == 'se':
            aggregation_squeeze_conv = ConvBNAct(
                total_chs,
                out_chs // 2,
                kernel_size=1,
                stride=1,
                use_lab=use_lab,
            )
            aggregation_excitation_conv = ConvBNAct(
                out_chs // 2,
                out_chs,
                kernel_size=1,
                stride=1,
                use_lab=use_lab,
            )
            self.aggregation = nn.Sequential(
                aggregation_squeeze_conv,
                aggregation_excitation_conv,
            )
        else:
            aggregation_conv = ConvBNAct(
                total_chs,
                out_chs,
                kernel_size=1,
                stride=1,
                use_lab=use_lab,
            )
            att = EseModule(out_chs)
            self.aggregation = nn.Sequential(
                aggregation_conv,
                att,
            )

        self.drop_path = nn.Dropout(drop_path) if drop_path else nn.Identity()

    def forward(self, x):
        identity = x
        output = [x]
        for layer in self.layers:
            x = layer(x)
            output.append(x)
        x = torch.cat(output, dim=1)
        x = self.aggregation(x)
        if self.residual:
            x = self.drop_path(x) + identity
        return x


class HG_Stage(nn.Module):
    def __init__(
            self,
            in_chs,
            mid_chs,
            out_chs,
            block_num,
            layer_num,
            downsample=True,
            light_block=False,
            kernel_size=3,
            use_lab=False,
            agg='se',
            drop_path=0.,
    ):
        super().__init__()
        self.downsample = downsample
        if downsample:
            self.downsample = ConvBNAct(
                in_chs,
                in_chs,
                kernel_size=3,
                stride=2,
                groups=in_chs,
                use_act=False,
                use_lab=use_lab,
            )
        else:
            self.downsample = nn.Identity()

        blocks_list = []
        for i in range(block_num):
            blocks_list.append(
                HG_Block(
                    in_chs if i == 0 else out_chs,
                    mid_chs,
                    out_chs,
                    layer_num,
                    residual=False if i == 0 else True,
                    kernel_size=kernel_size,
                    light_block=light_block,
                    use_lab=use_lab,
                    agg=agg,
                    drop_path=drop_path[i] if isinstance(drop_path, (list, tuple)) else drop_path,
                )
            )
        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x):
        x = self.downsample(x)
        x = self.blocks(x)
        return x



@register()
class HGNetv2(nn.Module):
    """
    HGNetV2
    Args:
        stem_channels: list. Number of channels for the stem block.
        stage_type: str. The stage configuration of HGNet. such as the number of channels, stride, etc.
        use_lab: boolean. Whether to use LearnableAffineBlock in network.
        lr_mult_list: list. Control the learning rate of different stages.
    Returns:
        model: nn.Layer. Specific HGNetV2 model depends on args.
    """

    arch_configs = {
        'B0': {
            'stem_channels': [3, 16, 16],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [16, 16, 64, 1, False, False, 3, 3],
                "stage2": [64, 32, 256, 1, True, False, 3, 3],
                "stage3": [256, 64, 512, 2, True, True, 5, 3],
                "stage4": [512, 128, 1024, 1, True, True, 5, 3],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B0_stage1.pth'
        },
        'B1': {
            'stem_channels': [3, 24, 32],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [32, 32, 64, 1, False, False, 3, 3],
                "stage2": [64, 48, 256, 1, True, False, 3, 3],
                "stage3": [256, 96, 512, 2, True, True, 5, 3],
                "stage4": [512, 192, 1024, 1, True, True, 5, 3],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B1_stage1.pth'
        },
        'B2': {
            'stem_channels': [3, 24, 32],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [32, 32, 96, 1, False, False, 3, 4],
                "stage2": [96, 64, 384, 1, True, False, 3, 4],
                "stage3": [384, 128, 768, 3, True, True, 5, 4],
                "stage4": [768, 256, 1536, 1, True, True, 5, 4],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B2_stage1.pth'
        },
        'B3': {
            'stem_channels': [3, 24, 32],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [32, 32, 128, 1, False, False, 3, 5],
                "stage2": [128, 64, 512, 1, True, False, 3, 5],
                "stage3": [512, 128, 1024, 3, True, True, 5, 5],
                "stage4": [1024, 256, 2048, 1, True, True, 5, 5],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B3_stage1.pth'
        },
        'B4': {
            'stem_channels': [3, 32, 48],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [48, 48, 128, 1, False, False, 3, 6],
                "stage2": [128, 96, 512, 1, True, False, 3, 6],
                "stage3": [512, 192, 1024, 3, True, True, 5, 6],
                "stage4": [1024, 384, 2048, 1, True, True, 5, 6],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B4_stage1.pth'
        },
        'B5': {
            'stem_channels': [3, 32, 64],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [64, 64, 128, 1, False, False, 3, 6],
                "stage2": [128, 128, 512, 2, True, False, 3, 6],
                "stage3": [512, 256, 1024, 5, True, True, 5, 6],
                "stage4": [1024, 512, 2048, 2, True, True, 5, 6],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B5_stage1.pth'
        },
        'B6': {
            'stem_channels': [3, 48, 96],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [96, 96, 192, 2, False, False, 3, 6],
                "stage2": [192, 192, 512, 3, True, False, 3, 6],
                "stage3": [512, 384, 1024, 6, True, True, 5, 6],
                "stage4": [1024, 768, 2048, 3, True, True, 5, 6],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B6_stage1.pth'
        },
    }

    def __init__(self,
                 name,
                 in_chs: int = 3,
                 input_channels: int | None = None,
                 use_lab=False,
                 return_idx=[1, 2, 3],
                 freeze_stem_only=True,
                 freeze_at=0,
                 freeze_norm=True,
                 pretrained=True,
                 local_model_dir='weight/hgnetv2/',
                 eemsa: Mapping[str, Any] | None = None,
                 ms_band_sep: Mapping[str, Any] | None = None,
                 ms_residual_stem: Mapping[str, Any] | None = None,
                 ms_group_align: Mapping[str, Any] | None = None):
        super().__init__()
        self.use_lab = use_lab
        self.return_idx = return_idx

        if input_channels is None:
            input_channels = in_chs
        if input_channels is None:
            input_channels = 3
        input_channels = int(input_channels)

        stem_channels = list(self.arch_configs[name]['stem_channels'])
        stem_channels[0] = input_channels
        stage_config = self.arch_configs[name]['stage_config']
        stage_keys = list(stage_config.keys())
        stage_in_channels = [int(stage_config[k][0]) for k in stage_keys]
        stage_out_channels = [int(stage_config[k][2]) for k in stage_keys]
        download_url = self.arch_configs[name]['url']

        self._out_strides = [4, 8, 16, 32]
        self._out_channels = [stage_config[k][2] for k in stage_config]
        self.ms_in_chs = input_channels

        # stem
        self.stem = StemBlock(
                in_chs=stem_channels[0],
                mid_chs=stem_channels[1],
                out_chs=stem_channels[2],
                use_lab=use_lab)

        # Optional MSI-only band-separated stem + CRGGA alignment for strict single-stream runs.
        ms_band_sep_cfg: dict[str, Any] = {}
        if ms_band_sep is not None:
            if isinstance(ms_band_sep, Mapping):
                ms_band_sep_cfg = dict(ms_band_sep)
            elif hasattr(ms_band_sep, "items"):
                ms_band_sep_cfg = {k: v for k, v in ms_band_sep.items()}  # type: ignore[assignment]
            else:
                raise TypeError(f"Unsupported ms_band_sep type: {type(ms_band_sep)}")
        self.ms_band_sep_cfg = ms_band_sep_cfg
        self.ms_band_sep_enabled = bool(ms_band_sep_cfg.get("enabled", ms_band_sep_cfg.get("enable", False)))
        self.ms_band_sep_stem: MSBandSeparatedStemAlign | None = None
        if self.ms_band_sep_enabled:
            c2_in_channels = int(next(iter(stage_config.values()))[0])
            embed_channels = int(ms_band_sep_cfg.get("embed_channels", ms_band_sep_cfg.get("c_emb", 16)) or 16)
            embed_use_bn = bool(ms_band_sep_cfg.get("embed_use_bn", True))
            extractor_type = str(ms_band_sep_cfg.get("extractor_type", ms_band_sep_cfg.get("stem_type", "light")) or "light")
            stem_norm_type = str(ms_band_sep_cfg.get("stem_norm_type", ms_band_sep_cfg.get("branch_norm", "gn")) or "gn")
            align_cfg_raw = ms_band_sep_cfg.get("align", ms_band_sep_cfg.get("align_cfg", None))
            align_cfg: dict[str, Any] | None = None
            if isinstance(align_cfg_raw, Mapping):
                align_cfg = dict(align_cfg_raw)
            elif align_cfg_raw is not None and hasattr(align_cfg_raw, "items"):
                align_cfg = {k: v for k, v in align_cfg_raw.items()}  # type: ignore[assignment]
            self.ms_band_sep_stem = MSBandSeparatedStemAlign(
                ms_in_chs=input_channels,
                c2_in_channels=c2_in_channels,
                embed_channels=embed_channels,
                embed_use_bn=embed_use_bn,
                extractor_type=extractor_type,
                stem_mid_channels=int(ms_band_sep_cfg.get("stem_mid_channels", stem_channels[1])),
                stem_out_channels=int(ms_band_sep_cfg.get("stem_out_channels", stem_channels[2])),
                stem_norm_type=stem_norm_type,
                align_cfg=align_cfg,
            )
            # The original stem is bypassed once ms_band_sep is enabled; freeze it to avoid DDP unused params.
            for p in self.stem.parameters():
                p.requires_grad_(False)

        ms_residual_stem_cfg: dict[str, Any] = {}
        if ms_residual_stem is not None:
            if isinstance(ms_residual_stem, Mapping):
                ms_residual_stem_cfg = dict(ms_residual_stem)
            elif hasattr(ms_residual_stem, "items"):
                ms_residual_stem_cfg = {k: v for k, v in ms_residual_stem.items()}  # type: ignore[assignment]
            else:
                raise TypeError(f"Unsupported ms_residual_stem type: {type(ms_residual_stem)}")
        self.ms_residual_stem_cfg = ms_residual_stem_cfg
        self.ms_residual_stem_enabled = bool(
            ms_residual_stem_cfg.get("enabled", ms_residual_stem_cfg.get("enable", False))
        )
        self.ms_residual_stem_branch: MSBandSeparatedStemAlign | None = None
        self.ms_residual_scale: nn.Parameter | None = None
        self.ms_residual_fusion_mode = "add"
        self.ms_residual_fuse_proj: nn.Module | None = None
        self.ms_residual_stem_interactive_enabled = False
        self.ms_residual_stem_interactive: StemCFInteractive2D | None = None
        self.ms_residual_post_align_enabled = False
        self.ms_residual_post_aligner: DeformableAlign2D | None = None
        self.ms_residual_post_align_ref_detach = True
        self.ms_residual_post_align_loss_weight = 0.0
        self.ms_residual_post_align_loss_offset_weight = 0.0
        self.ms_residual_post_align_loss_attn_norm_weight = 0.0
        self.ms_residual_post_align_loss_attn_entropy_weight = 0.0
        if self.ms_band_sep_enabled and self.ms_residual_stem_enabled:
            raise ValueError("ms_band_sep and ms_residual_stem cannot be enabled at the same time")
        if self.ms_residual_stem_enabled:
            c2_in_channels = int(next(iter(stage_config.values()))[0])
            embed_channels = int(ms_residual_stem_cfg.get("embed_channels", ms_residual_stem_cfg.get("c_emb", 16)) or 16)
            embed_use_bn = bool(ms_residual_stem_cfg.get("embed_use_bn", True))
            extractor_type = str(
                ms_residual_stem_cfg.get("extractor_type", ms_residual_stem_cfg.get("stem_type", "light")) or "light"
            )
            stem_norm_type = str(
                ms_residual_stem_cfg.get("stem_norm_type", ms_residual_stem_cfg.get("branch_norm", "gn")) or "gn"
            )
            merge_activation = str(ms_residual_stem_cfg.get("merge_activation", "identity") or "identity")
            align_cfg_raw = ms_residual_stem_cfg.get("align", ms_residual_stem_cfg.get("align_cfg", None))
            align_cfg: dict[str, Any] | None = None
            if isinstance(align_cfg_raw, Mapping):
                align_cfg = dict(align_cfg_raw)
            elif align_cfg_raw is not None and hasattr(align_cfg_raw, "items"):
                align_cfg = {k: v for k, v in align_cfg_raw.items()}  # type: ignore[assignment]
            self.ms_residual_stem_branch = MSBandSeparatedStemAlign(
                ms_in_chs=input_channels,
                c2_in_channels=c2_in_channels,
                embed_channels=embed_channels,
                embed_use_bn=embed_use_bn,
                extractor_type=extractor_type,
                stem_mid_channels=int(ms_residual_stem_cfg.get("stem_mid_channels", stem_channels[1])),
                stem_out_channels=int(ms_residual_stem_cfg.get("stem_out_channels", stem_channels[2])),
                stem_norm_type=stem_norm_type,
                merge_activation=merge_activation,
                align_cfg=align_cfg,
            )
            fusion_mode = str(ms_residual_stem_cfg.get("fusion_mode", "add") or "add").strip().lower()
            if fusion_mode in {"add", "sum", "residual_add"}:
                fusion_mode = "add"
            elif fusion_mode in {"concat", "concat_proj", "concat_residual", "concat_residual_proj"}:
                fusion_mode = "concat_proj"
            else:
                raise ValueError(
                    f"Unsupported ms_residual_stem.fusion_mode={fusion_mode} "
                    "(supported: add|concat_proj)"
                )
            self.ms_residual_fusion_mode = fusion_mode
            if self.ms_residual_fusion_mode == "concat_proj":
                self.ms_residual_fuse_proj = nn.Conv2d(c2_in_channels * 2, c2_in_channels, kernel_size=1, bias=True)
                nn.init.constant_(self.ms_residual_fuse_proj.weight, 0.0)
                if self.ms_residual_fuse_proj.bias is not None:
                    nn.init.constant_(self.ms_residual_fuse_proj.bias, 0.0)

            stem_interactive_cfg_raw = ms_residual_stem_cfg.get(
                "stem_interactive",
                ms_residual_stem_cfg.get("cf_interactive", ms_residual_stem_cfg.get("main_interactive", None)),
            )
            stem_interactive_cfg: dict[str, Any] = {}
            if isinstance(stem_interactive_cfg_raw, Mapping):
                stem_interactive_cfg = dict(stem_interactive_cfg_raw)
            elif stem_interactive_cfg_raw is not None and hasattr(stem_interactive_cfg_raw, "items"):
                stem_interactive_cfg = {k: v for k, v in stem_interactive_cfg_raw.items()}  # type: ignore[assignment]
            self.ms_residual_stem_interactive_enabled = bool(
                stem_interactive_cfg.get("enabled", stem_interactive_cfg.get("enable", False))
            )
            if self.ms_residual_stem_interactive_enabled:
                self.ms_residual_stem_interactive = StemCFInteractive2D(
                    in_channels=c2_in_channels,
                    num_heads=int(_cfg_value(stem_interactive_cfg, "num_heads", 4)),
                    num_points=int(_cfg_value(stem_interactive_cfg, "num_points", 4)),
                    memory_detach=bool(
                        stem_interactive_cfg.get(
                            "memory_detach",
                            stem_interactive_cfg.get("detach_memory", True),
                        )
                    ),
                    ref_shift_enabled=bool(
                        stem_interactive_cfg.get(
                            "ref_shift_enabled",
                            stem_interactive_cfg.get("support_ref_shift_enabled", True),
                        )
                    ),
                    ref_shift_scale=float(
                        _cfg_value(
                            stem_interactive_cfg,
                            "ref_shift_scale",
                            _cfg_value(stem_interactive_cfg, "support_ref_shift_scale", 0.02),
                        )
                    ),
                    delta_hidden_channels=int(
                        _cfg_value(stem_interactive_cfg, "delta_hidden_channels", c2_in_channels)
                    ),
                    scale_init=float(_cfg_value(stem_interactive_cfg, "scale_init", 0.01)),
                    scale_per_channel=bool(stem_interactive_cfg.get("scale_per_channel", True)),
                )

            post_align_cfg_raw = ms_residual_stem_cfg.get(
                "post_align",
                ms_residual_stem_cfg.get("align_to_main", ms_residual_stem_cfg.get("fuse_align", None)),
            )
            post_align_cfg: dict[str, Any] = {}
            if isinstance(post_align_cfg_raw, Mapping):
                post_align_cfg = dict(post_align_cfg_raw)
            elif post_align_cfg_raw is not None and hasattr(post_align_cfg_raw, "items"):
                post_align_cfg = {k: v for k, v in post_align_cfg_raw.items()}  # type: ignore[assignment]
            self.ms_residual_post_align_enabled = bool(
                post_align_cfg.get("enabled", post_align_cfg.get("enable", False))
            )
            if self.ms_residual_post_align_enabled:
                self.ms_residual_post_align_ref_detach = bool(
                    post_align_cfg.get("ref_detach", post_align_cfg.get("detach_ref", True))
                )
                self.ms_residual_post_align_loss_weight = float(post_align_cfg.get("loss_weight", 0.02))
                self.ms_residual_post_align_loss_offset_weight = float(
                    post_align_cfg.get("loss_offset_weight", post_align_cfg.get("offset_loss_weight", 0.0))
                )
                self.ms_residual_post_align_loss_attn_norm_weight = float(
                    post_align_cfg.get("loss_attn_norm_weight", post_align_cfg.get("attn_reg_weight", 0.0))
                )
                self.ms_residual_post_align_loss_attn_entropy_weight = float(
                    post_align_cfg.get("loss_attn_entropy_weight", post_align_cfg.get("attn_entropy_reg_weight", 0.0))
                )
                self.ms_residual_post_aligner = DeformableAlign2D(
                    in_channels=c2_in_channels,
                    num_keypoints=int(_cfg_value(post_align_cfg, "num_keypoints", 9)),
                    offset_scale=float(_cfg_value(post_align_cfg, "offset_scale", 3.0)),
                    offset_enabled=bool(post_align_cfg.get("offset_enabled", True)),
                    per_channel_offset=False,
                    attention_norm=str(_cfg_value(post_align_cfg, "attention_norm", "softmax")),
                    padding_mode=str(_cfg_value(post_align_cfg, "padding_mode", "border")),
                    align_corners=bool(post_align_cfg.get("align_corners", True)),
                    loss_type=str(_cfg_value(post_align_cfg, "loss_type", "infonce")),
                    loss_downsample=post_align_cfg.get("loss_downsample", 0.5),
                    nce_num_patches=int(_cfg_value(post_align_cfg, "nce_num_patches", 64)),
                    nce_patch_size=int(_cfg_value(post_align_cfg, "nce_patch_size", 5)),
                    nce_tau=float(_cfg_value(post_align_cfg, "nce_tau", 0.2)),
                    affine_enabled=bool(post_align_cfg.get("affine_enabled", post_align_cfg.get("affine", False))),
                    affine_scale=float(_cfg_value(post_align_cfg, "affine_scale", 0.1)),
                    affine_init_identity=bool(post_align_cfg.get("affine_init_identity", True)),
                    affine_per_channel=False,
                    affine_type=str(_cfg_value(post_align_cfg, "affine_type", "affine")),
                )

            residual_scale_init = float(
                ms_residual_stem_cfg.get(
                    "scale_init",
                    1.0 if self.ms_residual_fusion_mode == "concat_proj" else 0.05,
                )
            )
            residual_scale_per_channel = bool(ms_residual_stem_cfg.get("scale_per_channel", True))
            scale_shape = (1, c2_in_channels, 1, 1) if residual_scale_per_channel else (1,)
            self.ms_residual_scale = nn.Parameter(torch.full(scale_shape, residual_scale_init))

        # Optional CRGGA on MSI-only single-stream inputs.
        # - input_enabled: align raw MS bands before the original stem, keeping the stem unchanged.
        # - enabled: align projected stage features (B,C,H,W) at selected stages.
        ms_group_align_cfg: dict[str, Any] = {}
        if ms_group_align is not None:
            if isinstance(ms_group_align, Mapping):
                ms_group_align_cfg = dict(ms_group_align)
            elif hasattr(ms_group_align, "items"):
                ms_group_align_cfg = {k: v for k, v in ms_group_align.items()}  # type: ignore[assignment]
            else:
                raise TypeError(f"Unsupported ms_group_align type: {type(ms_group_align)}")
        self.ms_group_align_cfg = ms_group_align_cfg
        self.ms_group_align_enabled = bool(ms_group_align_cfg.get("enabled", ms_group_align_cfg.get("enable", False)))
        self.ms_group_align_input_enabled = bool(
            ms_group_align_cfg.get("input_enabled", ms_group_align_cfg.get("input_enable", False))
        )
        group_position_norm = str(ms_group_align_cfg.get("position", "pre_block") or "pre_block").strip().lower()
        if group_position_norm in {"pre", "pre_block", "before", "before_block"}:
            group_position_norm = "pre_block"
        elif group_position_norm in {"post", "post_block", "after", "after_block"}:
            group_position_norm = "post_block"
        else:
            raise ValueError(
                "Unsupported ms_group_align.position; expected pre_block/post_block, "
                f"got {ms_group_align_cfg.get('position')}"
            )
        self.ms_group_align_position = group_position_norm

        stage_idx_raw = ms_group_align_cfg.get("stage_idx", ms_group_align_cfg.get("stages", None))
        if stage_idx_raw is None:
            ms_group_stage_idx = [0] if self.ms_group_align_enabled else []
        elif isinstance(stage_idx_raw, (list, tuple)) or (
            hasattr(stage_idx_raw, "__iter__") and not isinstance(stage_idx_raw, (str, bytes))
        ):
            ms_group_stage_idx = [_normalize_stage_idx(v) for v in list(stage_idx_raw)]
        else:
            ms_group_stage_idx = [_normalize_stage_idx(stage_idx_raw)]
        dedup_group_stage: list[int] = []
        for stage_i in ms_group_stage_idx:
            if int(stage_i) not in dedup_group_stage:
                dedup_group_stage.append(int(stage_i))
        self.ms_group_align_stage_idx = dedup_group_stage

        self.ms_group_input_aligner: GroupwiseDeformableAlign2D | None = None
        self.ms_group_aligners = nn.ModuleDict()
        if self.ms_in_chs <= 0:
            self.ms_group_align_enabled = False
            self.ms_group_align_input_enabled = False
        else:
            ref_mode = str(_cfg_value(ms_group_align_cfg, "ref_mode", "spatial_weighted"))
            ref_band_index = ms_group_align_cfg.get("ref_band_index", ms_group_align_cfg.get("ref_channel", None))
            num_iters = int(_cfg_value(ms_group_align_cfg, "num_iters", 1))
            ref_detach = bool(ms_group_align_cfg.get("ref_detach", False))
            num_keypoints = int(_cfg_value(ms_group_align_cfg, "num_keypoints", 5))
            offset_scale = float(_cfg_value(ms_group_align_cfg, "offset_scale", 6.0))
            offset_enabled = bool(ms_group_align_cfg.get("offset_enabled", True))
            attention_norm = str(_cfg_value(ms_group_align_cfg, "attention_norm", "sigmoid"))
            padding_mode = str(_cfg_value(ms_group_align_cfg, "padding_mode", "border"))
            align_corners = bool(ms_group_align_cfg.get("align_corners", True))
            loss_type = str(_cfg_value(ms_group_align_cfg, "loss_type", "infonce"))
            loss_downsample = ms_group_align_cfg.get("loss_downsample", None)
            nce_num_patches = int(_cfg_value(ms_group_align_cfg, "nce_num_patches", 64))
            nce_patch_size = int(_cfg_value(ms_group_align_cfg, "nce_patch_size", 5))
            nce_tau = float(_cfg_value(ms_group_align_cfg, "nce_tau", 0.2))
            affine_enabled = bool(ms_group_align_cfg.get("affine_enabled", ms_group_align_cfg.get("affine", False)))
            affine_scale = float(_cfg_value(ms_group_align_cfg, "affine_scale", 0.1))
            affine_init_identity = bool(ms_group_align_cfg.get("affine_init_identity", True))
            affine_type = str(_cfg_value(ms_group_align_cfg, "affine_type", "affine"))
            loss_weight = float(_cfg_value(ms_group_align_cfg, "loss_weight", 0.02))
            loss_offset_weight = float(
                _cfg_value(
                    ms_group_align_cfg,
                    "loss_offset_weight",
                    _cfg_value(
                        ms_group_align_cfg,
                        "offset_loss_weight",
                        _cfg_value(ms_group_align_cfg, "offset_reg_weight", 0.0),
                    ),
                )
            )
            loss_attn_norm_weight = float(
                _cfg_value(
                    ms_group_align_cfg,
                    "loss_attn_norm_weight",
                    _cfg_value(
                        ms_group_align_cfg,
                        "attn_norm_weight",
                        _cfg_value(ms_group_align_cfg, "attn_reg_weight", 0.0),
                    ),
                )
            )
            loss_attn_entropy_weight = float(
                _cfg_value(
                    ms_group_align_cfg,
                    "loss_attn_entropy_weight",
                    _cfg_value(
                        ms_group_align_cfg,
                        "attn_entropy_weight",
                        _cfg_value(ms_group_align_cfg, "attn_entropy_reg_weight", 0.0),
                    ),
                )
            )

            if self.ms_group_align_input_enabled:
                input_ref_mode = str(_cfg_value(ms_group_align_cfg, "input_ref_mode", ref_mode))
                input_ref_band_index = ms_group_align_cfg.get(
                    "input_ref_band_index",
                    ms_group_align_cfg.get("input_ref_channel", ref_band_index),
                )
                input_num_iters = int(_cfg_value(ms_group_align_cfg, "input_num_iters", num_iters))
                input_ref_detach = bool(_cfg_value(ms_group_align_cfg, "input_ref_detach", ref_detach))
                input_num_keypoints = int(_cfg_value(ms_group_align_cfg, "input_num_keypoints", num_keypoints))
                input_offset_scale = float(_cfg_value(ms_group_align_cfg, "input_offset_scale", offset_scale))
                input_offset_enabled = bool(
                    _cfg_value(
                        ms_group_align_cfg,
                        "input_offset_enabled",
                        _cfg_value(ms_group_align_cfg, "input_use_offset", offset_enabled),
                    )
                )
                input_attention_norm = str(_cfg_value(ms_group_align_cfg, "input_attention_norm", attention_norm))
                input_padding_mode = str(_cfg_value(ms_group_align_cfg, "input_padding_mode", padding_mode))
                input_align_corners = bool(_cfg_value(ms_group_align_cfg, "input_align_corners", align_corners))
                input_loss_type = str(_cfg_value(ms_group_align_cfg, "input_loss_type", loss_type))
                input_loss_downsample = ms_group_align_cfg.get("input_loss_downsample", loss_downsample)
                input_nce_num_patches = int(_cfg_value(ms_group_align_cfg, "input_nce_num_patches", nce_num_patches))
                input_nce_patch_size = int(_cfg_value(ms_group_align_cfg, "input_nce_patch_size", nce_patch_size))
                input_nce_tau = float(_cfg_value(ms_group_align_cfg, "input_nce_tau", nce_tau))
                input_affine_enabled = bool(
                    _cfg_value(
                        ms_group_align_cfg,
                        "input_affine_enabled",
                        _cfg_value(ms_group_align_cfg, "input_affine", affine_enabled),
                    )
                )
                input_affine_scale = float(_cfg_value(ms_group_align_cfg, "input_affine_scale", affine_scale))
                input_affine_init_identity = bool(
                    _cfg_value(ms_group_align_cfg, "input_affine_init_identity", affine_init_identity)
                )
                input_affine_type = str(_cfg_value(ms_group_align_cfg, "input_affine_type", affine_type))
                input_loss_weight = float(_cfg_value(ms_group_align_cfg, "input_loss_weight", loss_weight))
                input_loss_offset_weight = float(
                    _cfg_value(ms_group_align_cfg, "input_loss_offset_weight", loss_offset_weight)
                )
                input_loss_attn_norm_weight = float(
                    _cfg_value(ms_group_align_cfg, "input_loss_attn_norm_weight", loss_attn_norm_weight)
                )
                input_loss_attn_entropy_weight = float(
                    _cfg_value(ms_group_align_cfg, "input_loss_attn_entropy_weight", loss_attn_entropy_weight)
                )

                self.ms_group_input_aligner = GroupwiseDeformableAlign2D(
                    in_channels=1,
                    ref_mode=input_ref_mode,
                    ref_band_index=input_ref_band_index,
                    num_iters=input_num_iters,
                    ref_detach=input_ref_detach,
                    num_keypoints=input_num_keypoints,
                    offset_scale=input_offset_scale,
                    offset_enabled=input_offset_enabled,
                    attention_norm=input_attention_norm,
                    padding_mode=input_padding_mode,
                    align_corners=input_align_corners,
                    loss_type=input_loss_type,
                    loss_downsample=input_loss_downsample,
                    nce_num_patches=input_nce_num_patches,
                    nce_patch_size=input_nce_patch_size,
                    nce_tau=input_nce_tau,
                    affine_enabled=input_affine_enabled,
                    affine_scale=input_affine_scale,
                    affine_init_identity=input_affine_init_identity,
                    affine_type=input_affine_type,
                    loss_weight=input_loss_weight,
                    loss_offset_weight=input_loss_offset_weight,
                    loss_attn_norm_weight=input_loss_attn_norm_weight,
                    loss_attn_entropy_weight=input_loss_attn_entropy_weight,
                )

            if self.ms_group_align_enabled:
                num_groups_raw = ms_group_align_cfg.get("num_groups", ms_group_align_cfg.get("groups", self.ms_in_chs))
                group_channels_raw = ms_group_align_cfg.get(
                    "group_channels",
                    ms_group_align_cfg.get("proj_channels", 8),
                )
                num_groups = self.ms_in_chs if num_groups_raw is None else int(num_groups_raw)
                group_channels = 8 if group_channels_raw is None else int(group_channels_raw)
                if num_groups <= 1:
                    raise ValueError(f"ms_group_align.num_groups must be > 1, got {num_groups}")
                if group_channels <= 0:
                    raise ValueError(f"ms_group_align.group_channels must be > 0, got {group_channels}")

                stage_channels = stage_out_channels if self.ms_group_align_position == "post_block" else stage_in_channels
                for stage_i in self.ms_group_align_stage_idx:
                    if stage_i < 0 or stage_i >= len(stage_channels):
                        raise ValueError(
                            f"ms_group_align.stage_idx contains invalid stage idx {stage_i} (num_stages={len(stage_channels)})"
                        )
                    in_channels = int(stage_channels[int(stage_i)])
                    self.ms_group_aligners[str(stage_i)] = ProjectedGroupwiseDeformableAlign2D(
                        in_channels=in_channels,
                        num_groups=num_groups,
                        group_channels=group_channels,
                        ref_mode=ref_mode,
                        ref_band_index=ref_band_index,
                        num_iters=num_iters,
                        ref_detach=ref_detach,
                        num_keypoints=num_keypoints,
                        offset_scale=offset_scale,
                        offset_enabled=offset_enabled,
                        attention_norm=attention_norm,
                        padding_mode=padding_mode,
                        align_corners=align_corners,
                        loss_type=loss_type,
                        loss_downsample=loss_downsample,
                        nce_num_patches=nce_num_patches,
                        nce_patch_size=nce_patch_size,
                        nce_tau=nce_tau,
                        affine_enabled=affine_enabled,
                        affine_scale=affine_scale,
                        affine_init_identity=affine_init_identity,
                        affine_type=affine_type,
                        loss_weight=loss_weight,
                        loss_offset_weight=loss_offset_weight,
                        loss_attn_norm_weight=loss_attn_norm_weight,
                        loss_attn_entropy_weight=loss_attn_entropy_weight,
                    )

        # Optional EEMSA (Edge-Enhanced Multi-Scale Attention) at stem / selected stages.
        self.eemsa_stem: EEMSA | None = None
        self.eemsa_stage_modules = nn.ModuleDict()
        eemsa_cfg_raw = eemsa
        if eemsa_cfg_raw is not None:
            if isinstance(eemsa_cfg_raw, Mapping):
                eemsa_cfg = dict(eemsa_cfg_raw)
            elif hasattr(eemsa_cfg_raw, "items"):
                eemsa_cfg = {k: v for k, v in eemsa_cfg_raw.items()}  # type: ignore[assignment]
            else:
                raise TypeError(f"Unsupported EEMSA config type: {type(eemsa_cfg_raw)}")

            enabled = bool(eemsa_cfg.get("enabled", eemsa_cfg.get("enable", False)))
            if enabled:
                mode = str(eemsa_cfg.get("mode", "insert") or "insert").strip().lower()
                if mode not in {"insert"}:
                    raise ValueError(
                        f"Unsupported EEMSA mode={mode}. Current implementation only supports insert."
                    )

                locations = _parse_eemsa_locations(eemsa_cfg.get("locations", eemsa_cfg.get("location", None)))
                ratio = float(eemsa_cfg.get("ratio", 0.25) or 0.25)
                min_channels = int(eemsa_cfg.get("min_channels", 32) or 32)
                edge_dw_kernel_size = int(eemsa_cfg.get("edge_dw_kernel_size", 3) or 3)
                edge_use_pointwise = bool(eemsa_cfg.get("edge_use_pointwise", True))
                ema_groups = int(eemsa_cfg.get("ema_groups", 8) or 8)
                ema_conv_kernel_size = int(eemsa_cfg.get("ema_conv_kernel_size", 3) or 3)
                ema_use_group_norm = bool(eemsa_cfg.get("ema_use_group_norm", True))
                fusion = str(eemsa_cfg.get("fusion", "weighted_sum") or "weighted_sum")
                norm = str(eemsa_cfg.get("norm", "bn") or "bn")
                act = str(eemsa_cfg.get("act", "silu") or "silu")
                use_eca = bool(eemsa_cfg.get("use_eca", True))
                eca_kernel_size = int(eemsa_cfg.get("eca_kernel_size", 3) or 3)
                alpha_init = float(eemsa_cfg.get("alpha_init", 0.1) or 0.1)

                if "stem" in locations:
                    self.eemsa_stem = EEMSA(
                        channels=int(stem_channels[2]),
                        ratio=ratio,
                        min_channels=min_channels,
                        edge_dw_kernel_size=edge_dw_kernel_size,
                        edge_use_pointwise=edge_use_pointwise,
                        ema_groups=ema_groups,
                        ema_conv_kernel_size=ema_conv_kernel_size,
                        ema_use_group_norm=ema_use_group_norm,
                        fusion=fusion,
                        norm=norm,
                        act=act,
                        use_eca=use_eca,
                        eca_kernel_size=eca_kernel_size,
                        alpha_init=alpha_init,
                    )

                for stage_idx_1b, stage_name in enumerate(["stage1", "stage2", "stage3", "stage4"], start=1):
                    if stage_name not in locations:
                        continue
                    stage_idx_0b = stage_idx_1b - 1
                    if stage_idx_0b >= len(stage_keys):
                        continue
                    stage_k = stage_keys[stage_idx_0b]
                    stage_out_channels = int(stage_config[stage_k][2])
                    self.eemsa_stage_modules[str(stage_idx_0b)] = EEMSA(
                        channels=stage_out_channels,
                        ratio=ratio,
                        min_channels=min_channels,
                        edge_dw_kernel_size=edge_dw_kernel_size,
                        edge_use_pointwise=edge_use_pointwise,
                        ema_groups=ema_groups,
                        ema_conv_kernel_size=ema_conv_kernel_size,
                        ema_use_group_norm=ema_use_group_norm,
                        fusion=fusion,
                        norm=norm,
                        act=act,
                        use_eca=use_eca,
                        eca_kernel_size=eca_kernel_size,
                        alpha_init=alpha_init,
                    )

        # stages
        self.stages = nn.ModuleList()
        for i, k in enumerate(stage_config):
            in_channels, mid_channels, out_channels, block_num, downsample, light_block, kernel_size, layer_num = stage_config[
                k]
            self.stages.append(
                HG_Stage(
                    in_channels,
                    mid_channels,
                    out_channels,
                    block_num,
                    layer_num,
                    downsample,
                    light_block,
                    kernel_size,
                    use_lab))

        if freeze_at >= 0:
            self._freeze_parameters(self.stem)
            if not freeze_stem_only:
                for i in range(min(freeze_at + 1, len(self.stages))):
                    self._freeze_parameters(self.stages[i])

        if freeze_norm:
            self._freeze_norm(self)

        if pretrained:
            RED, GREEN, RESET = "\033[91m", "\033[92m", "\033[0m"
            try:
                model_path = local_model_dir + 'PPHGNetV2_' + name + '_stage1.pth'
                if os.path.exists(model_path):
                    state = torch.load(model_path, map_location='cpu')
                    print(f"Loaded stage1 {name} HGNetV2 from local file.")
                else:
                    # If the file doesn't exist locally, download from the URL
                    if torch.distributed.get_rank() == 0:
                        print(GREEN + "If the pretrained HGNetV2 can't be downloaded automatically. Please check your network connection." + RESET)
                        print(GREEN + "Please check your network connection. Or download the model manually from " + RESET + f"{download_url}" + GREEN + " to " + RESET + f"{local_model_dir}." + RESET)
                        state = torch.hub.load_state_dict_from_url(download_url, map_location='cpu', model_dir=local_model_dir)
                        torch.distributed.barrier()
                    else:
                        torch.distributed.barrier()
                        state = torch.load(local_model_dir)

                    print(f"Loaded stage1 {name} HGNetV2 from URL.")

                if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
                    state = state["state_dict"]

                # 适配输入通道数：若 checkpoint 的 stem conv 为 3 通道，而当前模型为其它通道数，则做裁剪/扩展。
                if isinstance(state, dict) and stem_channels[0] != 3:
                    model_state = self.state_dict()
                    for k, v in list(state.items()):
                        if k not in model_state:
                            continue
                        target = model_state[k]
                        if (
                            isinstance(v, torch.Tensor)
                            and v.ndim == 4
                            and target.ndim == 4
                            and v.shape[0] == target.shape[0]
                            and v.shape[2:] == target.shape[2:]
                            and v.shape[1] == 3
                            and target.shape[1] == stem_channels[0]
                        ):
                            desired_in = int(stem_channels[0])
                            if desired_in < 3:
                                if desired_in == 1:
                                    state[k] = v.mean(dim=1, keepdim=True)
                                else:
                                    state[k] = v[:, :desired_in]
                            else:
                                expanded = target.new_zeros(target.shape)
                                expanded[:, :3] = v
                                if desired_in > 3:
                                    mean = v.mean(dim=1, keepdim=True)
                                    expanded[:, 3:] = mean.repeat(1, desired_in - 3, 1, 1)
                                state[k] = expanded

                self.load_state_dict(state, strict=False)

            except (Exception, KeyboardInterrupt) as e:
                if torch.distributed.get_rank() == 0:
                    print(f"{str(e)}")
                    logging.error(RED + "CRITICAL WARNING: Failed to load pretrained HGNetV2 model" + RESET)
                    logging.error(GREEN + "Please check your network connection. Or download the model manually from " \
                                + RESET + f"{download_url}" + GREEN + " to " + RESET + f"{local_model_dir}." + RESET)
                exit()




    def _freeze_norm(self, m: nn.Module):
        if isinstance(m, nn.BatchNorm2d):
            m = FrozenBatchNorm2d(m.num_features)
        else:
            for name, child in m.named_children():
                _child = self._freeze_norm(child)
                if _child is not child:
                    setattr(m, name, _child)
        return m

    def _freeze_parameters(self, m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False

    @staticmethod
    def _merge_aux_losses(
        aux_losses: dict[str, torch.Tensor],
        new_losses: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        for key, value in new_losses.items():
            if torch.is_tensor(value):
                aux_losses[key] = aux_losses.get(key, 0.0) + value
        return aux_losses

    @staticmethod
    def _safe_normalize_attention(attn: torch.Tensor) -> torch.Tensor:
        if attn.ndim != 4:
            raise ValueError(f"Expected attention tensor shaped (B,K,H,W), got {attn.shape}")
        denom = attn.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return attn / denom

    def _apply_ms_group_input_alignment(
        self,
        x: torch.Tensor,
        aux_losses: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not self.ms_group_align_input_enabled:
            return x, aux_losses
        if self.ms_group_input_aligner is None:
            raise RuntimeError("Missing ms_group_input_aligner")
        out = self.ms_group_input_aligner(x.unsqueeze(2))
        if self.training and isinstance(out, tuple) and len(out) == 2:
            x_aligned, stage_losses = out
            aux_losses = self._merge_aux_losses(aux_losses, stage_losses)
        else:
            x_aligned = out
        return x_aligned.squeeze(2), aux_losses

    def _apply_ms_group_alignment(
        self,
        idx: int,
        x: torch.Tensor,
        aux_losses: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not self.ms_group_align_enabled or idx not in self.ms_group_align_stage_idx:
            return x, aux_losses
        key = str(idx)
        if key not in self.ms_group_aligners:
            raise RuntimeError(f"Missing ms_group_align module for stage idx {idx}")
        aligner = self.ms_group_aligners[key]
        if self.training:
            out = aligner(x)
            if isinstance(out, tuple) and len(out) == 2:
                x_aligned, stage_losses = out
                aux_losses = self._merge_aux_losses(aux_losses, stage_losses)
                x = x_aligned
            else:
                x = out  # pragma: no cover
        else:
            out = aligner(x)
            x = out[0] if isinstance(out, tuple) else out
        return x, aux_losses

    def _apply_ms_residual_post_align(
        self,
        ref: torch.Tensor,
        residual: torch.Tensor,
        aux_losses: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not self.ms_residual_post_align_enabled or self.ms_residual_post_aligner is None:
            return residual, aux_losses

        aligner = self.ms_residual_post_aligner
        ref_pred = ref.detach() if self.ms_residual_post_align_ref_detach else ref
        if self.training:
            pred = aligner.predict(ref_pred, residual)
            if aligner.affine_enabled:
                offset_x, offset_y, attn_weights, affine_theta = pred
                residual_aligned, _, attn_exp = aligner.deform_with_attention(
                    residual,
                    offset_x=offset_x,
                    offset_y=offset_y,
                    attention_weights=attn_weights,
                    affine_theta=affine_theta,
                )
                loss_dict = aligner.loss_calculate(
                    ref_pred,
                    offset_x,
                    offset_y,
                    residual_aligned,
                    attn_exp,
                    affine_theta=affine_theta,
                )
            else:
                offset_x, offset_y, attn_weights = pred
                residual_aligned, _, attn_exp = aligner.deform_with_attention(
                    residual,
                    offset_x=offset_x,
                    offset_y=offset_y,
                    attention_weights=attn_weights,
                )
                loss_dict = aligner.loss_calculate(ref_pred, offset_x, offset_y, residual_aligned, attn_exp)

            if self.ms_residual_post_align_loss_weight > 0 and "loss_deform_align" in loss_dict:
                aux_losses["loss_deform_align"] = aux_losses.get("loss_deform_align", 0.0) + (
                    loss_dict["loss_deform_align"] * self.ms_residual_post_align_loss_weight
                )

            if (
                self.ms_residual_post_align_loss_offset_weight > 0
                or self.ms_residual_post_align_loss_attn_norm_weight > 0
                or self.ms_residual_post_align_loss_attn_entropy_weight > 0
            ):
                _, _, hh, ww = attn_weights.shape
                if self.ms_residual_post_align_loss_offset_weight > 0:
                    denom_x = max(int(ww) - 1, 1) / 2.0
                    denom_y = max(int(hh) - 1, 1) / 2.0
                    offset_x_px = offset_x * float(denom_x)
                    offset_y_px = offset_y * float(denom_y)
                    p = self._safe_normalize_attention(attn_weights)
                    fused_x = (p * offset_x_px).sum(dim=1)
                    fused_y = (p * offset_y_px).sum(dim=1)
                    aux_losses["loss_deform_offset"] = aux_losses.get("loss_deform_offset", 0.0) + (
                        torch.sqrt(fused_x ** 2 + fused_y ** 2 + 1e-8).mean()
                        * self.ms_residual_post_align_loss_offset_weight
                    )

                if self.ms_residual_post_align_loss_attn_norm_weight > 0:
                    attn_sum = attn_weights.sum(dim=1)
                    aux_losses["loss_deform_attn"] = aux_losses.get("loss_deform_attn", 0.0) + (
                        ((attn_sum - 1.0) ** 2).mean() * self.ms_residual_post_align_loss_attn_norm_weight
                    )

                if self.ms_residual_post_align_loss_attn_entropy_weight > 0:
                    p = self._safe_normalize_attention(attn_weights)
                    ent = -(p * torch.log(p.clamp_min(1e-8))).sum(dim=1).mean()
                    aux_losses["loss_deform_attn_entropy"] = aux_losses.get("loss_deform_attn_entropy", 0.0) + (
                        ent * self.ms_residual_post_align_loss_attn_entropy_weight
                    )

            return residual_aligned, aux_losses

        return aligner(ref_pred, residual), aux_losses

    def _apply_ms_residual_stem_interactive(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        aux_losses: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not self.ms_residual_stem_interactive_enabled or self.ms_residual_stem_interactive is None:
            return x, aux_losses
        x = self.ms_residual_stem_interactive(x, residual)
        return x, aux_losses

    def forward(self, x):
        aux_losses: dict[str, torch.Tensor] = {}
        x, aux_losses = self._apply_ms_group_input_alignment(x, aux_losses)
        stem_input = x
        if self.ms_band_sep_enabled and self.ms_band_sep_stem is not None:
            out = self.ms_band_sep_stem(stem_input)
            if self.training and isinstance(out, tuple) and len(out) == 2:
                x, stage_losses = out
                aux_losses = self._merge_aux_losses(aux_losses, stage_losses)
            else:
                x = out[0] if isinstance(out, tuple) else out
        else:
            x = self.stem(stem_input)
            if self.ms_residual_stem_enabled and self.ms_residual_stem_branch is not None and self.ms_residual_scale is not None:
                out = self.ms_residual_stem_branch(stem_input)
                if self.training and isinstance(out, tuple) and len(out) == 2:
                    residual, stage_losses = out
                    aux_losses = self._merge_aux_losses(aux_losses, stage_losses)
                else:
                    residual = out[0] if isinstance(out, tuple) else out
                residual, aux_losses = self._apply_ms_residual_post_align(x, residual, aux_losses)
                x, aux_losses = self._apply_ms_residual_stem_interactive(x, residual, aux_losses)
                if self.ms_residual_fusion_mode == "concat_proj":
                    if self.ms_residual_fuse_proj is None:
                        raise RuntimeError("Missing ms_residual_fuse_proj for concat_proj fusion")
                    delta = self.ms_residual_fuse_proj(torch.cat([x, residual], dim=1))
                    x = x + (self.ms_residual_scale * delta)
                else:
                    x = x + (self.ms_residual_scale * residual)
        if self.eemsa_stem is not None:
            x = self.eemsa_stem(x)
        outs = []
        for idx, stage in enumerate(self.stages):
            if self.ms_group_align_position == "pre_block":
                x, aux_losses = self._apply_ms_group_alignment(idx, x, aux_losses)
            x = stage(x)
            if self.ms_group_align_position == "post_block":
                x, aux_losses = self._apply_ms_group_alignment(idx, x, aux_losses)
            edge_key = str(idx)
            if edge_key in self.eemsa_stage_modules:
                x = self.eemsa_stage_modules[edge_key](x)
            if idx in self.return_idx:
                outs.append(x)
        if self.training and aux_losses:
            return outs, {k: v for k, v in aux_losses.items() if torch.is_tensor(v)}
        return outs
