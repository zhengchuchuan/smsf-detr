from __future__ import annotations

"""
Dual-stream HGNetv2 backbones.

This file intentionally keeps multi-modal (RGB/MS) dual-stream logic separate from the original HGNetv2
implementation in `hgnetv2.py`.
"""

# High-level layout:
# - small attention helpers (ECA/SE) for the MS branch
# - config normalization utilities
# - HGNetv2DualStream backbone with optional fusion/alignment modules
# - forward path that returns multi-scale features + optional aux losses

import math
from typing import Any, Callable, Dict, List, Mapping, Sequence

import torch
import torch.nn as nn

from .adaptive_sampling_ms_fusion import ASMFusion2D
from .coattention import CoAttention2D
from .deform_align import DeformableAlign2D
from .group_deform_align import GroupwiseDeformableAlign2D, ProjectedGroupwiseDeformableAlign2D
from .ms_band_sep import MSBandSeparatedStemAlign
from .gpt_fusion import GPTFusion2D
from .hgnetv2 import EseModule, HGNetv2
from .mrt_fusion import MRTCrossSpectrumCorrFusion2D, MRTCrossSpectrumFusion2D
from .stem_cf_interactive import StemCFInteractive2D
from ..core import register

__all__ = [
    "HGNetv2DualStream",
    "HGNetv2DualStreamCoAttn",
    "HGNetv2DualStreamGPT",
]


class ECA2D(nn.Module):
    """
    Efficient Channel Attention (ECA) for 2D feature maps.

    Paper: https://arxiv.org/abs/1910.03151
    """

    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int | None = None,
        gamma: int = 2,
        b: int = 1,
    ) -> None:
        super().__init__()
        ch = int(channels)
        if ch <= 0:
            raise ValueError(f"ECA2D channels must be > 0, got {channels}")

        k = kernel_size
        if k is None:
            t = int(abs((math.log2(ch) + float(b)) / float(gamma)))
            k = t if t % 2 == 1 else t + 1
        k = max(1, int(k))
        if k % 2 == 0:
            k += 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Global pooling -> 1D conv -> sigmoid gate -> channel-wise reweighting.
        y = self.avg_pool(x)  # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(1, 2)  # (B, 1, C)
        y = self.conv(y)
        y = self.sigmoid(y)
        y = y.transpose(1, 2).unsqueeze(-1)  # (B, C, 1, 1)
        return x * y


class SqueezeExcite2D(nn.Module):
    def __init__(self, channels: int, *, reduction: int = 16) -> None:
        super().__init__()
        ch = int(channels)
        r = int(reduction)
        if ch <= 0:
            raise ValueError(f"SqueezeExcite2D channels must be > 0, got {channels}")
        if r <= 0:
            raise ValueError(f"SqueezeExcite2D reduction must be > 0, got {reduction}")
        hidden = max(1, ch // r)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(ch, hidden, kernel_size=1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, ch, kernel_size=1, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Squeeze -> excitation -> channel gate.
        w = self.avg_pool(x)
        w = self.fc1(w)
        w = self.act(w)
        w = self.fc2(w)
        w = self.gate(w)
        return x * w


def _build_channel_attn(
    kind: str,
    channels: int,
    *,
    eca_kernel_size: int | None = None,
    se_reduction: int = 16,
) -> nn.Module:
    # Factory for lightweight channel attention blocks on the MS branch.
    k = str(kind).strip().lower()
    if k in {"", "none", "no", "identity"}:
        return nn.Identity()
    if k in {"eca"}:
        return ECA2D(int(channels), kernel_size=eca_kernel_size)
    if k in {"se", "squeeze_excite", "squeeze-excite"}:
        return SqueezeExcite2D(int(channels), reduction=int(se_reduction))
    if k in {"ese"}:
        return EseModule(int(channels))
    raise ValueError(f"Unsupported channel attention kind: {kind} (supported: none|eca|se|ese)")


def _normalize_fuse_kv_stride(
    kv_stride: int | Sequence[int] | Mapping[str, int] | Mapping[int, int] | None,
    *,
    fuse_stage_idx: Sequence[int],
) -> Dict[int, int]:
    # Normalize stage-wise kv stride (int, list, or mapping keyed by stage/c-level).
    if kv_stride is None:
        # Default for C3/C4/C5 (stage idx 1/2/3): 80->10, 40->10, 20->10 for 640 input.
        default = {1: 8, 2: 4, 3: 2}
        return {int(i): int(default.get(int(i), 4)) for i in fuse_stage_idx}

    if isinstance(kv_stride, int):
        return {int(i): int(kv_stride) for i in fuse_stage_idx}

    if isinstance(kv_stride, (list, tuple)):
        if len(kv_stride) != len(fuse_stage_idx):
            raise ValueError(
                f"kv_stride as list/tuple must match fuse_stage_idx length, got kv_stride={kv_stride} fuse_stage_idx={fuse_stage_idx}"
            )
        return {int(stage_i): int(stride) for stage_i, stride in zip(fuse_stage_idx, kv_stride)}

    if isinstance(kv_stride, Mapping):
        # Support both {1:8,2:4,3:2} and {c3:8,c4:4,c5:2}.
        mapped: Dict[int, int] = {}
        for k, v in kv_stride.items():
            if isinstance(k, str):
                kk = k.strip().lower()
                if kk in {"c3", "p3"}:
                    mapped[1] = int(v)
                elif kk in {"c4", "p4"}:
                    mapped[2] = int(v)
                elif kk in {"c5", "p5"}:
                    mapped[3] = int(v)
                else:
                    raise ValueError(f"Unsupported kv_stride key: {k}")
            else:
                mapped[int(k)] = int(v)
        return {int(i): int(mapped.get(int(i), 4)) for i in fuse_stage_idx}

    raise TypeError(f"Unsupported kv_stride type: {type(kv_stride)}")


def _parse_bool(value: Any) -> bool:
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes", "y"}:
            return True
        if text in {"false", "0", "no", "n"}:
            return False
    return bool(value)


def _cfg_value(cfg: Mapping[str, Any], key: str, default: Any) -> Any:
    value = cfg.get(key, default)
    return default if value is None else value


def _normalize_stage_param(
    value: Any,
    *,
    fuse_stage_idx: Sequence[int],
    default: Any | Mapping[int, Any] | None,
    param_name: str,
    cast: Callable[[Any], Any],
) -> Dict[int, Any]:
    # Accept a scalar/list/dict and return a {stage_idx -> value} mapping.
    if value is None:
        if default is None:
            raise ValueError(f"Missing required param: {param_name}")
        if isinstance(default, Mapping):
            default_map: Dict[int, Any] = {_normalize_stage_idx(k): cast(v) for k, v in default.items()}
            out: Dict[int, Any] = {}
            for stage_i in fuse_stage_idx:
                key = int(stage_i)
                if key not in default_map:
                    raise ValueError(f"{param_name} missing stage idx {key} and no default provided")
                out[key] = default_map[key]
            return out
        return {int(i): cast(default) for i in fuse_stage_idx}

    if isinstance(value, (list, tuple)):
        if len(value) != len(fuse_stage_idx):
            raise ValueError(
                f"{param_name} as list/tuple must match fuse_stage_idx length, got {value} fuse_stage_idx={fuse_stage_idx}"
            )
        return {int(stage_i): cast(v) for stage_i, v in zip(fuse_stage_idx, value)}

    if isinstance(value, Mapping):
        mapped: Dict[int, Any] = {}
        for k, v in value.items():
            mapped[_normalize_stage_idx(k)] = cast(v)
        out: Dict[int, Any] = {}
        default_map: Dict[int, Any] | None = None
        if isinstance(default, Mapping):
            default_map = {_normalize_stage_idx(k): cast(v) for k, v in default.items()}
        for stage_i in fuse_stage_idx:
            key = int(stage_i)
            if key in mapped:
                out[key] = mapped[key]
            elif default_map is not None:
                if key not in default_map:
                    raise ValueError(f"{param_name} missing stage idx {key} and no default provided")
                out[key] = default_map[key]
            elif default is not None:
                out[key] = cast(default)
            else:
                raise ValueError(f"{param_name} missing stage idx {key} and no default provided")
        return out

    return {int(i): cast(value) for i in fuse_stage_idx}


def _normalize_stage_idx(key: Any) -> int:
    # Map c2/c3/c4/c5 to stage indices 0..3; pass through ints.
    if isinstance(key, str):
        kk = key.strip().lower()
        if kk in {"c2", "p2"}:
            return 0
        if kk in {"c3", "p3"}:
            return 1
        if kk in {"c4", "p4"}:
            return 2
        if kk in {"c5", "p5"}:
            return 3
        return int(kk)
    return int(key)


def _normalize_merge_mode(mode: Any) -> str:
    # Canonicalize output merge modes (avg->wavg, concat variants -> concat1x1).
    m = str(mode).strip().lower()
    if m == "avg":
        m = "wavg"
    if m in {"sum", "plus"}:
        m = "add"
    if m in {"concat", "concat_1x1", "concat-1x1", "contact"}:
        m = "concat1x1"
    return m


def _normalize_output_merge(
    output_merge: Any,
    *,
    return_idx: Sequence[int],
) -> Dict[int, str]:
    # Normalize output merge behavior per return_idx stage.
    supported = {"wavg", "add", "concat1x1"}
    if isinstance(output_merge, str):
        mode = _normalize_merge_mode(output_merge)
        if mode not in supported:
            raise ValueError(f"Unsupported output_merge: {mode}")
        return {int(i): mode for i in return_idx}

    if isinstance(output_merge, Mapping):
        mapped: Dict[int, str] = {}
        default_mode: str | None = None
        for k, v in output_merge.items():
            if isinstance(k, str) and k.strip().lower() in {"default", "all"}:
                default_mode = _normalize_merge_mode(v)
                continue
            mapped[_normalize_stage_idx(k)] = _normalize_merge_mode(v)

        out: Dict[int, str] = {}
        for stage_i in return_idx:
            stage_i = int(stage_i)
            mode = mapped.get(stage_i, default_mode)
            if mode is None:
                raise ValueError(
                    "output_merge as mapping must provide per-stage modes for all return_idx, "
                    f"missing stage idx {stage_i} (return_idx={list(return_idx)})"
                )
            if mode not in supported:
                raise ValueError(f"Unsupported output_merge for stage idx {stage_i}: {mode}")
            out[stage_i] = mode
        return out

    raise TypeError(f"Unsupported output_merge type: {type(output_merge)}")


def _select_eemsa_cfg(eemsa: Mapping[str, Any] | None, stream: str) -> Dict[str, Any] | None:
    if eemsa is None:
        return None
    if not isinstance(eemsa, Mapping):
        if hasattr(eemsa, "items"):
            return {k: v for k, v in eemsa.items()}  # type: ignore[return-value]
        raise TypeError(f"Unsupported EEMSA config type: {type(eemsa)}")

    # Shared form:
    # backbone_eemsa:
    #   enabled: true
    #   ...
    shared_keys = {
        "enabled",
        "enable",
        "mode",
        "locations",
        "location",
        "ratio",
        "min_channels",
        "edge_dw_kernel_size",
        "edge_use_pointwise",
        "ema_groups",
        "ema_conv_kernel_size",
        "ema_use_group_norm",
        "fusion",
        "norm",
        "act",
        "use_eca",
        "eca_kernel_size",
        "alpha_init",
    }
    if any(k in eemsa for k in shared_keys):
        return dict(eemsa)

    # Per-stream form:
    # backbone_eemsa:
    #   rgb: {...}
    #   ms: {...}
    #   default: {...}
    selected = eemsa.get(stream, eemsa.get("default", None))
    if selected is None:
        return None
    if isinstance(selected, Mapping):
        return dict(selected)
    if hasattr(selected, "items"):
        return {k: v for k, v in selected.items()}  # type: ignore[return-value]
    raise TypeError(f"Unsupported EEMSA config for stream={stream}: {type(selected)}")


def _normalize_align_ref_mode(mode: str) -> str:
    # Normalize alignment reference modes for MS alignment modules.
    mode_norm = str(mode).strip().lower()
    if mode_norm in {"rgb", "cross", "cross_modal", "cross-modal"}:
        return "rgb"
    if mode_norm in {"ms", "ms_internal", "ms_internal_mean", "ms_mean", "mean"}:
        return "ms_mean"
    if mode_norm in {"ms_channel", "ms_chan", "ms_ch", "channel", "ch"}:
        return "ms_channel"
    if mode_norm in {"ms_weighted", "ms_weight", "weighted", "learned", "learnable"}:
        return "ms_weighted"
    raise ValueError(
        f"Unsupported align_ref_mode={mode} (supported: rgb|ms_mean|ms_channel|ms_weighted)"
    )


@register()
class HGNetv2DualStream(nn.Module):
    """
    Dual-stream HGNetv2 backbone with selectable cross-modal fusion at selected stages (e.g. C3/C4/C5).

    Inputs are expected to be channel-stacked (RGB first, then MS), and will be split inside the backbone.
    """

    def __init__(
        self,
        name: str,
        *,
        fusion_type: str = "coattention",
        output_fusion_type: str = "merge",
        rgb_in_chs: int = 3,
        ms_in_chs: int = 7,
        use_lab: bool = False,
        return_idx: List[int] = [1, 2, 3],
        fuse_stage_idx: List[int] | None = None,
        fusion_position: str = "pre_block",
        output_merge: str | Mapping[str, str] | Mapping[int, str] = "avg",
        ms_channel_attn: str = "none",
        ms_channel_attn_position: str = "pre_fuse",
        ms_channel_attn_stage_idx: List[int] | None = None,
        ms_eca_kernel_size: int | None = None,
        ms_se_reduction: int = 16,
        align_enabled: bool = False,
        align_stage_idx: List[int] | None = None,
        align_num_keypoints: int = 5,
        align_offset_scale: float = 6.0,
        align_offset_enabled: bool = True,
        align_per_channel_offset: bool = False,
        align_attention_norm: str = "sigmoid",
        align_position: str = "pre_block",
        align_padding_mode: str = "border",
        align_align_corners: bool = True,
        align_loss_type: str = "cosine",
        align_loss_downsample: float | None = None,
        align_nce_num_patches: int = 64,
        align_nce_patch_size: int = 5,
        align_nce_tau: float = 0.2,
        align_affine_enabled: bool = False,
        align_affine_scale: float = 0.1,
        align_affine_init_identity: bool = True,
        align_affine_per_channel: bool = False,
        align_affine_type: str = "affine",
        align_ref_mode: str = "rgb",
        align_ref_channel: int = 0,
        align_input_enabled: bool = False,
        align_input_proj: str = "rgb_to_ms",
        align_input_ref_mode: str = "rgb",
        align_input_ref_channel: int = 0,
        align_input_num_keypoints: int = 5,
        align_input_offset_enabled: bool = True,
        align_input_offset_scale: float = 6.0,
        align_input_per_channel_offset: bool = False,
        align_input_attention_norm: str = "sigmoid",
        align_input_padding_mode: str = "border",
        align_input_align_corners: bool = True,
        align_input_loss_type: str = "cosine",
        align_input_loss_downsample: float | None = None,
        align_input_nce_num_patches: int = 64,
        align_input_nce_patch_size: int = 5,
        align_input_nce_tau: float = 0.2,
        align_input_affine_enabled: bool = False,
        align_input_affine_scale: float = 0.1,
        align_input_affine_init_identity: bool = True,
        align_input_affine_per_channel: bool = False,
        align_input_affine_type: str = "affine",
        ms_band_sep: Mapping[str, Any] | None = None,
        ms_residual_stem: Mapping[str, Any] | None = None,
        ms_group_align: Mapping[str, Any] | None = None,
        eemsa: Mapping[str, Any] | None = None,
        fusion_d_model: int = 128,
        fusion_nhead: int = 8,
        fusion_block_exp: int = 4,
        fusion_n_layer: int = 8,
        fusion_vert_anchors: int = 8,
        fusion_horz_anchors: int = 8,
        fusion_dropout: float = 0.0,
        fusion_alpha_init: float = 0.0,
        fusion_writeback_merge: str = "add",
        fusion_kv_stride: int | Sequence[int] | Mapping[str, int] | Mapping[int, int] | None = None,
        fusion_num_stages: int = 2,
        fusion_use_pos_encoding: bool = True,
        fusion_c2former_groups: int | Sequence[int] | Mapping[str, int] | Mapping[int, int] | None = None,
        fusion_c2former_cca_stride: int | Sequence[int] | Mapping[str, int] | Mapping[int, int] | None = None,
        fusion_c2former_offset_range_factor: float | Sequence[float] | Mapping[str, float] | Mapping[int, float] | None = None,
        fusion_c2former_no_offset: bool | Sequence[bool] | Mapping[str, bool] | Mapping[int, bool] | None = None,
        fusion_c2former_attn_drop: float | None = None,
        fusion_c2former_proj_drop: float | None = None,
        fusion_c2former_offset_kernel_size: int | Sequence[int] | Mapping[str, int] | Mapping[int, int] | None = None,
        fusion_c2former_padding_mode: str | None = None,
        fusion_c2former_align_corners: bool | None = None,
        fusion_c2former_offset_on: str = "ms",
        fusion_c2former_global_kv: bool = False,
        fusion_c2former_global_vert_anchors: int = 8,
        fusion_c2former_global_horz_anchors: int = 8,
        fusion_c2former_use_pos_encoding: bool = False,
        fusion_c2former_pos_temperature: float = 10000.0,
        output_mrt_d_model: int = 128,
        output_mrt_num_stages: int = 2,
        output_mrt_fused_channels: int = 64,
        output_mrt_norm: str = "gn",
        output_mrt_use_pos_encoding: bool = True,
        output_cft_d_model: int = 128,
        output_cft_nhead: int = 8,
        output_cft_block_exp: int = 4,
        output_cft_n_layer: int = 8,
        output_cft_vert_anchors: int = 8,
        output_cft_horz_anchors: int = 8,
        output_cft_dropout: float = 0.0,
        output_cft_alpha_init: float = 0.0,
        output_cft_writeback_merge: str = "add",
        freeze_stem_only: bool = True,
        freeze_at: int = 0,
        freeze_norm: bool = True,
        pretrained: bool = True,
        local_model_dir: str = "weight/hgnetv2/",
    ) -> None:
        super().__init__()
        self.name = str(name)
        self.rgb_in_chs = int(rgb_in_chs)
        self.ms_in_chs = int(ms_in_chs)
        self.return_idx = [int(i) for i in return_idx]
        self.output_merge = output_merge
        self.output_merge_map = _normalize_output_merge(output_merge, return_idx=self.return_idx)

        # Normalize fusion type aliases (coattention/c2former/gpt/mrt/none).
        fusion_type_norm = str(fusion_type).strip().lower()
        if fusion_type_norm in {"none", "off", "disable", "disabled", ""}:
            fusion_type_norm = "none"
        elif fusion_type_norm in {"coattention", "co-attention", "co_attention", "coattn"}:
            fusion_type_norm = "coattention"
        elif fusion_type_norm in {"c2former", "c2f"}:
            fusion_type_norm = "c2former"
        elif fusion_type_norm in {"gpt", "cft", "fusion_transformer", "cft_gpt"}:
            fusion_type_norm = "gpt"
        elif fusion_type_norm in {"mrt", "cross_spectrum", "cross-spectrum", "crossspectrum"}:
            fusion_type_norm = "mrt"
        else:
            raise ValueError(f"Unsupported fusion_type: {fusion_type}")
        self.fusion_type = fusion_type_norm

        # Optional output-side fusion on top of per-stage features (before output merge).
        output_fusion_norm = str(output_fusion_type).strip().lower()
        if output_fusion_norm in {"", "none", "off", "disable", "disabled"}:
            # No extra output fusion module; fall back to `output_merge`.
            output_fusion_norm = "merge"
        elif output_fusion_norm in {"merge", "output_merge"}:
            output_fusion_norm = "merge"
        elif output_fusion_norm in {"gpt", "cft", "fusion_transformer", "cft_gpt"}:
            output_fusion_norm = "cft"
        elif output_fusion_norm in {"mrt", "mrt_corr", "mrt-corr", "corr", "cross_spectrum", "cross-spectrum"}:
            output_fusion_norm = "mrt_corr"
        else:
            raise ValueError(f"Unsupported output_fusion_type: {output_fusion_type}")
        self.output_fusion_type = output_fusion_norm

        if self.rgb_in_chs < 0 or self.ms_in_chs < 0:
            raise ValueError(f"rgb_in_chs/ms_in_chs must be >= 0, got rgb={self.rgb_in_chs} ms={self.ms_in_chs}")
        if self.rgb_in_chs + self.ms_in_chs <= 0:
            raise ValueError("At least one modality must have channels > 0.")

        # Default fusion stages follow return_idx if not specified.
        if fuse_stage_idx is None:
            fuse_stage_idx = list(self.return_idx)
        self.fuse_stage_idx = [int(i) for i in fuse_stage_idx]

        # Control where fusion is inserted relative to HGNetv2 stage internals.
        fusion_position_norm = str(fusion_position).strip().lower()
        if fusion_position_norm in {"pre_downsample", "pre-ds", "preds", "pre_stage", "stage_start", "before_downsample"}:
            fusion_position_norm = "pre_downsample"
        elif fusion_position_norm in {"pre", "pre_block", "before", "before_block", "before_blocks"}:
            fusion_position_norm = "pre_block"
        elif fusion_position_norm in {
            "post",
            "post_block",
            "after",
            "after_block",
            "after_blocks",
            "post_stage",
            "after_stage",
            "stage_end",
        }:
            # "stage end" in HGNetv2 means: after stage.blocks, before next stage.
            fusion_position_norm = "post_block"
        else:
            raise ValueError(
                "Unsupported fusion_position; expected pre_downsample/pre_block/post_block, "
                f"got {fusion_position}"
            )
        self.fusion_position = fusion_position_norm

        # Fetch HGNetv2 per-stage channel sizes for building fusion/alignment modules.
        stem_channels = list(HGNetv2.arch_configs[self.name]["stem_channels"])
        stage_config = HGNetv2.arch_configs[self.name]["stage_config"]
        stage_out_channels = [stage_config[k][2] for k in stage_config]
        stage_in_channels = [stage_config[k][0] for k in stage_config]

        # Optional channel attention on MS branch (pre/post block).
        self.ms_channel_attn = str(ms_channel_attn).strip().lower()
        self.ms_channel_attn_position = str(ms_channel_attn_position).strip().lower()
        if ms_channel_attn_stage_idx is None:
            ms_channel_attn_stage_idx = list(self.fuse_stage_idx)
        self.ms_channel_attn_stage_idx = [int(i) for i in ms_channel_attn_stage_idx]
        self.ms_eca_kernel_size = ms_eca_kernel_size
        self.ms_se_reduction = int(ms_se_reduction)

        self.rgb_backbone: HGNetv2 | None = None
        self.ms_backbone: HGNetv2 | None = None
        eemsa_cfg_rgb = _select_eemsa_cfg(eemsa, "rgb")
        eemsa_cfg_ms = _select_eemsa_cfg(eemsa, "ms")

        # Build independent HGNetv2 branches for each modality (RGB/MS).
        if self.rgb_in_chs > 0:
            self.rgb_backbone = HGNetv2(
                name=self.name,
                in_chs=self.rgb_in_chs,
                use_lab=use_lab,
                return_idx=self.return_idx,
                freeze_stem_only=freeze_stem_only,
                freeze_at=freeze_at,
                freeze_norm=freeze_norm,
                pretrained=pretrained,
                local_model_dir=local_model_dir,
                eemsa=eemsa_cfg_rgb,
            )
        if self.ms_in_chs > 0:
            self.ms_backbone = HGNetv2(
                name=self.name,
                in_chs=self.ms_in_chs,
                use_lab=use_lab,
                return_idx=self.return_idx,
                freeze_stem_only=freeze_stem_only,
                freeze_at=freeze_at,
                freeze_norm=freeze_norm,
                pretrained=pretrained,
                local_model_dir=local_model_dir,
                eemsa=eemsa_cfg_ms,
            )

        # Optional: band-separated MS stem + (true) groupwise alignment at C2, before any cross-band fusion.
        ms_band_sep_cfg: Dict[str, Any] = {}
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
            if self.ms_backbone is None or self.ms_in_chs <= 0:
                self.ms_band_sep_enabled = False
            else:
                # Replace MS stem with band-separated embedding + CRGGA alignment.
                c2_in_channels = int(stage_in_channels[0])
                embed_channels = int(
                    ms_band_sep_cfg.get("embed_channels", ms_band_sep_cfg.get("c_emb", 16)) or 16
                )
                embed_use_bn = bool(ms_band_sep_cfg.get("embed_use_bn", True))
                align_cfg_raw = ms_band_sep_cfg.get("align", ms_band_sep_cfg.get("align_cfg", None))
                align_cfg: Dict[str, Any] | None = None
                if isinstance(align_cfg_raw, Mapping):
                    align_cfg = dict(align_cfg_raw)
                elif align_cfg_raw is not None and hasattr(align_cfg_raw, "items"):
                    align_cfg = {k: v for k, v in align_cfg_raw.items()}  # type: ignore[assignment]
                self.ms_band_sep_stem = MSBandSeparatedStemAlign(
                    ms_in_chs=self.ms_in_chs,
                    c2_in_channels=c2_in_channels,
                    embed_channels=embed_channels,
                    embed_use_bn=embed_use_bn,
                    align_cfg=align_cfg,
                )
                # Important for DDP: when we bypass `ms_backbone.stem` in forward, its parameters would be unused
                # (no grad). Mark them as frozen to avoid DDP unused-parameter errors without enabling
                # find_unused_parameters=True.
                try:
                    if hasattr(self.ms_backbone, "stem") and isinstance(self.ms_backbone.stem, nn.Module):
                        for p in self.ms_backbone.stem.parameters():
                            p.requires_grad_(False)
                except Exception:
                    # Best-effort: freezing is only an optimization/stability guard for DDP.
                    pass

        # Optional residual MS stem branch:
        # keep the original MS stem as the main path, then inject a lightweight aligned residual branch.
        ms_residual_stem_cfg: Dict[str, Any] = {}
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
            if self.ms_backbone is None or self.ms_in_chs <= 0:
                self.ms_residual_stem_enabled = False
            else:
                c2_in_channels = int(stage_in_channels[0])
                embed_channels = int(
                    ms_residual_stem_cfg.get("embed_channels", ms_residual_stem_cfg.get("c_emb", 16)) or 16
                )
                embed_use_bn = bool(ms_residual_stem_cfg.get("embed_use_bn", True))
                extractor_type = str(
                    ms_residual_stem_cfg.get("extractor_type", ms_residual_stem_cfg.get("stem_type", "light")) or "light"
                )
                stem_norm_type = str(
                    ms_residual_stem_cfg.get("stem_norm_type", ms_residual_stem_cfg.get("branch_norm", "gn")) or "gn"
                )
                merge_activation = str(ms_residual_stem_cfg.get("merge_activation", "identity") or "identity")
                align_cfg_raw = ms_residual_stem_cfg.get("align", ms_residual_stem_cfg.get("align_cfg", None))
                align_cfg: Dict[str, Any] | None = None
                if isinstance(align_cfg_raw, Mapping):
                    align_cfg = dict(align_cfg_raw)
                elif align_cfg_raw is not None and hasattr(align_cfg_raw, "items"):
                    align_cfg = {k: v for k, v in align_cfg_raw.items()}  # type: ignore[assignment]
                self.ms_residual_stem_branch = MSBandSeparatedStemAlign(
                    ms_in_chs=self.ms_in_chs,
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
                stem_interactive_cfg: Dict[str, Any] = {}
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
                post_align_cfg: Dict[str, Any] = {}
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

        # Build optional MS channel attention modules (per stage, pre/post block).
        self.ms_attn_pre = nn.ModuleDict()
        self.ms_attn_post = nn.ModuleDict()
        if self.ms_channel_attn not in {"", "none", "no", "identity"}:
            supported_pos = {"pre_fuse", "pre_block", "post_block", "both"}
            if self.ms_channel_attn_position not in supported_pos:
                raise ValueError(
                    f"Unsupported ms_channel_attn_position: {ms_channel_attn_position} (supported: {sorted(supported_pos)})"
                )
            for stage_i in self.ms_channel_attn_stage_idx:
                if stage_i < 0 or stage_i >= len(stage_in_channels):
                    raise ValueError(
                        f"ms_channel_attn_stage_idx contains invalid stage idx {stage_i} (num_stages={len(stage_in_channels)})"
                    )
                if self.ms_channel_attn_position in {"pre_fuse", "pre_block", "both"}:
                    self.ms_attn_pre[str(stage_i)] = _build_channel_attn(
                        self.ms_channel_attn,
                        int(stage_in_channels[int(stage_i)]),
                        eca_kernel_size=self.ms_eca_kernel_size,
                        se_reduction=self.ms_se_reduction,
                    )
                if self.ms_channel_attn_position in {"post_block", "both"}:
                    self.ms_attn_post[str(stage_i)] = _build_channel_attn(
                        self.ms_channel_attn,
                        int(stage_out_channels[int(stage_i)]),
                        eca_kernel_size=self.ms_eca_kernel_size,
                        se_reduction=self.ms_se_reduction,
                    )

        # Intra-MS groupwise alignment (reference-band-free) based on GroupwiseDeformableAlign2D.
        # Config is passed from the top-level (Hydra) as `model.backbone_group_align`.
        ms_group_align_cfg: Dict[str, Any] = {}
        if ms_group_align is not None:
            if isinstance(ms_group_align, Mapping):
                ms_group_align_cfg = dict(ms_group_align)
            elif hasattr(ms_group_align, "items"):
                ms_group_align_cfg = {k: v for k, v in ms_group_align.items()}  # type: ignore[assignment]
            else:
                raise TypeError(f"Unsupported ms_group_align type: {type(ms_group_align)}")
        self.ms_group_align_cfg = ms_group_align_cfg

        # Groupwise alignment can run on raw input or stage features (projected groups).
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
        # Hydra/OmegaConf uses ListConfig; treat any non-string iterable as a stage list.
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
        if self.ms_in_chs <= 0 or self.ms_backbone is None:
            self.ms_group_align_enabled = False
            self.ms_group_align_input_enabled = False
        else:
            # Shared defaults for input/stage aligners.
            ref_mode = str(ms_group_align_cfg.get("ref_mode", "spatial_weighted") or "spatial_weighted")
            ref_band_index = ms_group_align_cfg.get("ref_band_index", ms_group_align_cfg.get("ref_channel", None))
            num_iters = int(ms_group_align_cfg.get("num_iters", 1) or 1)
            ref_detach = bool(ms_group_align_cfg.get("ref_detach", False))
            num_keypoints = int(ms_group_align_cfg.get("num_keypoints", 5) or 5)
            offset_scale = float(ms_group_align_cfg.get("offset_scale", 6.0) or 6.0)
            offset_enabled = bool(ms_group_align_cfg.get("offset_enabled", True))
            attention_norm = str(ms_group_align_cfg.get("attention_norm", "sigmoid") or "sigmoid")
            padding_mode = str(ms_group_align_cfg.get("padding_mode", "border") or "border")
            align_corners = bool(ms_group_align_cfg.get("align_corners", True))
            loss_type = str(ms_group_align_cfg.get("loss_type", "infonce") or "infonce")
            loss_downsample = ms_group_align_cfg.get("loss_downsample", None)
            nce_num_patches = int(ms_group_align_cfg.get("nce_num_patches", 64) or 64)
            nce_patch_size = int(ms_group_align_cfg.get("nce_patch_size", 5) or 5)
            nce_tau = float(ms_group_align_cfg.get("nce_tau", 0.2) or 0.2)
            affine_enabled = bool(ms_group_align_cfg.get("affine_enabled", ms_group_align_cfg.get("affine", False)))
            affine_scale = float(ms_group_align_cfg.get("affine_scale", 0.1) or 0.1)
            affine_init_identity = bool(ms_group_align_cfg.get("affine_init_identity", True))
            affine_type = str(ms_group_align_cfg.get("affine_type", "affine") or "affine")
            loss_weight = float(ms_group_align_cfg.get("loss_weight", 0.02) or 0.02)
            loss_offset_weight = float(
                ms_group_align_cfg.get(
                    "loss_offset_weight",
                    ms_group_align_cfg.get("offset_loss_weight", ms_group_align_cfg.get("offset_reg_weight", 0.0)),
                )
                or 0.0
            )
            loss_attn_norm_weight = float(
                ms_group_align_cfg.get(
                    "loss_attn_norm_weight",
                    ms_group_align_cfg.get("attn_norm_weight", ms_group_align_cfg.get("attn_reg_weight", 0.0)),
                )
                or 0.0
            )
            loss_attn_entropy_weight = float(
                ms_group_align_cfg.get(
                    "loss_attn_entropy_weight",
                    ms_group_align_cfg.get("attn_entropy_weight", ms_group_align_cfg.get("attn_entropy_reg_weight", 0.0)),
                )
                or 0.0
            )

            # Input-level: align raw MS bands (treat each band as C=1).
            if self.ms_group_align_input_enabled:
                input_ref_mode = str(ms_group_align_cfg.get("input_ref_mode", ref_mode) or ref_mode)
                input_ref_band_index = ms_group_align_cfg.get(
                    "input_ref_band_index",
                    ms_group_align_cfg.get("input_ref_channel", ref_band_index),
                )
                input_num_iters = int(ms_group_align_cfg.get("input_num_iters", num_iters) or num_iters)
                input_ref_detach = bool(ms_group_align_cfg.get("input_ref_detach", ref_detach))
                input_num_keypoints = int(ms_group_align_cfg.get("input_num_keypoints", num_keypoints) or num_keypoints)
                input_offset_scale = float(ms_group_align_cfg.get("input_offset_scale", offset_scale) or offset_scale)
                input_offset_enabled = bool(
                    ms_group_align_cfg.get("input_offset_enabled", ms_group_align_cfg.get("input_use_offset", offset_enabled))
                )
                input_attention_norm = str(ms_group_align_cfg.get("input_attention_norm", attention_norm) or attention_norm)
                input_padding_mode = str(ms_group_align_cfg.get("input_padding_mode", padding_mode) or padding_mode)
                input_align_corners = bool(ms_group_align_cfg.get("input_align_corners", align_corners))
                input_loss_type = str(ms_group_align_cfg.get("input_loss_type", loss_type) or loss_type)
                input_loss_downsample = ms_group_align_cfg.get("input_loss_downsample", loss_downsample)
                input_nce_num_patches = int(ms_group_align_cfg.get("input_nce_num_patches", nce_num_patches) or nce_num_patches)
                input_nce_patch_size = int(ms_group_align_cfg.get("input_nce_patch_size", nce_patch_size) or nce_patch_size)
                input_nce_tau = float(ms_group_align_cfg.get("input_nce_tau", nce_tau) or nce_tau)
                input_affine_enabled = bool(
                    ms_group_align_cfg.get("input_affine_enabled", ms_group_align_cfg.get("input_affine", affine_enabled))
                )
                input_affine_scale = float(ms_group_align_cfg.get("input_affine_scale", affine_scale) or affine_scale)
                input_affine_init_identity = bool(
                    ms_group_align_cfg.get("input_affine_init_identity", affine_init_identity)
                )
                input_affine_type = str(ms_group_align_cfg.get("input_affine_type", affine_type) or affine_type)
                input_loss_weight = float(ms_group_align_cfg.get("input_loss_weight", loss_weight) or loss_weight)
                input_loss_offset_weight = float(
                    ms_group_align_cfg.get("input_loss_offset_weight", loss_offset_weight) or loss_offset_weight
                )
                input_loss_attn_norm_weight = float(
                    ms_group_align_cfg.get("input_loss_attn_norm_weight", loss_attn_norm_weight) or loss_attn_norm_weight
                )
                input_loss_attn_entropy_weight = float(
                    ms_group_align_cfg.get("input_loss_attn_entropy_weight", loss_attn_entropy_weight)
                    or loss_attn_entropy_weight
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

            # Stage-level: apply projected groupwise alignment on MS feature maps.
            if self.ms_group_align_enabled:
                num_groups = int(ms_group_align_cfg.get("num_groups", ms_group_align_cfg.get("groups", self.ms_in_chs)) or self.ms_in_chs)
                group_channels = int(
                    ms_group_align_cfg.get("group_channels", ms_group_align_cfg.get("proj_channels", 8)) or 8
                )
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

        # Stage-level alignment aligns MS features to a reference (RGB or MS-internal) per stage.
        self.align_enabled = bool(align_enabled)
        self.align_offset_enabled = bool(align_offset_enabled)
        self.align_per_channel_offset = bool(align_per_channel_offset)
        align_position_norm = str(align_position).strip().lower()
        if align_position_norm in {"pre", "pre_block", "before", "before_block"}:
            align_position_norm = "pre_block"
        elif align_position_norm in {"post", "post_block", "after", "after_block"}:
            align_position_norm = "post_block"
        else:
            raise ValueError(
                "Unsupported align_position; expected pre_block/post_block, "
                f"got {align_position} (normalized={align_position_norm})"
            )
        self.align_position = align_position_norm
        align_ref_mode_norm = _normalize_align_ref_mode(align_ref_mode)
        input_ref_raw = str(align_input_ref_mode).strip().lower()
        if input_ref_raw in {"", "none", "same", "auto"}:
            align_input_ref_mode_norm = align_ref_mode_norm
        else:
            align_input_ref_mode_norm = _normalize_align_ref_mode(align_input_ref_mode)
        self.align_ref_mode = align_ref_mode_norm
        self.align_ref_channel = int(align_ref_channel)
        self.align_input_ref_mode = align_input_ref_mode_norm
        self.align_input_ref_channel = int(align_input_ref_channel)
        if self.align_ref_channel < 0:
            raise ValueError(f"align_ref_channel must be >= 0, got {self.align_ref_channel}")
        if self.align_input_ref_channel < 0:
            raise ValueError(f"align_input_ref_channel must be >= 0, got {self.align_input_ref_channel}")
        if self.align_input_ref_mode == "ms_channel" and self.align_input_ref_channel >= self.ms_in_chs:
            raise ValueError(
                "align_input_ref_channel out of range for ms input: "
                f"ref_channel={self.align_input_ref_channel} ms_in_chs={self.ms_in_chs}"
            )
        if align_stage_idx is None:
            # Default align stages follow fusion stages when alignment is enabled.
            align_stage_idx = list(self.fuse_stage_idx) if self.align_enabled else []
        align_stage_idx_norm = [int(i) for i in align_stage_idx]
        # Preserve order while de-duplicating.
        dedup_align: list[int] = []
        for stage_i in align_stage_idx_norm:
            if stage_i not in dedup_align:
                dedup_align.append(stage_i)
        self.align_stage_idx = dedup_align

        # Build per-stage DeformableAlign2D blocks (only when enabled and modalities exist).
        self.ms_aligners = nn.ModuleDict()
        self.ms_ref_weighters = nn.ModuleDict()
        if self.align_enabled:
            if self.ms_backbone is None:
                self.align_enabled = False
            elif self.align_ref_mode == "rgb" and self.rgb_backbone is None:
                self.align_enabled = False
            else:
                align_channels = stage_out_channels if self.align_position == "post_block" else stage_in_channels
                for stage_i in self.align_stage_idx:
                    if stage_i < 0 or stage_i >= len(stage_in_channels):
                        raise ValueError(
                            f"align_stage_idx contains invalid stage idx {stage_i} (num_stages={len(stage_in_channels)})"
                        )
                    in_channels = int(align_channels[int(stage_i)])
                    self.ms_aligners[str(stage_i)] = DeformableAlign2D(
                        in_channels=in_channels,
                        num_keypoints=int(align_num_keypoints),
                        offset_scale=float(align_offset_scale),
                        offset_enabled=bool(self.align_offset_enabled),
                        per_channel_offset=self.align_per_channel_offset,
                        attention_norm=str(align_attention_norm),
                        padding_mode=str(align_padding_mode),
                        align_corners=bool(align_align_corners),
                        loss_type=str(align_loss_type),
                        loss_downsample=align_loss_downsample,
                        nce_num_patches=int(align_nce_num_patches),
                        nce_patch_size=int(align_nce_patch_size),
                        nce_tau=float(align_nce_tau),
                        affine_enabled=bool(align_affine_enabled),
                        affine_scale=float(align_affine_scale),
                        affine_init_identity=bool(align_affine_init_identity),
                        affine_per_channel=bool(align_affine_per_channel),
                        affine_type=str(align_affine_type),
                    )
                    if self.align_ref_mode == "ms_weighted":
                        self.ms_ref_weighters[str(stage_i)] = nn.Conv2d(
                            in_channels, 1, kernel_size=1, bias=True
                        )

        # Input-level alignment (before stem) to reduce gross misalignment between modalities.
        self.align_input_enabled = bool(align_input_enabled)
        self.align_input_proj_mode = "rgb_to_ms"
        self.align_input_rgb_proj: nn.Module | None = None
        self.align_input_ms_proj: nn.Module | None = None
        self.input_aligner: DeformableAlign2D | None = None
        self.input_ref_weight: nn.Module | None = None
        if self.align_input_enabled:
            if self.ms_backbone is None:
                self.align_input_enabled = False
            elif self.align_input_ref_mode == "rgb" and self.rgb_backbone is None:
                self.align_input_enabled = False
            else:
                # Align input channels by optional 1x1 projections (rgb<->ms) before DeformableAlign2D.
                if self.align_input_ref_mode == "rgb":
                    proj_mode = str(align_input_proj).strip().lower()
                    if proj_mode in {"rgb", "rgb_to_ms", "rgb2ms"}:
                        self.align_input_proj_mode = "rgb_to_ms"
                        target_channels = self.ms_in_chs
                        if self.rgb_in_chs != target_channels:
                            self.align_input_rgb_proj = nn.Conv2d(
                                self.rgb_in_chs,
                                target_channels,
                                kernel_size=1,
                                bias=False,
                            )
                        align_in_channels = int(target_channels)
                    elif proj_mode in {"ms", "ms_to_rgb", "ms2rgb"}:
                        self.align_input_proj_mode = "ms_to_rgb"
                        target_channels = self.rgb_in_chs
                        if self.ms_in_chs != target_channels:
                            self.align_input_ms_proj = nn.Conv2d(
                                self.ms_in_chs,
                                target_channels,
                                kernel_size=1,
                                bias=False,
                            )
                        align_in_channels = int(target_channels)
                    elif proj_mode in {"none", "identity"}:
                        if self.rgb_in_chs != self.ms_in_chs:
                            raise ValueError(
                                "align_input_proj=none requires rgb/ms channels to match, "
                                f"got rgb={self.rgb_in_chs} ms={self.ms_in_chs}"
                            )
                        self.align_input_proj_mode = "none"
                        align_in_channels = int(self.rgb_in_chs)
                    else:
                        raise ValueError(f"Unsupported align_input_proj={align_input_proj}")
                else:
                    self.align_input_proj_mode = "ms_internal"
                    align_in_channels = int(self.ms_in_chs)

                if align_input_per_channel_offset and align_in_channels != int(self.ms_in_chs):
                    raise ValueError(
                        "align_input_per_channel_offset requires align_in_channels == ms_in_chs "
                        f"(align_in_channels={align_in_channels}, ms_in_chs={self.ms_in_chs})"
                    )

                self.input_aligner = DeformableAlign2D(
                    in_channels=align_in_channels,
                    num_keypoints=int(align_input_num_keypoints),
                    offset_scale=float(align_input_offset_scale),
                    offset_enabled=bool(align_input_offset_enabled),
                    per_channel_offset=bool(align_input_per_channel_offset),
                    attention_norm=str(align_input_attention_norm),
                    padding_mode=str(align_input_padding_mode),
                    align_corners=bool(align_input_align_corners),
                    loss_type=str(align_input_loss_type),
                    loss_downsample=align_input_loss_downsample,
                    nce_num_patches=int(align_input_nce_num_patches),
                    nce_patch_size=int(align_input_nce_patch_size),
                    nce_tau=float(align_input_nce_tau),
                    affine_enabled=bool(align_input_affine_enabled),
                    affine_scale=float(align_input_affine_scale),
                    affine_init_identity=bool(align_input_affine_init_identity),
                    affine_per_channel=bool(align_input_affine_per_channel),
                    affine_type=str(align_input_affine_type),
                )
                if self.align_input_ref_mode == "ms_weighted":
                    self.input_ref_weight = nn.Conv2d(
                        int(self.ms_in_chs), 1, kernel_size=1, bias=True
                    )

        # Cross-modal fusion modules instantiated per stage in fuse_stage_idx.
        self.fusions = nn.ModuleDict()
        kv_stride_map: Dict[int, int] = {}
        c2former_groups_map: Dict[int, int] = {}
        c2former_stride_map: Dict[int, int] = {}
        c2former_offset_range_map: Dict[int, float] = {}
        c2former_no_offset_map: Dict[int, bool] = {}
        c2former_kernel_map: Dict[int, int] = {}
        c2former_attn_drop = fusion_c2former_attn_drop
        c2former_proj_drop = fusion_c2former_proj_drop
        c2former_padding_mode = fusion_c2former_padding_mode
        c2former_align_corners = fusion_c2former_align_corners
        c2former_offset_on = str(fusion_c2former_offset_on).strip().lower()
        c2former_global_kv = bool(fusion_c2former_global_kv)
        c2former_global_vert_anchors = int(fusion_c2former_global_vert_anchors)
        c2former_global_horz_anchors = int(fusion_c2former_global_horz_anchors)
        c2former_use_pos_encoding = bool(fusion_c2former_use_pos_encoding)
        c2former_pos_temperature = float(fusion_c2former_pos_temperature)
        if self.fusion_type == "coattention":
            kv_stride_map = _normalize_fuse_kv_stride(fusion_kv_stride, fuse_stage_idx=self.fuse_stage_idx)
        if self.fusion_type == "c2former":
            if c2former_attn_drop is None:
                c2former_attn_drop = fusion_dropout
            if c2former_proj_drop is None:
                c2former_proj_drop = fusion_dropout
            if c2former_padding_mode is None:
                c2former_padding_mode = "zeros"
            if c2former_align_corners is None:
                c2former_align_corners = True
            if c2former_offset_on not in {"rgb", "ms"}:
                raise ValueError(f"Unsupported fusion_c2former_offset_on={fusion_c2former_offset_on}")
            if c2former_global_kv and (c2former_global_vert_anchors <= 0 or c2former_global_horz_anchors <= 0):
                raise ValueError(
                    "fusion_c2former_global_vert_anchors/fusion_c2former_global_horz_anchors must be > 0 "
                    f"when fusion_c2former_global_kv=True, got vert={c2former_global_vert_anchors} "
                    f"horz={c2former_global_horz_anchors}"
                )
            if c2former_pos_temperature <= 0:
                raise ValueError(f"fusion_c2former_pos_temperature must be > 0, got {c2former_pos_temperature}")

            default_kernel = {0: 9, 1: 7, 2: 5, 3: 3}
            c2former_groups_map = _normalize_stage_param(
                fusion_c2former_groups,
                fuse_stage_idx=self.fuse_stage_idx,
                default=1,
                param_name="fusion_c2former_groups",
                cast=int,
            )
            c2former_stride_map = _normalize_stage_param(
                fusion_c2former_cca_stride,
                fuse_stage_idx=self.fuse_stage_idx,
                default=3,
                param_name="fusion_c2former_cca_stride",
                cast=int,
            )
            c2former_offset_range_map = _normalize_stage_param(
                fusion_c2former_offset_range_factor,
                fuse_stage_idx=self.fuse_stage_idx,
                default=2.0,
                param_name="fusion_c2former_offset_range_factor",
                cast=float,
            )
            c2former_no_offset_map = _normalize_stage_param(
                fusion_c2former_no_offset,
                fuse_stage_idx=self.fuse_stage_idx,
                default=False,
                param_name="fusion_c2former_no_offset",
                cast=_parse_bool,
            )
            c2former_kernel_map = _normalize_stage_param(
                fusion_c2former_offset_kernel_size,
                fuse_stage_idx=self.fuse_stage_idx,
                default=default_kernel,
                param_name="fusion_c2former_offset_kernel_size",
                cast=int,
            )

        if self.fusion_type != "none":
            for stage_i in self.fuse_stage_idx:
                if self.fusion_position in {"pre_downsample", "pre_block"}:
                    # Fusion is applied after per-stage downsample and before per-stage blocks,
                    # so the channel dimension matches the stage input channels (not output channels).
                    in_channels = int(stage_in_channels[stage_i])
                else:
                    # Fusion is applied at stage end (after per-stage blocks), matching stage output channels.
                    in_channels = int(stage_out_channels[stage_i])
                d_model = int(fusion_d_model)
                if d_model <= 0:
                    d_model = in_channels

                if self.fusion_type == "coattention":
                    self.fusions[str(stage_i)] = CoAttention2D(
                        in_channels=in_channels,
                        d_model=d_model,
                        nhead=int(fusion_nhead),
                        kv_stride=int(kv_stride_map[int(stage_i)]),
                        dropout=float(fusion_dropout),
                        alpha_init=float(fusion_alpha_init),
                    )
                elif self.fusion_type == "gpt":
                    self.fusions[str(stage_i)] = GPTFusion2D(
                        in_channels=in_channels,
                        d_model=d_model,
                        nhead=int(fusion_nhead),
                        block_exp=int(fusion_block_exp),
                        n_layer=int(fusion_n_layer),
                        vert_anchors=int(fusion_vert_anchors),
                        horz_anchors=int(fusion_horz_anchors),
                        dropout=float(fusion_dropout),
                        writeback_merge=str(fusion_writeback_merge),
                        alpha_init=float(fusion_alpha_init),
                    )
                elif self.fusion_type == "mrt":
                    self.fusions[str(stage_i)] = MRTCrossSpectrumFusion2D(
                        in_channels=in_channels,
                        d_model=d_model,
                        num_stages=int(fusion_num_stages),
                        writeback_merge=str(fusion_writeback_merge),
                        alpha_init=float(fusion_alpha_init),
                        use_pos_encoding=bool(fusion_use_pos_encoding),
                    )
                elif self.fusion_type == "c2former":
                    key = int(stage_i)
                    self.fusions[str(stage_i)] = ASMFusion2D(
                        in_channels=in_channels,
                        d_model=d_model,
                        nhead=int(fusion_nhead),
                        groups=int(c2former_groups_map[key]),
                        cca_stride=int(c2former_stride_map[key]),
                        offset_range_factor=float(c2former_offset_range_map[key]),
                        no_offset=bool(c2former_no_offset_map[key]),
                        attn_drop=float(c2former_attn_drop),
                        proj_drop=float(c2former_proj_drop),
                        offset_kernel_size=int(c2former_kernel_map[key]),
                        padding_mode=str(c2former_padding_mode),
                        align_corners=bool(c2former_align_corners),
                        offset_on=str(c2former_offset_on),
                        global_kv=bool(c2former_global_kv),
                        global_vert_anchors=int(c2former_global_vert_anchors),
                        global_horz_anchors=int(c2former_global_horz_anchors),
                        use_pos_encoding=bool(c2former_use_pos_encoding),
                        pos_temperature=float(c2former_pos_temperature),
                    )
                else:
                    raise RuntimeError(f"Unsupported fusion_type: {self.fusion_type}")

        # Optional output-side fusion (applied on returned stages before output merge).
        self.output_fusions = nn.ModuleDict()
        if self.output_fusion_type == "mrt_corr":
            for stage_i in self.return_idx:
                ch = int(stage_out_channels[int(stage_i)])
                d_model = int(output_mrt_d_model)
                if d_model <= 0:
                    d_model = ch
                self.output_fusions[str(stage_i)] = MRTCrossSpectrumCorrFusion2D(
                    in_channels=ch,
                    d_model=d_model,
                    num_stages=int(output_mrt_num_stages),
                    fused_channels=int(output_mrt_fused_channels),
                    norm=str(output_mrt_norm),
                    use_pos_encoding=bool(output_mrt_use_pos_encoding),
                )
        elif self.output_fusion_type == "cft":
            for stage_i in self.return_idx:
                ch = int(stage_out_channels[int(stage_i)])
                d_model = int(output_cft_d_model)
                if d_model <= 0:
                    d_model = ch
                self.output_fusions[str(stage_i)] = GPTFusion2D(
                    in_channels=ch,
                    d_model=d_model,
                    nhead=int(output_cft_nhead),
                    block_exp=int(output_cft_block_exp),
                    n_layer=int(output_cft_n_layer),
                    vert_anchors=int(output_cft_vert_anchors),
                    horz_anchors=int(output_cft_horz_anchors),
                    dropout=float(output_cft_dropout),
                    writeback_merge=str(output_cft_writeback_merge),
                    alpha_init=float(output_cft_alpha_init),
                )

        # Output merge helpers for concat1x1 and wavg modes.
        self.concat_projs = nn.ModuleDict()
        for stage_i in self.return_idx:
            if self.output_merge_map[int(stage_i)] != "concat1x1":
                continue
            in_channels = int(stage_out_channels[stage_i])
            self.concat_projs[str(stage_i)] = nn.Conv2d(2 * in_channels, in_channels, kernel_size=1, bias=False)

        # Learnable gates for weighted average output merge (wavg).
        self.merge_gates = nn.ParameterDict()
        # Learnable weighted average: out = (1-w)*rgb + w*ms, where w=sigmoid(gate) in (0,1).
        # Initialize gate=0 -> w=0.5 to match plain avg at start.
        for stage_i in self.return_idx:
            if self.output_merge_map[int(stage_i)] != "wavg":
                continue
            self.merge_gates[str(stage_i)] = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def _merge_outputs(self, rgb: torch.Tensor | None, ms: torch.Tensor | None, *, stage_i: int) -> torch.Tensor:
        # Merge per-stage RGB/MS features into a single tensor for the encoder.
        if rgb is None:
            assert ms is not None
            return ms
        if ms is None:
            return rgb
        mode = self.output_merge_map[int(stage_i)]
        if mode == "wavg":
            key = str(stage_i)
            if key not in self.merge_gates:
                raise RuntimeError(f"Missing wavg gate for stage idx {stage_i}")
            w = torch.sigmoid(self.merge_gates[key]).to(dtype=rgb.dtype, device=rgb.device)
            return rgb * (1.0 - w) + ms * w
        if mode == "add":
            return rgb + ms
        if mode == "concat1x1":
            key = str(stage_i)
            if key not in self.concat_projs:
                raise RuntimeError(f"Missing concat1x1 projection for stage idx {stage_i}")
            return self.concat_projs[key](torch.cat([rgb, ms], dim=1))
        raise ValueError(f"Unsupported output_merge for stage idx {stage_i}: {mode}")

    @staticmethod
    def _apply_eemsa_stem(feat: torch.Tensor, backbone: HGNetv2 | None) -> torch.Tensor:
        if backbone is None:
            return feat
        eemsa_stem = getattr(backbone, "eemsa_stem", None)
        if eemsa_stem is None:
            return feat
        return eemsa_stem(feat)

    @staticmethod
    def _apply_eemsa_stage(feat: torch.Tensor, backbone: HGNetv2 | None, idx: int) -> torch.Tensor:
        if backbone is None:
            return feat
        modules = getattr(backbone, "eemsa_stage_modules", None)
        if modules is None:
            return feat
        key = str(idx)
        if key not in modules:
            return feat
        return modules[key](feat)  # type: ignore[index]

    def _build_ms_internal_ref(
        self,
        ms: torch.Tensor,
        *,
        mode: str,
        channel: int,
        weight_module: nn.Module | None = None,
    ) -> torch.Tensor:
        # Build an MS reference feature and broadcast it across MS channels.
        if mode == "ms_mean":
            ref = ms.mean(dim=1, keepdim=True)
        elif mode == "ms_channel":
            idx = int(channel)
            if idx < 0 or idx >= ms.shape[1]:
                raise ValueError(
                    "align_ref_channel out of range for ms tensor: "
                    f"ref_channel={idx} ms_channels={ms.shape[1]}"
                )
            ref = ms[:, idx : idx + 1]
        elif mode == "ms_weighted":
            if weight_module is None:
                raise RuntimeError("Missing ms_weighted reference module")
            ref = weight_module(ms)
        else:
            raise ValueError(f"Unsupported ms internal ref mode: {mode}")
        return ref.expand(-1, ms.shape[1], -1, -1).contiguous()

    @staticmethod
    def _safe_normalize_attention(attn: torch.Tensor) -> torch.Tensor:
        p = attn.clamp_min(0.0)
        return p / p.sum(dim=1, keepdim=True).clamp_min(1e-6)

    def _apply_ms_residual_post_align(
        self,
        ref: torch.Tensor,
        residual: torch.Tensor,
        aux_losses: dict[str, torch.Tensor],
        aux_count: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], int]:
        if not self.ms_residual_post_align_enabled or self.ms_residual_post_aligner is None:
            return residual, aux_losses, aux_count

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

            loss_added = False
            if self.ms_residual_post_align_loss_weight > 0 and "loss_deform_align" in loss_dict:
                aux_losses["loss_deform_align"] = aux_losses.get("loss_deform_align", 0.0) + (
                    loss_dict["loss_deform_align"] * self.ms_residual_post_align_loss_weight
                )
                loss_added = True

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
                    loss_added = True

                if self.ms_residual_post_align_loss_attn_norm_weight > 0:
                    attn_sum = attn_weights.sum(dim=1)
                    aux_losses["loss_deform_attn"] = aux_losses.get("loss_deform_attn", 0.0) + (
                        ((attn_sum - 1.0) ** 2).mean() * self.ms_residual_post_align_loss_attn_norm_weight
                    )
                    loss_added = True

                if self.ms_residual_post_align_loss_attn_entropy_weight > 0:
                    p = self._safe_normalize_attention(attn_weights)
                    ent = -(p * torch.log(p.clamp_min(1e-8))).sum(dim=1).mean()
                    aux_losses["loss_deform_attn_entropy"] = aux_losses.get("loss_deform_attn_entropy", 0.0) + (
                        ent * self.ms_residual_post_align_loss_attn_entropy_weight
                    )
                    loss_added = True

            if loss_added:
                aux_count += 1
            return residual_aligned, aux_losses, aux_count

        return aligner(ref_pred, residual), aux_losses, aux_count

    def _apply_ms_residual_stem_interactive(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        if not self.ms_residual_stem_interactive_enabled or self.ms_residual_stem_interactive is None:
            return x
        return self.ms_residual_stem_interactive(x, residual)

    def _apply_input_alignment(
        self,
        rgb: torch.Tensor | None,
        ms: torch.Tensor | None,
        aux_losses: dict[str, torch.Tensor],
        aux_count: int,
    ) -> tuple[torch.Tensor | None, dict[str, torch.Tensor], int]:
        # Input-level alignment; when training, collect aux losses for criterion.
        if not self.align_input_enabled or ms is None:
            return ms, aux_losses, aux_count
        if self.input_aligner is None:
            raise RuntimeError("Missing input alignment module")

        ref: torch.Tensor
        ms_src = ms
        if self.align_input_ref_mode == "rgb":
            if rgb is None:
                return ms, aux_losses, aux_count
            rgb_ref = rgb
            if self.align_input_proj_mode == "rgb_to_ms":
                if self.align_input_rgb_proj is not None:
                    rgb_ref = self.align_input_rgb_proj(rgb_ref)
            elif self.align_input_proj_mode == "ms_to_rgb":
                if self.align_input_ms_proj is not None:
                    ms_src = self.align_input_ms_proj(ms_src)
            ref = rgb_ref
        else:
            weight_module = None
            if self.align_input_ref_mode == "ms_weighted":
                if self.input_ref_weight is None:
                    raise RuntimeError("Missing ms_weighted input ref module")
                weight_module = self.input_ref_weight
            ref = self._build_ms_internal_ref(
                ms,
                mode=self.align_input_ref_mode,
                channel=self.align_input_ref_channel,
                weight_module=weight_module,
            )

        aligner = self.input_aligner
        if self.training:
            pred = aligner.predict(ref, ms_src)
            if aligner.affine_enabled:
                offset_x, offset_y, attn_weights, affine_theta = pred
            else:
                offset_x, offset_y, attn_weights = pred
                affine_theta = None

            ms_aligned_loss, _, _ = aligner.deform_with_attention(
                ms_src,
                offset_x=offset_x,
                offset_y=offset_y,
                attention_weights=attn_weights,
                affine_theta=affine_theta,
            )
            if ms_src is ms:
                ms_aligned = ms_aligned_loss
            else:
                ms_aligned, _, _ = aligner.deform_with_attention(
                    ms,
                    offset_x=offset_x,
                    offset_y=offset_y,
                    attention_weights=attn_weights,
                    affine_theta=affine_theta,
                )
            stage_losses = aligner.loss_calculate(
                ref,
                offset_x,
                offset_y,
                ms_aligned_loss,
                attn_weights,
                affine_theta=affine_theta,
            )
            ms = ms_aligned
            for lk, lv in stage_losses.items():
                if torch.is_tensor(lv):
                    aux_losses[lk] = aux_losses.get(lk, 0.0) + lv
            aux_count += 1
        else:
            pred = aligner.predict(ref, ms_src)
            if aligner.affine_enabled:
                offset_x, offset_y, attn_weights, affine_theta = pred
            else:
                offset_x, offset_y, attn_weights = pred
                affine_theta = None
            ms, _, _ = aligner.deform_with_attention(
                ms,
                offset_x=offset_x,
                offset_y=offset_y,
                attention_weights=attn_weights,
                affine_theta=affine_theta,
            )
        return ms, aux_losses, aux_count

    def _apply_ms_group_input_alignment(
        self,
        ms: torch.Tensor | None,
        aux_losses: dict[str, torch.Tensor],
        aux_count: int,
    ) -> tuple[torch.Tensor | None, dict[str, torch.Tensor], int]:
        """
        Align raw MS bands (B,7,H,W) inside the MS modality without picking a fixed reference band.

        Implemented via GroupwiseDeformableAlign2D on (B,N,1,H,W).
        """
        if not self.ms_group_align_input_enabled or ms is None:
            return ms, aux_losses, aux_count
        if self.ms_group_input_aligner is None:
            raise RuntimeError("Missing ms_group_input_aligner")
        x = ms.unsqueeze(2)  # (B,N,1,H,W)
        out = self.ms_group_input_aligner(x)
        if self.training and isinstance(out, tuple) and len(out) == 2:
            x_aligned, stage_losses = out
            for lk, lv in stage_losses.items():
                if torch.is_tensor(lv):
                    aux_losses[lk] = aux_losses.get(lk, 0.0) + lv
            aux_count += 1
        else:
            x_aligned = out
        ms = x_aligned.squeeze(2)
        return ms, aux_losses, aux_count

    def _apply_ms_group_alignment(
        self,
        idx: int,
        ms: torch.Tensor | None,
        aux_losses: dict[str, torch.Tensor],
        aux_count: int,
    ) -> tuple[torch.Tensor | None, dict[str, torch.Tensor], int]:
        """
        Stage-level projected groupwise alignment on MS feature maps (B,C,H,W).
        """
        if not self.ms_group_align_enabled or ms is None or idx not in self.ms_group_align_stage_idx:
            return ms, aux_losses, aux_count
        key = str(idx)
        if key not in self.ms_group_aligners:
            raise RuntimeError(f"Missing ms_group_align module for stage idx {idx}")
        aligner = self.ms_group_aligners[key]
        if self.training:
            out = aligner(ms)
            if isinstance(out, tuple) and len(out) == 2:
                ms_aligned, stage_losses = out
                for lk, lv in stage_losses.items():
                    if torch.is_tensor(lv):
                        aux_losses[lk] = aux_losses.get(lk, 0.0) + lv
                aux_count += 1
                ms = ms_aligned
            else:
                ms = out  # pragma: no cover
        else:
            out = aligner(ms)
            ms = out[0] if isinstance(out, tuple) else out
        return ms, aux_losses, aux_count

    def _apply_alignment(
        self,
        idx: int,
        rgb: torch.Tensor | None,
        ms: torch.Tensor | None,
        aux_losses: dict[str, torch.Tensor],
        aux_count: int,
    ) -> tuple[torch.Tensor | None, dict[str, torch.Tensor], int]:
        # Stage-level MS alignment to the chosen reference (RGB or MS-internal).
        if not self.align_enabled or ms is None or idx not in self.align_stage_idx:
            return ms, aux_losses, aux_count
        key = str(idx)
        if key not in self.ms_aligners:
            raise RuntimeError(f"Missing align module for stage idx {idx}")
        if self.align_ref_mode == "rgb":
            if rgb is None:
                return ms, aux_losses, aux_count
            ref = rgb
        else:
            weight_module = None
            if self.align_ref_mode == "ms_weighted":
                if key not in self.ms_ref_weighters:
                    raise RuntimeError(f"Missing ms_weighted ref module for stage idx {idx}")
                weight_module = self.ms_ref_weighters[key]
            ref = self._build_ms_internal_ref(
                ms,
                mode=self.align_ref_mode,
                channel=self.align_ref_channel,
                weight_module=weight_module,
            )
        aligner = self.ms_aligners[key]
        if self.training:
            pred = aligner.predict(ref, ms)
            if aligner.affine_enabled:
                offset_x, offset_y, attn_weights, affine_theta = pred
                ms_aligned, _, attn_weights = aligner.deform_with_attention(
                    ms,
                    offset_x=offset_x,
                    offset_y=offset_y,
                    attention_weights=attn_weights,
                    affine_theta=affine_theta,
                )
                stage_losses = aligner.loss_calculate(
                    ref, offset_x, offset_y, ms_aligned, attn_weights, affine_theta=affine_theta
                )
            else:
                offset_x, offset_y, attn_weights = pred
                ms_aligned, _, attn_weights = aligner.deform_with_attention(
                    ms, offset_x=offset_x, offset_y=offset_y, attention_weights=attn_weights
                )
                stage_losses = aligner.loss_calculate(ref, offset_x, offset_y, ms_aligned, attn_weights)
            ms = ms_aligned
            for lk, lv in stage_losses.items():
                if torch.is_tensor(lv):
                    aux_losses[lk] = aux_losses.get(lk, 0.0) + lv
            aux_count += 1
        else:
            ms = aligner(ref, ms)
        return ms, aux_losses, aux_count

    def forward(self, x: torch.Tensor):
        # Input is channel-stacked [RGB | MS]; split inside the backbone.
        if x.ndim != 4:
            raise ValueError(f"HGNetv2DualStream expects BCHW tensor, got {x.shape}")

        rgb: torch.Tensor | None = None
        ms: torch.Tensor | None = None
        if self.rgb_in_chs > 0:
            rgb = x[:, : self.rgb_in_chs]
        if self.ms_in_chs > 0:
            start = self.rgb_in_chs
            ms = x[:, start : start + self.ms_in_chs]

        aux_losses: dict[str, torch.Tensor] = {}
        aux_count = 0
        # Optional input-level alignment on raw MS bands before stem.
        if self.ms_group_align_input_enabled:
            ms, aux_losses, aux_count = self._apply_ms_group_input_alignment(ms, aux_losses, aux_count)
        if self.align_input_enabled:
            ms, aux_losses, aux_count = self._apply_input_alignment(rgb, ms, aux_losses, aux_count)

        # Per-modality stem (MS can be replaced by MSBandSeparatedStemAlign).
        if self.rgb_backbone is not None:
            assert rgb is not None
            rgb = self.rgb_backbone.stem(rgb)
            rgb = self._apply_eemsa_stem(rgb, self.rgb_backbone)
        if self.ms_backbone is not None:
            assert ms is not None
            ms_stem_input = ms
            if self.ms_band_sep_enabled and self.ms_band_sep_stem is not None:
                out = self.ms_band_sep_stem(ms_stem_input)
                if self.training and isinstance(out, tuple) and len(out) == 2:
                    ms, stem_losses = out
                    for lk, lv in stem_losses.items():
                        if torch.is_tensor(lv):
                            aux_losses[lk] = aux_losses.get(lk, 0.0) + lv
                    aux_count += 1
                else:
                    ms = out[0] if isinstance(out, tuple) else out
            else:
                ms = self.ms_backbone.stem(ms_stem_input)
                if self.ms_residual_stem_enabled and self.ms_residual_stem_branch is not None and self.ms_residual_scale is not None:
                    out = self.ms_residual_stem_branch(ms_stem_input)
                    if self.training and isinstance(out, tuple) and len(out) == 2:
                        residual, stem_losses = out
                        for lk, lv in stem_losses.items():
                            if torch.is_tensor(lv):
                                aux_losses[lk] = aux_losses.get(lk, 0.0) + lv
                        aux_count += 1
                    else:
                        residual = out[0] if isinstance(out, tuple) else out
                    ms, aux_losses, aux_count = self._apply_ms_residual_post_align(ms, residual, aux_losses, aux_count)
                    ms = self._apply_ms_residual_stem_interactive(ms, residual)
                    if self.ms_residual_fusion_mode == "concat_proj":
                        if self.ms_residual_fuse_proj is None:
                            raise RuntimeError("Missing ms_residual_fuse_proj for concat_proj fusion")
                        delta = self.ms_residual_fuse_proj(torch.cat([ms, residual], dim=1))
                        ms = ms + (self.ms_residual_scale * delta)
                    else:
                        ms = ms + (self.ms_residual_scale * residual)
            ms = self._apply_eemsa_stem(ms, self.ms_backbone)

        ret_stage_idx: list[int] = []
        ret_rgb: list[torch.Tensor | None] = []
        ret_ms: list[torch.Tensor | None] = []
        if self.rgb_backbone is not None:
            num_stages = len(self.rgb_backbone.stages)
        elif self.ms_backbone is not None:
            num_stages = len(self.ms_backbone.stages)
        else:
            raise RuntimeError("At least one backbone must be initialized.")

        for idx in range(num_stages):
            rgb_stage: HG_Stage | None = self.rgb_backbone.stages[idx] if self.rgb_backbone is not None else None
            ms_stage: HG_Stage | None = self.ms_backbone.stages[idx] if self.ms_backbone is not None else None

            # Keep HGNetv2 stage structure intact. Fusion position is configurable:
            # - pre_downsample:      fuse -> downsample -> (align/attn) -> blocks -> (align/attn)
            # - pre_block (default): downsample -> (align/attn) -> fuse -> blocks -> (align/attn)
            # - post_block:          downsample -> (align/attn) -> blocks -> (align/attn) -> fuse
            if (
                self.fusion_position == "pre_downsample"
                and self.fusion_type != "none"
                and rgb is not None
                and ms is not None
                and idx in self.fuse_stage_idx
            ):
                key = str(idx)
                if key not in self.fusions:
                    raise RuntimeError(f"Missing fusion module for stage idx {idx}")
                rgb, ms = self.fusions[key](rgb, ms)

            if rgb_stage is not None:
                rgb = rgb_stage.downsample(rgb)  # type: ignore[arg-type]
            if ms_stage is not None:
                ms = ms_stage.downsample(ms)  # type: ignore[arg-type]
                if self.ms_group_align_position == "pre_block":
                    ms, aux_losses, aux_count = self._apply_ms_group_alignment(idx, ms, aux_losses, aux_count)
                if self.align_position == "pre_block":
                    ms, aux_losses, aux_count = self._apply_alignment(idx, rgb, ms, aux_losses, aux_count)

                key = str(idx)
                if ms is not None and key in self.ms_attn_pre:
                    ms = self.ms_attn_pre[key](ms)

            if (
                self.fusion_position == "pre_block"
                and self.fusion_type != "none"
                and rgb is not None
                and ms is not None
                and idx in self.fuse_stage_idx
            ):
                key = str(idx)
                if key not in self.fusions:
                    raise RuntimeError(f"Missing fusion module for stage idx {idx}")
                rgb, ms = self.fusions[key](rgb, ms)

            if rgb_stage is not None:
                rgb = rgb_stage.blocks(rgb)  # type: ignore[arg-type]
                rgb = self._apply_eemsa_stage(rgb, self.rgb_backbone, idx)
            if ms_stage is not None:
                ms = ms_stage.blocks(ms)  # type: ignore[arg-type]
                if self.ms_group_align_position == "post_block":
                    ms, aux_losses, aux_count = self._apply_ms_group_alignment(idx, ms, aux_losses, aux_count)
                if self.align_position == "post_block":
                    ms, aux_losses, aux_count = self._apply_alignment(idx, rgb, ms, aux_losses, aux_count)
                key = str(idx)
                if ms is not None and key in self.ms_attn_post:
                    ms = self.ms_attn_post[key](ms)
                ms = self._apply_eemsa_stage(ms, self.ms_backbone, idx)

            if (
                self.fusion_position == "post_block"
                and self.fusion_type != "none"
                and rgb is not None
                and ms is not None
                and idx in self.fuse_stage_idx
            ):
                key = str(idx)
                if key not in self.fusions:
                    raise RuntimeError(f"Missing fusion module for stage idx {idx}")
                rgb, ms = self.fusions[key](rgb, ms)

            if idx in self.return_idx:
                ret_stage_idx.append(int(idx))
                ret_rgb.append(rgb)
                ret_ms.append(ms)

        # Output fusion (optional) + output merge (wavg/add/concat1x1) per return stage.
        outs: list[torch.Tensor] = []
        for stage_i, rgb_feat, ms_feat in zip(ret_stage_idx, ret_rgb, ret_ms):
            if self.output_fusion_type == "mrt_corr":
                if rgb_feat is None:
                    assert ms_feat is not None
                    outs.append(ms_feat)
                    continue
                if ms_feat is None:
                    outs.append(rgb_feat)
                    continue
                key = str(stage_i)
                if key not in self.output_fusions:
                    raise RuntimeError(f"Missing output fusion module for stage idx {stage_i}")
                outs.append(self.output_fusions[key](rgb_feat, ms_feat))
            elif self.output_fusion_type == "cft":
                if rgb_feat is None:
                    assert ms_feat is not None
                    outs.append(ms_feat)
                    continue
                if ms_feat is None:
                    outs.append(rgb_feat)
                    continue
                key = str(stage_i)
                if key not in self.output_fusions:
                    raise RuntimeError(f"Missing output fusion module for stage idx {stage_i}")
                rgb_feat, ms_feat = self.output_fusions[key](rgb_feat, ms_feat)
                outs.append(self._merge_outputs(rgb_feat, ms_feat, stage_i=stage_i))
            else:
                outs.append(self._merge_outputs(rgb_feat, ms_feat, stage_i=stage_i))

        if self.training and aux_losses and aux_count > 0:
            # Keep the loss scale stable when enabling multiple stages.
            aux_losses = {k: v / float(aux_count) for k, v in aux_losses.items() if torch.is_tensor(v)}
            return outs, aux_losses
        return outs


@register()
class HGNetv2DualStreamCoAttn(HGNetv2DualStream):
    """
    Backward-compatible alias: use `HGNetv2DualStream(fusion_type='coattention', ...)` instead.
    """

    def __init__(
        self,
        name: str,
        *,
        rgb_in_chs: int = 3,
        ms_in_chs: int = 7,
        use_lab: bool = False,
        return_idx: List[int] = [1, 2, 3],
        fuse_stage_idx: List[int] | None = None,
        output_merge: str | Mapping[str, str] | Mapping[int, str] = "avg",
        ms_channel_attn: str = "none",
        ms_channel_attn_position: str = "pre_fuse",
        ms_channel_attn_stage_idx: List[int] | None = None,
        ms_eca_kernel_size: int | None = None,
        ms_se_reduction: int = 16,
        fusion_d_model: int = 128,
        fusion_nhead: int = 8,
        fusion_dropout: float = 0.0,
        fusion_alpha_init: float = 0.0,
        fusion_kv_stride: int | Sequence[int] | Mapping[str, int] | Mapping[int, int] | None = None,
        freeze_stem_only: bool = True,
        freeze_at: int = 0,
        freeze_norm: bool = True,
        pretrained: bool = True,
        local_model_dir: str = "weight/hgnetv2/",
    ) -> None:
        super().__init__(
            name=name,
            fusion_type="coattention",
            rgb_in_chs=rgb_in_chs,
            ms_in_chs=ms_in_chs,
            use_lab=use_lab,
            return_idx=return_idx,
            fuse_stage_idx=fuse_stage_idx,
            output_merge=output_merge,
            ms_channel_attn=ms_channel_attn,
            ms_channel_attn_position=ms_channel_attn_position,
            ms_channel_attn_stage_idx=ms_channel_attn_stage_idx,
            ms_eca_kernel_size=ms_eca_kernel_size,
            ms_se_reduction=ms_se_reduction,
            fusion_d_model=fusion_d_model,
            fusion_nhead=fusion_nhead,
            fusion_dropout=fusion_dropout,
            fusion_alpha_init=fusion_alpha_init,
            fusion_kv_stride=fusion_kv_stride,
            freeze_stem_only=freeze_stem_only,
            freeze_at=freeze_at,
            freeze_norm=freeze_norm,
            pretrained=pretrained,
            local_model_dir=local_model_dir,
        )


@register()
class HGNetv2DualStreamGPT(HGNetv2DualStream):
    """
    Backward-compatible alias: use `HGNetv2DualStream(fusion_type='gpt', ...)` instead.
    """

    def __init__(
        self,
        name: str,
        *,
        rgb_in_chs: int = 3,
        ms_in_chs: int = 7,
        use_lab: bool = False,
        return_idx: List[int] = [1, 2, 3],
        fuse_stage_idx: List[int] | None = None,
        output_merge: str | Mapping[str, str] | Mapping[int, str] = "avg",
        ms_channel_attn: str = "none",
        ms_channel_attn_position: str = "pre_fuse",
        ms_channel_attn_stage_idx: List[int] | None = None,
        ms_eca_kernel_size: int | None = None,
        ms_se_reduction: int = 16,
        fusion_d_model: int = 128,
        fusion_nhead: int = 8,
        fusion_block_exp: int = 4,
        fusion_n_layer: int = 8,
        fusion_vert_anchors: int = 8,
        fusion_horz_anchors: int = 8,
        fusion_dropout: float = 0.1,
        fusion_alpha_init: float = 0.0,
        fusion_writeback_merge: str = "add",
        freeze_stem_only: bool = True,
        freeze_at: int = 0,
        freeze_norm: bool = True,
        pretrained: bool = True,
        local_model_dir: str = "weight/hgnetv2/",
    ) -> None:
        super().__init__(
            name=name,
            fusion_type="gpt",
            rgb_in_chs=rgb_in_chs,
            ms_in_chs=ms_in_chs,
            use_lab=use_lab,
            return_idx=return_idx,
            fuse_stage_idx=fuse_stage_idx,
            output_merge=output_merge,
            ms_channel_attn=ms_channel_attn,
            ms_channel_attn_position=ms_channel_attn_position,
            ms_channel_attn_stage_idx=ms_channel_attn_stage_idx,
            ms_eca_kernel_size=ms_eca_kernel_size,
            ms_se_reduction=ms_se_reduction,
            fusion_d_model=fusion_d_model,
            fusion_nhead=fusion_nhead,
            fusion_block_exp=fusion_block_exp,
            fusion_n_layer=fusion_n_layer,
            fusion_vert_anchors=fusion_vert_anchors,
            fusion_horz_anchors=fusion_horz_anchors,
            fusion_dropout=fusion_dropout,
            fusion_alpha_init=fusion_alpha_init,
            fusion_writeback_merge=fusion_writeback_merge,
            freeze_stem_only=freeze_stem_only,
            freeze_at=freeze_at,
            freeze_norm=freeze_norm,
            pretrained=pretrained,
            local_model_dir=local_model_dir,
        )
