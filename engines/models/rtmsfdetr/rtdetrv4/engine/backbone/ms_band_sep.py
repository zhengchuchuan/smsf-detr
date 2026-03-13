from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import torch
import torch.nn as nn

from .fixed_band_cmda import FixedBandCMDA
from .group_deform_align import GroupwiseDeformableAlign2D

__all__ = ["MSBandSeparatedStemAlign"]


def _gn_groups(num_channels: int, *, max_groups: int = 8, min_channels_per_group: int = 4) -> int:
    """
    Pick a GroupNorm group count that divides num_channels and avoids tiny groups.

    Defaults are tuned for small stems (e.g., C=16/32) used in MSBandSeparatedStemAlign.
    """
    c = int(num_channels)
    if c <= 0:
        raise ValueError(f"num_channels must be > 0, got {num_channels}")
    g = min(int(max_groups), max(1, c // int(min_channels_per_group)))
    while g > 1 and (c % g) != 0:
        g -= 1
    return max(1, g)


def _make_gn(num_channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=_gn_groups(num_channels), num_channels=int(num_channels))


def _cfg_value(cfg: Mapping[str, Any], key: str, default: Any) -> Any:
    value = cfg.get(key, default)
    return default if value is None else value


class _SharedPerBandEmbedding(nn.Module):
    """
    A lightweight per-band stem that keeps the band dimension explicit.

    It applies the same small CNN to each band independently by reshaping:
        (B, N, H, W) -> (B*N, 1, H, W) -> (B, N, C_emb, H', W')

    This avoids any cross-band mixing before alignment.
    """

    def __init__(self, *, embed_channels: int, use_bn: bool = True) -> None:
        super().__init__()
        c = int(embed_channels)
        if c <= 0:
            raise ValueError(f"embed_channels must be > 0, got {embed_channels}")
        if use_bn:
            self.net = nn.Sequential(
                nn.Conv2d(1, c, kernel_size=3, stride=2, padding=1, bias=False),
                # Use GroupNorm instead of BatchNorm: we reshape (B,N,...) -> (B*N,...), and BN would
                # mix statistics across bands/samples. GN is per-sample and more stable for small batch.
                _make_gn(c),
                nn.ReLU(inplace=True),
                nn.Conv2d(c, c, kernel_size=3, stride=2, padding=1, bias=False),
                _make_gn(c),
                nn.ReLU(inplace=True),
            )
        else:
            self.net = nn.Sequential(
                nn.Conv2d(1, c, kernel_size=3, stride=2, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(c, c, kernel_size=3, stride=2, padding=1, bias=False),
                nn.ReLU(inplace=True),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, H, W) -> (B, N, C, H/4, W/4) C是原始通道数
        if x.ndim != 4:
            raise ValueError(f"_SharedPerBandEmbedding expects (B,N,H,W), got {x.shape}")
        b, n, h, w = x.shape
        y = self.net(x.reshape(b * n, 1, h, w))
        _, c, hh, ww = y.shape
        return y.reshape(b, n, c, hh, ww)


class MSBandSeparatedStemAlign(nn.Module):
    """
    Scheme-1 MS stem: avoid cross-band fusion before C2 alignment.

    Pipeline:
      1) per-band embedding (shared small CNN): (B,7,H,W) -> (B,7,Cemb,H/4,W/4)
      2) explicit-band alignment/fusion on the kept band axis (optional)
      3) merge to the original backbone's expected C2 input channels: (B,7*Cemb,...) -> (B,C2_in,...)

    This module is designed to be plugged into HGNetv2DualStream *instead of* `ms_backbone.stem`.
    """

    def __init__(
        self,
        *,
        ms_in_chs: int,
        c2_in_channels: int,
        embed_channels: int = 16,
        embed_use_bn: bool = True,
        # Alignment config (CRGGA or fixed-band CMDA). If None/disabled -> no alignment.
        align_cfg: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.ms_in_chs = int(ms_in_chs)
        self.c2_in_channels = int(c2_in_channels)
        if self.ms_in_chs <= 0:
            raise ValueError(f"ms_in_chs must be > 0, got {ms_in_chs}")
        if self.c2_in_channels <= 0:
            raise ValueError(f"c2_in_channels must be > 0, got {c2_in_channels}")

        self.embed_channels = int(embed_channels)
        self.embed = _SharedPerBandEmbedding(embed_channels=self.embed_channels, use_bn=bool(embed_use_bn))

        cfg: dict[str, Any] = {}
        if align_cfg is not None:
            if isinstance(align_cfg, Mapping):
                cfg = dict(align_cfg)
            elif hasattr(align_cfg, "items"):
                cfg = {k: v for k, v in align_cfg.items()}  # type: ignore[assignment]

        align_enabled = bool(cfg.get("enabled", cfg.get("enable", False)))
        # If the caller provides an align config without an explicit enable, assume they intended to enable it.
        if align_cfg is not None and "enabled" not in cfg and "enable" not in cfg:
            align_enabled = True
        self.align_enabled = align_enabled

        self.aligner: nn.Module | None = None
        if self.align_enabled:
            align_type_raw = str(_cfg_value(cfg, "type", _cfg_value(cfg, "align_type", "crgga"))).strip().lower()
            if align_type_raw in {"groupwise", "group_align", "group_deform", "crgga"}:
                align_type_raw = "crgga"
            elif align_type_raw in {"cmda", "band_cmda", "fixedbandcmda", "fixed_band_cmda"}:
                align_type_raw = "fixed_band_cmda"
            else:
                raise ValueError(
                    f"Unsupported align.type={align_type_raw} (supported: crgga|fixed_band_cmda)"
                )
            if align_type_raw == "crgga":
                ref_mode_raw = str(_cfg_value(cfg, "ref_mode", "spatial_weighted")).strip().lower()
                if ref_mode_raw in {"fixed", "single_band", "band"}:
                    ref_mode_raw = "fixed_band"
                if ref_mode_raw not in {"mean", "global_weighted", "spatial_weighted", "fixed_band"}:
                    raise ValueError(
                        f"Unsupported ref_mode={ref_mode_raw} (supported: mean|global_weighted|spatial_weighted|fixed_band)"
                    )
                ref_mode = cast(Any, ref_mode_raw)
                self.aligner = GroupwiseDeformableAlign2D(
                    in_channels=self.embed_channels,
                    ref_mode=ref_mode,
                    ref_band_index=cfg.get("ref_band_index", cfg.get("ref_channel", None)),
                    num_iters=int(_cfg_value(cfg, "num_iters", 1)),
                    ref_detach=bool(cfg.get("ref_detach", True)),
                    num_keypoints=int(_cfg_value(cfg, "num_keypoints", 9)),
                    offset_scale=float(_cfg_value(cfg, "offset_scale", 3.0)),
                    offset_enabled=bool(cfg.get("offset_enabled", True)),
                    attention_norm=str(_cfg_value(cfg, "attention_norm", "softmax")),
                    padding_mode=str(_cfg_value(cfg, "padding_mode", "border")),
                    align_corners=bool(cfg.get("align_corners", True)),
                    loss_type=str(_cfg_value(cfg, "loss_type", "infonce")),
                    loss_downsample=cfg.get("loss_downsample", 0.5),
                    nce_num_patches=int(_cfg_value(cfg, "nce_num_patches", 64)),
                    nce_patch_size=int(_cfg_value(cfg, "nce_patch_size", 5)),
                    nce_tau=float(_cfg_value(cfg, "nce_tau", 0.2)),
                    affine_enabled=bool(cfg.get("affine_enabled", cfg.get("affine", False))),
                    affine_scale=float(_cfg_value(cfg, "affine_scale", 0.1)),
                    affine_init_identity=bool(cfg.get("affine_init_identity", True)),
                    affine_type=str(_cfg_value(cfg, "affine_type", "affine")),
                    loss_weight=float(_cfg_value(cfg, "loss_weight", 0.02)),
                    loss_offset_weight=float(_cfg_value(cfg, "loss_offset_weight", 0.01)),
                    loss_attn_norm_weight=float(_cfg_value(cfg, "loss_attn_norm_weight", 0.0)),
                    loss_attn_entropy_weight=float(_cfg_value(cfg, "loss_attn_entropy_weight", 0.001)),
                )
            else:
                self.aligner = FixedBandCMDA(
                    in_channels=self.embed_channels,
                    anchor_band_index=cfg.get(
                        "anchor_band_index",
                        cfg.get("ref_band_index", cfg.get("ref_channel", None)),
                    ),
                    num_iters=int(_cfg_value(cfg, "num_iters", 1)),
                    anchor_detach=bool(cfg.get("anchor_detach", cfg.get("ref_detach", True))),
                    num_keypoints=int(_cfg_value(cfg, "num_keypoints", 9)),
                    offset_scale=float(_cfg_value(cfg, "offset_scale", 3.0)),
                    offset_enabled=bool(cfg.get("offset_enabled", True)),
                    attention_norm=str(_cfg_value(cfg, "attention_norm", "softmax")),
                    padding_mode=str(_cfg_value(cfg, "padding_mode", "border")),
                    align_corners=bool(cfg.get("align_corners", True)),
                    loss_type=str(_cfg_value(cfg, "loss_type", "infonce")),
                    loss_downsample=cfg.get("loss_downsample", 0.5),
                    nce_num_patches=int(_cfg_value(cfg, "nce_num_patches", 64)),
                    nce_patch_size=int(_cfg_value(cfg, "nce_patch_size", 5)),
                    nce_tau=float(_cfg_value(cfg, "nce_tau", 0.2)),
                    affine_enabled=bool(cfg.get("affine_enabled", cfg.get("affine", False))),
                    affine_scale=float(_cfg_value(cfg, "affine_scale", 0.1)),
                    affine_init_identity=bool(cfg.get("affine_init_identity", True)),
                    affine_type=str(_cfg_value(cfg, "affine_type", "affine")),
                    loss_weight=float(_cfg_value(cfg, "loss_weight", 0.02)),
                    loss_offset_weight=float(_cfg_value(cfg, "loss_offset_weight", 0.01)),
                    loss_attn_norm_weight=float(_cfg_value(cfg, "loss_attn_norm_weight", 0.0)),
                    loss_attn_entropy_weight=float(_cfg_value(cfg, "loss_attn_entropy_weight", 0.001)),
                    fuse_hidden_channels=int(_cfg_value(cfg, "fuse_hidden_channels", self.embed_channels)),
                )

        merge_in = int(self.ms_in_chs * self.embed_channels)
        self.merge = nn.Sequential(
            nn.Conv2d(merge_in, self.c2_in_channels, kernel_size=1, bias=False),
            _make_gn(self.c2_in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, ms: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if ms.ndim != 4:
            raise ValueError(f"MSBandSeparatedStemAlign expects BCHW tensor, got {ms.shape}")
        b, c, h, w = ms.shape
        if c != self.ms_in_chs:
            raise ValueError(f"Channel mismatch: expected ms_in_chs={self.ms_in_chs}, got C={c}")

        z: torch.Tensor = self.embed(ms)  # (B, N, Cemb, H/4, W/4)
        aux_losses: dict[str, torch.Tensor] = {}
        if self.aligner is not None:
            out = self.aligner(z)
            if self.training and isinstance(out, tuple) and len(out) >= 2 and torch.is_tensor(out[0]) and isinstance(out[1], dict):
                z = out[0]
                aux_losses = out[1]
            else:
                z = out[0] if isinstance(out, tuple) else out
            if not torch.is_tensor(z):
                raise RuntimeError("Unexpected aligner output type in MSBandSeparatedStemAlign")

        z_flat = z.reshape(b, self.ms_in_chs * self.embed_channels, z.shape[-2], z.shape[-1])
        y = self.merge(z_flat)
        if self.training and aux_losses:
            return y, aux_losses
        return y
