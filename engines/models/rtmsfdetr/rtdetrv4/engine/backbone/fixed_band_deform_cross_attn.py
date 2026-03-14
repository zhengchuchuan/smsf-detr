from __future__ import annotations

import torch
import torch.nn as nn

from engines.models.msifdetr.common.ops.modules import MSDeformAttn

__all__ = ["FixedBandDeformCrossAttn"]


def _gn_groups(num_channels: int, *, max_groups: int = 8, min_channels_per_group: int = 4) -> int:
    c = int(num_channels)
    if c <= 0:
        raise ValueError(f"num_channels must be > 0, got {num_channels}")
    g = min(int(max_groups), max(1, c // int(min_channels_per_group)))
    while g > 1 and (c % g) != 0:
        g -= 1
    return max(1, g)


def _make_gn(num_channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=_gn_groups(num_channels), num_channels=int(num_channels))


class FixedBandDeformCrossAttn(nn.Module):
    """
    Fixed-band soft deformable cross-attention on explicit shallow MS features.

    Input:
        x: (B, N, C, H, W), where N is the band count and C is the per-band feature dim.

    Behavior:
        - keep one fixed anchor band explicit
        - use the anchor band as query
        - treat the remaining bands as pseudo feature levels for deformable cross-attention
        - write the corrected anchor feature back into the explicit band tensor

    This is intentionally softer than CMDA/warp-based alignment:
        - no explicit pairwise warp target is constructed
        - the module learns an anchor-conditioned residual from support bands
        - the corrected anchor is then merged by the outer residual MS branch
    """

    def __init__(
        self,
        *,
        in_channels: int,
        num_bands: int,
        anchor_band_index: int | str | None = None,
        anchor_detach: bool = False,
        num_heads: int = 4,
        num_points: int = 4,
        band_embed_enabled: bool = True,
        support_ref_shift_enabled: bool = True,
        support_ref_shift_scale: float = 0.02,
        delta_hidden_channels: int | None = None,
        delta_scale_init: float = 0.05,
        delta_scale_per_channel: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_bands = int(num_bands)
        if self.in_channels <= 0:
            raise ValueError(f"in_channels must be > 0, got {in_channels}")
        if self.num_bands < 2:
            raise ValueError(f"num_bands must be >= 2, got {num_bands}")

        if anchor_band_index is None:
            self.anchor_band_index = None
        elif isinstance(anchor_band_index, str):
            idx_norm = anchor_band_index.strip().lower()
            if idx_norm in {"mid", "middle", "center", "centre", "auto"}:
                self.anchor_band_index = None
            else:
                self.anchor_band_index = int(anchor_band_index)
        else:
            self.anchor_band_index = int(anchor_band_index)

        self.anchor_detach = bool(anchor_detach)
        self.num_support_levels = self.num_bands - 1
        self.num_heads = int(num_heads)
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be > 0, got {num_heads}")
        if self.in_channels % self.num_heads != 0:
            raise ValueError(
                f"FixedBandDeformCrossAttn requires in_channels % num_heads == 0, "
                f"got C={self.in_channels}, heads={self.num_heads}"
            )
        self.num_points = int(num_points)
        if self.num_points <= 0:
            raise ValueError(f"num_points must be > 0, got {num_points}")

        hidden_channels = int(delta_hidden_channels) if delta_hidden_channels is not None else self.in_channels
        if hidden_channels <= 0:
            raise ValueError(f"delta_hidden_channels must be > 0, got {delta_hidden_channels}")

        self.query_pre = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, bias=False),
            _make_gn(self.in_channels),
            nn.ReLU(inplace=True),
        )
        self.support_pre = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, bias=False),
            _make_gn(self.in_channels),
            nn.ReLU(inplace=True),
        )
        self.cross_attn = MSDeformAttn(
            d_model=self.in_channels,
            n_levels=self.num_support_levels,
            n_heads=self.num_heads,
            n_points=self.num_points,
        )
        self.delta_fuse = nn.Sequential(
            nn.Conv2d(self.in_channels * 2, hidden_channels, kernel_size=1, bias=False),
            _make_gn(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, self.in_channels, kernel_size=1, bias=True),
        )

        if bool(band_embed_enabled):
            self.band_embed = nn.Parameter(torch.zeros(self.num_bands, self.in_channels))
            nn.init.normal_(self.band_embed, mean=0.0, std=0.02)
        else:
            self.band_embed = None

        self.support_ref_shift_enabled = bool(support_ref_shift_enabled)
        self.support_ref_shift_scale = float(support_ref_shift_scale)
        if self.support_ref_shift_scale < 0:
            raise ValueError(f"support_ref_shift_scale must be >= 0, got {support_ref_shift_scale}")
        if self.support_ref_shift_enabled:
            self.support_ref_shift = nn.Parameter(torch.zeros(self.num_support_levels, 2))
        else:
            self.support_ref_shift = None

        scale_shape = (1, self.in_channels, 1, 1) if bool(delta_scale_per_channel) else (1,)
        self.delta_scale = nn.Parameter(torch.full(scale_shape, float(delta_scale_init)))

    def _resolve_anchor_index(self, num_bands: int) -> int:
        if num_bands <= 0:
            raise ValueError(f"num_bands must be > 0, got {num_bands}")
        if self.anchor_band_index is None:
            return int(num_bands // 2)
        idx = self.anchor_band_index if self.anchor_band_index >= 0 else num_bands + self.anchor_band_index
        if idx < 0 or idx >= num_bands:
            raise ValueError(f"anchor_band_index out of range: idx={idx}, num_bands={num_bands}")
        return int(idx)

    @staticmethod
    def _build_reference_points(
        h: int,
        w: int,
        *,
        num_levels: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, h - 0.5, h, dtype=torch.float32, device=device),
            torch.linspace(0.5, w - 0.5, w, dtype=torch.float32, device=device),
            indexing="ij",
        )
        ref_y = (ref_y.reshape(-1) / float(h)).to(dtype=dtype)
        ref_x = (ref_x.reshape(-1) / float(w)).to(dtype=dtype)
        ref = torch.stack((ref_x, ref_y), dim=-1)  # (HW, 2)
        ref = ref.unsqueeze(1).expand(-1, int(num_levels), -1)  # (HW, L, 2)
        return ref.unsqueeze(0).expand(int(batch_size), -1, -1, -1).contiguous()

    @staticmethod
    def _build_level_start_index(spatial_shapes: torch.Tensor) -> torch.Tensor:
        if spatial_shapes.ndim != 2 or spatial_shapes.shape[1] != 2:
            raise ValueError(f"Expected spatial_shapes as (L,2), got {tuple(spatial_shapes.shape)}")
        counts = spatial_shapes[:, 0] * spatial_shapes[:, 1]
        start = counts.cumsum(dim=0) - counts
        return start.to(dtype=torch.long)

    def _support_band_indices(self, anchor_idx: int) -> list[int]:
        return [i for i in range(self.num_bands) if i != int(anchor_idx)]

    def _add_band_embed(self, feat: torch.Tensor, band_idx: int) -> torch.Tensor:
        if self.band_embed is None:
            return feat
        embed = self.band_embed[int(band_idx)].view(1, self.in_channels, 1, 1).to(dtype=feat.dtype, device=feat.device)
        return feat + embed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"FixedBandDeformCrossAttn expects (B,N,C,H,W), got {tuple(x.shape)}")
        b, n, c, h, w = x.shape
        if n != self.num_bands:
            raise ValueError(f"Band mismatch: expected N={self.num_bands}, got N={n}")
        if c != self.in_channels:
            raise ValueError(f"Channel mismatch: expected C={self.in_channels}, got C={c}")
        if n < 2:
            return x

        anchor_idx = self._resolve_anchor_index(n)
        support_indices = self._support_band_indices(anchor_idx)

        anchor = x[:, anchor_idx]
        anchor_query = anchor.detach() if self.anchor_detach else anchor
        anchor_query = self._add_band_embed(anchor_query, anchor_idx)
        query = self.query_pre(anchor_query).flatten(2).transpose(1, 2).contiguous()  # (B, HW, C)

        support_levels = []
        for band_idx in support_indices:
            feat = self._add_band_embed(x[:, band_idx], band_idx)
            support_levels.append(self.support_pre(feat))
        support = torch.stack(support_levels, dim=1)  # (B, L, C, H, W)

        memory = support.permute(0, 1, 3, 4, 2).reshape(b, self.num_support_levels * h * w, c).contiguous()
        spatial_shapes = torch.tensor(
            [(h, w)] * self.num_support_levels,
            dtype=torch.long,
            device=x.device,
        )
        level_start_index = self._build_level_start_index(spatial_shapes)
        reference_points = self._build_reference_points(
            h,
            w,
            num_levels=self.num_support_levels,
            batch_size=b,
            device=x.device,
            dtype=x.dtype,
        )
        if self.support_ref_shift is not None:
            shift = (self.support_ref_shift_scale * torch.tanh(self.support_ref_shift)).to(dtype=x.dtype, device=x.device)
            reference_points = (reference_points + shift.view(1, 1, self.num_support_levels, 2)).clamp(0.0, 1.0)

        delta = self.cross_attn(
            query,
            reference_points,
            memory,
            spatial_shapes,
            level_start_index,
            input_padding_mask=None,
        )
        delta = delta.transpose(1, 2).reshape(b, c, h, w).contiguous()
        fused_anchor = anchor + self.delta_scale * self.delta_fuse(torch.cat([anchor, delta], dim=1))

        out = x.clone()
        out[:, anchor_idx] = fused_anchor
        return out
