from __future__ import annotations

import torch
import torch.nn as nn

from engines.models.msifdetr.common.ops.modules import MSDeformAttn

__all__ = ["StemCFInteractive2D"]


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


class StemCFInteractive2D(nn.Module):
    """
    Single-scale CF-style deformable cross interaction at stem/C2 resolution.

    Query:
        main stem feature, shaped (B, C, H, W)
    Memory:
        residual MS branch feature, shaped (B, C, H, W)

    Behavior:
        - keep the original stem feature as the query branch
        - use the residual MS branch as the support memory
        - apply dense single-scale deformable cross-attention on the C2 lattice
        - write back a small residual correction to the main stem feature

    This is intentionally conservative:
        - output projection starts from zero -> near identity at initialization
        - the learned output scale is small by default
        - memory can optionally be detached for more stable coupling
    """

    def __init__(
        self,
        *,
        in_channels: int,
        num_heads: int = 4,
        num_points: int = 4,
        memory_detach: bool = True,
        ref_shift_enabled: bool = True,
        ref_shift_scale: float = 0.02,
        delta_hidden_channels: int | None = None,
        scale_init: float = 0.01,
        scale_per_channel: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        if self.in_channels <= 0:
            raise ValueError(f"in_channels must be > 0, got {in_channels}")

        self.num_heads = int(num_heads)
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be > 0, got {num_heads}")
        if self.in_channels % self.num_heads != 0:
            raise ValueError(
                f"StemCFInteractive2D requires in_channels % num_heads == 0, "
                f"got C={self.in_channels}, heads={self.num_heads}"
            )

        self.num_points = int(num_points)
        if self.num_points <= 0:
            raise ValueError(f"num_points must be > 0, got {num_points}")

        self.memory_detach = bool(memory_detach)
        self.ref_shift_enabled = bool(ref_shift_enabled)
        self.ref_shift_scale = float(ref_shift_scale)
        if self.ref_shift_scale < 0:
            raise ValueError(f"ref_shift_scale must be >= 0, got {ref_shift_scale}")

        hidden_channels = int(delta_hidden_channels) if delta_hidden_channels is not None else self.in_channels
        if hidden_channels <= 0:
            raise ValueError(f"delta_hidden_channels must be > 0, got {delta_hidden_channels}")

        self.query_pre = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, bias=False),
            _make_gn(self.in_channels),
            nn.ReLU(inplace=True),
        )
        self.memory_pre = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, bias=False),
            _make_gn(self.in_channels),
            nn.ReLU(inplace=True),
        )
        self.cross_attn = MSDeformAttn(
            d_model=self.in_channels,
            n_levels=1,
            n_heads=self.num_heads,
            n_points=self.num_points,
        )
        self.delta_fuse = nn.Sequential(
            nn.Conv2d(self.in_channels * 2, hidden_channels, kernel_size=1, bias=False),
            _make_gn(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, self.in_channels, kernel_size=1, bias=True),
        )

        last = self.delta_fuse[-1]
        assert isinstance(last, nn.Conv2d)
        nn.init.constant_(last.weight, 0.0)
        if last.bias is not None:
            nn.init.constant_(last.bias, 0.0)

        if self.ref_shift_enabled:
            self.ref_shift = nn.Parameter(torch.zeros(1, 2))
        else:
            self.ref_shift = None

        scale_shape = (1, self.in_channels, 1, 1) if bool(scale_per_channel) else (1,)
        self.output_scale = nn.Parameter(torch.full(scale_shape, float(scale_init)))

    @staticmethod
    def _build_reference_points(
        h: int,
        w: int,
        *,
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
        ref = ref.unsqueeze(1)  # (HW, 1, 2)
        return ref.unsqueeze(0).expand(int(batch_size), -1, -1, -1).contiguous()

    def forward(self, query_feat: torch.Tensor, memory_feat: torch.Tensor) -> torch.Tensor:
        if query_feat.ndim != 4 or memory_feat.ndim != 4:
            raise ValueError(
                "StemCFInteractive2D expects BCHW tensors, "
                f"got query={tuple(query_feat.shape)}, memory={tuple(memory_feat.shape)}"
            )
        if query_feat.shape != memory_feat.shape:
            raise ValueError(
                f"Query/memory shape mismatch: query={tuple(query_feat.shape)} memory={tuple(memory_feat.shape)}"
            )

        b, c, h, w = query_feat.shape
        if c != self.in_channels:
            raise ValueError(f"Channel mismatch: expected C={self.in_channels}, got C={c}")

        memory_src = memory_feat.detach() if self.memory_detach else memory_feat

        query = self.query_pre(query_feat).flatten(2).transpose(1, 2).contiguous()  # (B, HW, C)
        memory = self.memory_pre(memory_src).permute(0, 2, 3, 1).reshape(b, h * w, c).contiguous()

        spatial_shapes = torch.tensor([(h, w)], dtype=torch.long, device=query_feat.device)
        level_start_index = torch.zeros(1, dtype=torch.long, device=query_feat.device)
        reference_points = self._build_reference_points(
            h,
            w,
            batch_size=b,
            device=query_feat.device,
            dtype=query_feat.dtype,
        )
        if self.ref_shift is not None:
            shift = (self.ref_shift_scale * torch.tanh(self.ref_shift)).to(
                dtype=query_feat.dtype,
                device=query_feat.device,
            )
            reference_points = (reference_points + shift.view(1, 1, 1, 2)).clamp(0.0, 1.0)

        delta = self.cross_attn(
            query,
            reference_points,
            memory,
            spatial_shapes,
            level_start_index,
            input_padding_mask=None,
        )
        delta = delta.transpose(1, 2).reshape(b, c, h, w).contiguous()
        correction = self.delta_fuse(torch.cat([query_feat, delta], dim=1))
        return query_feat + self.output_scale * correction
