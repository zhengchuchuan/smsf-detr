from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention2D(nn.Module):
    """
    Cross-attention for 2D feature maps (B,C,H,W).

    Design goal: keep compute bounded by downsampling KV (Q stays full-res).
    """

    def __init__(
        self,
        *,
        in_channels: int,
        d_model: int = 128,
        nhead: int = 8,
        kv_stride: int = 8,
        dropout: float = 0.0,
        alpha_init: float = 0.0,
    ) -> None:
        super().__init__()
        in_channels = int(in_channels)
        d_model = int(d_model)
        nhead = int(nhead)
        kv_stride = int(kv_stride)
        dropout = float(dropout)
        alpha_init = float(alpha_init)

        if in_channels <= 0:
            raise ValueError(f"in_channels must be > 0, got {in_channels}")
        if d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {d_model}")
        if nhead <= 0:
            raise ValueError(f"nhead must be > 0, got {nhead}")
        if d_model % nhead != 0:
            raise ValueError(f"d_model must be divisible by nhead, got d_model={d_model} nhead={nhead}")
        if kv_stride <= 0:
            raise ValueError(f"kv_stride must be > 0, got {kv_stride}")

        self.in_channels = in_channels
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.kv_stride = kv_stride
        self.dropout_p = dropout

        self.q_proj = nn.Conv2d(in_channels, d_model, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(in_channels, d_model, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(in_channels, d_model, kernel_size=1, bias=False)
        self.out_proj = nn.Conv2d(d_model, in_channels, kernel_size=1, bias=False)

        self.q_norm = nn.LayerNorm(d_model)
        self.k_norm = nn.LayerNorm(d_model)
        self.v_norm = nn.LayerNorm(d_model)

        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))

        self.kv_pool: nn.Module
        if kv_stride == 1:
            self.kv_pool = nn.Identity()
        else:
            self.kv_pool = nn.AvgPool2d(kernel_size=kv_stride, stride=kv_stride, ceil_mode=True)

    def _project(
        self, x: torch.Tensor, proj: nn.Conv2d, norm: nn.LayerNorm
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        b, _, h, w = x.shape
        x = proj(x).flatten(2).transpose(1, 2)  # (B, HW, D)
        x = norm(x)
        return x, (h, w)

    def forward(self, q_feat: torch.Tensor, kv_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q_feat: (B, C, H, W)
            kv_feat: (B, C, H, W) (same scale as q_feat)
        Returns:
            delta: (B, C, H, W), already scaled by learnable alpha.
        """
        if q_feat.ndim != 4 or kv_feat.ndim != 4:
            raise ValueError(f"CrossAttention2D expects 4D tensors, got q={q_feat.shape} kv={kv_feat.shape}")
        if q_feat.shape[:2] != kv_feat.shape[:2]:
            raise ValueError(f"q/kv must share (B,C), got q={q_feat.shape} kv={kv_feat.shape}")

        kv = self.kv_pool(kv_feat)

        q, (h, w) = self._project(q_feat, self.q_proj, self.q_norm)
        k, _ = self._project(kv, self.k_proj, self.k_norm)
        v, _ = self._project(kv, self.v_proj, self.v_norm)

        b, lq, _ = q.shape
        lk = k.shape[1]

        q = q.view(b, lq, self.nhead, self.head_dim).transpose(1, 2)  # (B, nh, Lq, hd)
        k = k.view(b, lk, self.nhead, self.head_dim).transpose(1, 2)  # (B, nh, Lk, hd)
        v = v.view(b, lk, self.nhead, self.head_dim).transpose(1, 2)  # (B, nh, Lk, hd)

        dropout_p = self.dropout_p if self.training and self.dropout_p > 0 else 0.0
        # Stability: AMP can overflow in attention; do attention in fp32 when inputs are fp16/bf16.
        orig_dtype = q.dtype
        if orig_dtype in (torch.float16, torch.bfloat16):
            q = q.float()
            k = k.float()
            v = v.float()
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=False)
        if out.dtype != orig_dtype:
            out = out.to(dtype=orig_dtype)
        out = out.transpose(1, 2).contiguous().view(b, lq, self.d_model)  # (B, Lq, D)
        out = out.transpose(1, 2).reshape(b, self.d_model, h, w)  # (B, D, H, W)

        delta = self.out_proj(out)
        # Use bounded residual scaling to avoid runaway amplification.
        scale = torch.tanh(self.alpha).to(dtype=delta.dtype)
        return scale * delta


class CoAttention2D(nn.Module):
    """
    Symmetric co-attention (cross-attn both directions) for same-scale feature maps.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        d_model: int = 128,
        nhead: int = 8,
        kv_stride: int = 8,
        dropout: float = 0.0,
        alpha_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.rgb_from_ms = CrossAttention2D(
            in_channels=in_channels,
            d_model=d_model,
            nhead=nhead,
            kv_stride=kv_stride,
            dropout=dropout,
            alpha_init=alpha_init,
        )
        self.ms_from_rgb = CrossAttention2D(
            in_channels=in_channels,
            d_model=d_model,
            nhead=nhead,
            kv_stride=kv_stride,
            dropout=dropout,
            alpha_init=alpha_init,
        )

    def forward(self, rgb: torch.Tensor, ms: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if rgb.shape != ms.shape:
            raise ValueError(f"CoAttention2D expects same shapes, got rgb={rgb.shape} ms={ms.shape}")
        delta_rgb = self.rgb_from_ms(rgb, ms)
        delta_ms = self.ms_from_rgb(ms, rgb)
        return rgb + delta_rgb, ms + delta_ms


__all__ = ["CrossAttention2D", "CoAttention2D"]
