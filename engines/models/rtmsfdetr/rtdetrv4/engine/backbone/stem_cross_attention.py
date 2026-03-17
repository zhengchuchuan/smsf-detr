from __future__ import annotations

import torch
import torch.nn as nn

from .coattention import CrossAttention2D

__all__ = ["StemCrossAttention2D"]


class StemCrossAttention2D(nn.Module):
    """
    Standard cross-attention interaction at stem/C2 resolution.

    Query:
        main stem feature, shaped (B, C, H, W)
    Memory:
        residual MS branch feature, shaped (B, C, H, W)

    Compared with StemCFInteractive2D, this variant uses standard scaled
    dot-product cross-attention on pooled KV tokens instead of deformable
    reference-point sampling.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        d_model: int = 128,
        nhead: int = 4,
        kv_stride: int = 8,
        dropout: float = 0.0,
        alpha_init: float = 0.0,
        memory_detach: bool = True,
    ) -> None:
        super().__init__()
        self.memory_detach = bool(memory_detach)
        self.cross_attn = CrossAttention2D(
            in_channels=int(in_channels),
            d_model=int(d_model),
            nhead=int(nhead),
            kv_stride=int(kv_stride),
            dropout=float(dropout),
            alpha_init=float(alpha_init),
        )

    def forward(self, query_feat: torch.Tensor, memory_feat: torch.Tensor) -> torch.Tensor:
        if query_feat.ndim != 4 or memory_feat.ndim != 4:
            raise ValueError(
                "StemCrossAttention2D expects BCHW tensors, "
                f"got query={tuple(query_feat.shape)}, memory={tuple(memory_feat.shape)}"
            )
        if query_feat.shape != memory_feat.shape:
            raise ValueError(
                f"Query/memory shape mismatch: query={tuple(query_feat.shape)} memory={tuple(memory_feat.shape)}"
            )

        memory_src = memory_feat.detach() if self.memory_detach else memory_feat
        delta = self.cross_attn(query_feat, memory_src)
        return query_feat + delta
