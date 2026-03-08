from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


def _build_2d_sincos_pos_embed(
    channels: int,
    height: int,
    width: int,
    *,
    temperature: float = 10000.0,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    ch = int(channels)
    h = int(height)
    w = int(width)
    if ch <= 0:
        raise ValueError(f"channels must be > 0, got {channels}")
    if h <= 0 or w <= 0:
        raise ValueError(f"height/width must be > 0, got height={height} width={width}")

    pos_dim = ch // 4
    if pos_dim == 0:
        return torch.zeros((1, ch, h, w), device=device, dtype=dtype)

    grid_y = torch.arange(h, device=device, dtype=torch.float32)
    grid_x = torch.arange(w, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(grid_y, grid_x, indexing="ij")

    omega = torch.arange(pos_dim, device=device, dtype=torch.float32) / float(pos_dim)
    omega = 1.0 / (float(temperature) ** omega)  # (pos_dim,)

    out_x = xx.reshape(-1)[:, None] * omega[None, :]  # (H*W, pos_dim)
    out_y = yy.reshape(-1)[:, None] * omega[None, :]  # (H*W, pos_dim)

    emb = torch.cat([out_x.sin(), out_x.cos(), out_y.sin(), out_y.cos()], dim=1)  # (H*W, 4*pos_dim)
    emb = emb.transpose(0, 1).reshape(4 * pos_dim, h, w)  # (4*pos_dim, H, W)

    if emb.shape[0] < ch:
        pad = torch.zeros((ch - emb.shape[0], h, w), device=device, dtype=torch.float32)
        emb = torch.cat([emb, pad], dim=0)

    return emb.unsqueeze(0).to(dtype=dtype)


class MRTCrossSpectrumAttention2D(nn.Module):
    """
    Cross-spectrum attention adapted from `third_party/MRT-DETR/rtdetr_pytorch/src/zoo/rtdetr/cross_attention.py`.

    Notes:
    - Operates on 2D feature maps (B,C,H,W) and returns a single fused feature map (B,C,H,W).
    - Optionally projects to a smaller `d_model` for compute reduction, then projects back to `in_channels`.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        d_model: int | None = None,
        num_stages: int = 2,
        use_pos_encoding: bool = True,
        temperature: float = 10000.0,
    ) -> None:
        super().__init__()
        in_channels = int(in_channels)
        if d_model is None or int(d_model) <= 0:
            d_model = in_channels
        d_model = int(d_model)
        num_stages = int(num_stages)
        use_pos_encoding = bool(use_pos_encoding)
        temperature = float(temperature)

        if in_channels <= 0:
            raise ValueError(f"in_channels must be > 0, got {in_channels}")
        if d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {d_model}")
        if num_stages <= 0:
            raise ValueError(f"num_stages must be > 0, got {num_stages}")
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")

        self.in_channels = in_channels
        self.d_model = d_model
        self.num_stages = num_stages
        self.use_pos_encoding = use_pos_encoding
        self.temperature = temperature

        self.in_proj: nn.Module
        self.out_proj: nn.Module
        if d_model == in_channels:
            self.in_proj = nn.Identity()
            self.out_proj = nn.Identity()
        else:
            self.in_proj = nn.Conv2d(in_channels, d_model, kernel_size=1, bias=False)
            self.out_proj = nn.Conv2d(d_model, in_channels, kernel_size=1, bias=False)

        self.q_convs = nn.ModuleList([nn.Conv2d(d_model, d_model, kernel_size=1, bias=False) for _ in range(num_stages)])
        self.k_convs = nn.ModuleList([nn.Conv2d(d_model, d_model, kernel_size=1, bias=False) for _ in range(num_stages)])
        self.v_convs = nn.ModuleList([nn.Conv2d(d_model, d_model, kernel_size=1, bias=False) for _ in range(num_stages)])

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        weight_x1: torch.Tensor | None = None,
        weight_x2: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x1.shape != x2.shape:
            raise ValueError(f"MRTCrossSpectrumAttention2D expects same shapes, got x1={x1.shape} x2={x2.shape}")
        if x1.ndim != 4:
            raise ValueError(f"MRTCrossSpectrumAttention2D expects BCHW tensors, got {x1.shape}")

        b, _, h, w = x1.shape
        x1 = self.in_proj(x1)
        x2 = self.in_proj(x2)

        pos: torch.Tensor | None = None
        if self.use_pos_encoding:
            pos = _build_2d_sincos_pos_embed(
                self.d_model, h, w, temperature=self.temperature, device=x1.device, dtype=x1.dtype
            )

        if weight_x1 is not None:
            if weight_x1.ndim == 3:
                weight_x1 = weight_x1.unsqueeze(1)
            weight_x1 = weight_x1.to(device=x1.device, dtype=x1.dtype)
            if weight_x1.shape[0] != b:
                raise ValueError(f"weight_x1 batch mismatch: x1={b} weight_x1={weight_x1.shape[0]}")
            if weight_x1.shape[-2:] != (h, w):
                weight_x1 = torch.nn.functional.interpolate(weight_x1, size=(h, w), mode="nearest")
            x1 = x1 * weight_x1
        if weight_x2 is not None:
            if weight_x2.ndim == 3:
                weight_x2 = weight_x2.unsqueeze(1)
            weight_x2 = weight_x2.to(device=x2.device, dtype=x2.dtype)
            if weight_x2.shape[0] != b:
                raise ValueError(f"weight_x2 batch mismatch: x2={b} weight_x2={weight_x2.shape[0]}")
            if weight_x2.shape[-2:] != (h, w):
                weight_x2 = torch.nn.functional.interpolate(weight_x2, size=(h, w), mode="nearest")
            x2 = x2 * weight_x2

        scale = 1.0 / math.sqrt(float(w)) if w > 0 else 1.0
        for i in range(self.num_stages):
            q = self.q_convs[i](x1)
            k = self.k_convs[i](x2)
            v = self.v_convs[i](x2)

            if pos is not None:
                q = q + pos
                k = k + pos

            attn_logits = torch.matmul(q, k.transpose(-1, -2)) * scale  # (B, D, H, H)
            attn = torch.softmax(attn_logits, dim=-1)
            fused = torch.matmul(attn, v)  # (B, D, H, W)

            x1 = fused
            x2 = fused

        return self.out_proj(x1)


class MRTCrossSpectrumFusion2D(nn.Module):
    """
    A dual-stream write-back wrapper over MRTCrossSpectrumAttention2D.

    Produces a fused feature map and writes it back to both modalities with configurable merge:
    - add: x = x + alpha * fused
    - wavg: x = (1-w)*x + w*fused  (learnable scalar gates)
    - concat1x1: x = x + proj(cat([x, alpha*fused]))
    """

    def __init__(
        self,
        *,
        in_channels: int,
        d_model: int | None = None,
        num_stages: int = 2,
        writeback_merge: str = "add",
        alpha_init: float = 0.0,
        use_pos_encoding: bool = True,
    ) -> None:
        super().__init__()
        in_channels = int(in_channels)
        if d_model is None or int(d_model) <= 0:
            d_model = in_channels
        d_model = int(d_model)
        num_stages = int(num_stages)
        writeback_merge = str(writeback_merge).strip().lower()
        alpha_init = float(alpha_init)

        if in_channels <= 0:
            raise ValueError(f"in_channels must be > 0, got {in_channels}")
        if d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {d_model}")

        if writeback_merge == "avg":
            writeback_merge = "wavg"
        if writeback_merge in {"concat", "concat_1x1", "concat-1x1"}:
            writeback_merge = "concat1x1"
        self.writeback_merge = writeback_merge

        self.attn = MRTCrossSpectrumAttention2D(
            in_channels=in_channels,
            d_model=d_model,
            num_stages=num_stages,
            use_pos_encoding=bool(use_pos_encoding),
        )

        self.alpha: nn.Parameter | None
        self.merge_gate_rgb: nn.Parameter | None
        self.merge_gate_ms: nn.Parameter | None
        self.concat_proj_rgb: nn.Conv2d | None
        self.concat_proj_ms: nn.Conv2d | None

        if self.writeback_merge == "wavg":
            self.alpha = None
            self.merge_gate_rgb = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
            self.merge_gate_ms = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
            self.concat_proj_rgb = None
            self.concat_proj_ms = None
        elif self.writeback_merge == "concat1x1":
            self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
            self.merge_gate_rgb = None
            self.merge_gate_ms = None
            self.concat_proj_rgb = nn.Conv2d(2 * in_channels, in_channels, kernel_size=1, bias=False)
            self.concat_proj_ms = nn.Conv2d(2 * in_channels, in_channels, kernel_size=1, bias=False)
            nn.init.zeros_(self.concat_proj_rgb.weight)
            nn.init.zeros_(self.concat_proj_ms.weight)
        elif self.writeback_merge == "add":
            self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
            self.merge_gate_rgb = None
            self.merge_gate_ms = None
            self.concat_proj_rgb = None
            self.concat_proj_ms = None
        else:
            raise ValueError(f"Unsupported writeback_merge: {self.writeback_merge}")

    def forward(self, rgb: torch.Tensor, ms: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if rgb.shape != ms.shape:
            raise ValueError(f"MRTCrossSpectrumFusion2D expects same shapes, got rgb={rgb.shape} ms={ms.shape}")
        if rgb.ndim != 4:
            raise ValueError(f"MRTCrossSpectrumFusion2D expects BCHW tensors, got {rgb.shape}")

        fused = self.attn(rgb, ms)

        if self.writeback_merge == "wavg":
            assert self.merge_gate_rgb is not None and self.merge_gate_ms is not None
            w_rgb = torch.sigmoid(self.merge_gate_rgb).to(dtype=rgb.dtype, device=rgb.device)
            w_ms = torch.sigmoid(self.merge_gate_ms).to(dtype=ms.dtype, device=ms.device)
            rgb = rgb * (1.0 - w_rgb) + fused * w_rgb
            ms = ms * (1.0 - w_ms) + fused * w_ms
        elif self.writeback_merge == "concat1x1":
            assert self.alpha is not None
            assert self.concat_proj_rgb is not None and self.concat_proj_ms is not None
            scale = torch.tanh(self.alpha).to(dtype=fused.dtype, device=fused.device)
            fused = fused * scale
            rgb = rgb + self.concat_proj_rgb(torch.cat([rgb, fused], dim=1))
            ms = ms + self.concat_proj_ms(torch.cat([ms, fused], dim=1))
        elif self.writeback_merge == "add":
            assert self.alpha is not None
            scale = torch.tanh(self.alpha).to(dtype=fused.dtype, device=fused.device)
            fused = fused * scale
            rgb = rgb + fused
            ms = ms + fused
        else:
            raise RuntimeError(f"Unsupported writeback_merge: {self.writeback_merge}")

        return rgb, ms


def _make_norm2d(kind: str, channels: int) -> nn.Module:
    k = str(kind).strip().lower()
    ch = int(channels)
    if k in {"", "none", "identity", "id"}:
        return nn.Identity()
    if k in {"bn", "batchnorm", "batch_norm"}:
        return nn.BatchNorm2d(ch)
    if k in {"gn", "groupnorm", "group_norm"}:
        # Pick a valid group count (<=32) that divides channels.
        max_groups = min(32, ch)
        groups = 1
        for g in range(max_groups, 0, -1):
            if ch % g == 0:
                groups = g
                break
        return nn.GroupNorm(groups, ch)
    raise ValueError(f"Unsupported norm kind: {kind} (supported: none|bn|gn)")


class MRTCrossSpectrumCorrFusion2D(nn.Module):
    """
    MRT-style corr fusion for two feature maps:

    - compute fused = CrossSpectrumAttention(x1, x2)
    - output = Conv1x1( concat([x1, x2, proj(fused)]) )

    This matches the overall idea used in MRT-DETR (cross-attn + redim_corr), but without brightness weights.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        d_model: int | None = None,
        num_stages: int = 2,
        fused_channels: int = 64,
        norm: str = "gn",
        use_pos_encoding: bool = True,
    ) -> None:
        super().__init__()
        in_channels = int(in_channels)
        if d_model is None or int(d_model) <= 0:
            d_model = in_channels
        d_model = int(d_model)
        num_stages = int(num_stages)
        fused_channels = int(fused_channels)
        norm = str(norm)

        if in_channels <= 0:
            raise ValueError(f"in_channels must be > 0, got {in_channels}")
        if d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {d_model}")
        if num_stages <= 0:
            raise ValueError(f"num_stages must be > 0, got {num_stages}")
        if fused_channels <= 0:
            raise ValueError(f"fused_channels must be > 0, got {fused_channels}")

        self.attn = MRTCrossSpectrumAttention2D(
            in_channels=in_channels,
            d_model=d_model,
            num_stages=num_stages,
            use_pos_encoding=bool(use_pos_encoding),
        )

        self.fused_proj: nn.Module
        if fused_channels == in_channels:
            self.fused_proj = nn.Identity()
        else:
            self.fused_proj = nn.Conv2d(in_channels, fused_channels, kernel_size=1, bias=False)

        out_channels = in_channels
        self.redim = nn.Sequential(
            nn.Conv2d(2 * in_channels + fused_channels, out_channels, kernel_size=1, bias=False),
            _make_norm2d(norm, out_channels),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        if x1.shape != x2.shape:
            raise ValueError(f"MRTCrossSpectrumCorrFusion2D expects same shapes, got x1={x1.shape} x2={x2.shape}")
        if x1.ndim != 4:
            raise ValueError(f"MRTCrossSpectrumCorrFusion2D expects BCHW tensors, got {x1.shape}")

        fused = self.attn(x1, x2)
        fused = self.fused_proj(fused)
        return self.redim(torch.cat([x1, x2, fused], dim=1))


__all__ = ["MRTCrossSpectrumAttention2D", "MRTCrossSpectrumFusion2D", "MRTCrossSpectrumCorrFusion2D"]
