from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _GptSelfAttention(nn.Module):
    def __init__(self, *, d_model: int, nhead: int, attn_pdrop: float, resid_pdrop: float) -> None:
        super().__init__()
        d_model = int(d_model)
        nhead = int(nhead)
        attn_pdrop = float(attn_pdrop)
        resid_pdrop = float(resid_pdrop)

        if d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {d_model}")
        if nhead <= 0:
            raise ValueError(f"nhead must be > 0, got {nhead}")
        if d_model % nhead != 0:
            raise ValueError(f"d_model must be divisible by nhead, got d_model={d_model} nhead={nhead}")
        if attn_pdrop < 0 or resid_pdrop < 0:
            raise ValueError(f"dropout must be >= 0, got attn_pdrop={attn_pdrop} resid_pdrop={resid_pdrop}")

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.attn_pdrop = attn_pdrop

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.resid_drop = nn.Dropout(resid_pdrop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"self-attn expects (B,L,D), got {x.shape}")
        b, l, d = x.shape
        if d != self.d_model:
            raise ValueError(f"self-attn expects D={self.d_model}, got {d}")

        qkv = self.qkv(x)  # (B, L, 3D)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(b, l, self.nhead, self.head_dim).transpose(1, 2)  # (B, nh, L, hd)
        k = k.view(b, l, self.nhead, self.head_dim).transpose(1, 2)  # (B, nh, L, hd)
        v = v.view(b, l, self.nhead, self.head_dim).transpose(1, 2)  # (B, nh, L, hd)

        dropout_p = self.attn_pdrop if self.training and self.attn_pdrop > 0 else 0.0
        orig_dtype = q.dtype
        if orig_dtype in (torch.float16, torch.bfloat16):
            q = q.float()
            k = k.float()
            v = v.float()
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=False)
        if out.dtype != orig_dtype:
            out = out.to(dtype=orig_dtype)
        out = out.transpose(1, 2).contiguous().view(b, l, d)  # (B, L, D)
        out = self.out_proj(out)
        out = self.resid_drop(out)
        return out


class _GptTransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        nhead: int,
        block_exp: int,
        attn_pdrop: float,
        resid_pdrop: float,
    ) -> None:
        super().__init__()
        d_model = int(d_model)
        block_exp = int(block_exp)

        if block_exp <= 0:
            raise ValueError(f"block_exp must be > 0, got {block_exp}")

        self.input_norm = nn.LayerNorm(d_model)
        self.output_norm = nn.LayerNorm(d_model)
        self.attn = _GptSelfAttention(d_model=d_model, nhead=nhead, attn_pdrop=attn_pdrop, resid_pdrop=resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, block_exp * d_model),
            nn.GELU(),
            nn.Linear(block_exp * d_model, d_model),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.input_norm(x))
        x = x + self.mlp(self.output_norm(x))
        return x


class GPTFusion2D(nn.Module):
    """
    Cross-Modality Fusion Transformer (CFT) / GPT-style fusion for 2D feature maps.

    Adapted from `third_party/multispectral-object-detection/models/common.py::GPT`:
    - AvgPool both modalities to a fixed token grid (vert_anchors x horz_anchors)
    - Concatenate tokens (rgb + ms) and run multi-layer self-attention
    - Split tokens back to per-modality feature maps and upsample to original resolution
    - Write-back fused features to both modalities with configurable merge:
      - add: rgb=rgb+rgb_out, ms=ms+ms_out
      - wavg: rgb=(1-w)*rgb+w*rgb_out, ms=(1-w)*ms+w*ms_out (learnable scalar gates)
      - concat1x1: rgb=rgb+proj(cat([rgb,rgb_out])), ms=ms+proj(cat([ms,ms_out])) (1x1 projection)
    """

    def __init__(
        self,
        *,
        in_channels: int,
        d_model: int | None = None,
        nhead: int = 8,
        block_exp: int = 4,
        n_layer: int = 8,
        vert_anchors: int = 8,
        horz_anchors: int = 8,
        dropout: float = 0.1,
        writeback_merge: str = "add",
        alpha_init: float = 0.0,
    ) -> None:
        super().__init__()
        in_channels = int(in_channels)
        nhead = int(nhead)
        block_exp = int(block_exp)
        n_layer = int(n_layer)
        vert_anchors = int(vert_anchors)
        horz_anchors = int(horz_anchors)
        dropout = float(dropout)
        writeback_merge = str(writeback_merge).strip().lower()
        alpha_init = float(alpha_init)

        if in_channels <= 0:
            raise ValueError(f"in_channels must be > 0, got {in_channels}")
        if d_model is None:
            d_model = in_channels
        d_model = int(d_model)
        if d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {d_model}")
        if nhead <= 0 or d_model % nhead != 0:
            raise ValueError(f"Invalid nhead/d_model: d_model={d_model} nhead={nhead}")
        if n_layer <= 0:
            raise ValueError(f"n_layer must be > 0, got {n_layer}")
        if vert_anchors <= 0 or horz_anchors <= 0:
            raise ValueError(f"anchors must be > 0, got vert={vert_anchors} horz={horz_anchors}")
        if dropout < 0:
            raise ValueError(f"dropout must be >= 0, got {dropout}")

        self.in_channels = in_channels
        self.d_model = d_model
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors

        if writeback_merge == "avg":
            writeback_merge = "wavg"
        if writeback_merge in {"concat", "concat_1x1", "concat-1x1", "contact"}:
            writeback_merge = "concat1x1"
        self.writeback_merge = writeback_merge

        if self.writeback_merge == "wavg":
            # Learnable weighted average: out = (1-w)*x + w*fused, where w=sigmoid(gate) in (0,1).
            # Initialize gate=0 -> w=0.5 (plain avg) by default.
            self.merge_gate_rgb = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
            self.merge_gate_ms = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
            self.concat_proj_rgb = None
            self.concat_proj_ms = None
        elif self.writeback_merge == "concat1x1":
            self.merge_gate_rgb = None
            self.merge_gate_ms = None
            self.concat_proj_rgb = nn.Conv2d(2 * in_channels, in_channels, kernel_size=1, bias=False)
            self.concat_proj_ms = nn.Conv2d(2 * in_channels, in_channels, kernel_size=1, bias=False)
            nn.init.zeros_(self.concat_proj_rgb.weight)
            nn.init.zeros_(self.concat_proj_ms.weight)
        elif self.writeback_merge == "add":
            self.merge_gate_rgb = None
            self.merge_gate_ms = None
            self.concat_proj_rgb = None
            self.concat_proj_ms = None
        else:
            raise ValueError(f"Unsupported writeback_merge: {self.writeback_merge}")

        self.avgpool = nn.AdaptiveAvgPool2d((vert_anchors, horz_anchors))

        self.in_proj: nn.Module
        self.out_proj: nn.Module
        if d_model == in_channels:
            self.in_proj = nn.Identity()
            self.out_proj = nn.Identity()
        else:
            self.in_proj = nn.Conv2d(in_channels, d_model, kernel_size=1, bias=False)
            self.out_proj = nn.Conv2d(d_model, in_channels, kernel_size=1, bias=False)

        token_len = 2 * vert_anchors * horz_anchors
        self.pos_emb = nn.Parameter(torch.zeros(1, token_len, d_model))
        self.drop = nn.Dropout(dropout)
        self.trans_blocks = nn.Sequential(
            *[
                _GptTransformerBlock(
                    d_model=d_model,
                    nhead=nhead,
                    block_exp=block_exp,
                    attn_pdrop=dropout,
                    resid_pdrop=dropout,
                )
                for _ in range(n_layer)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, rgb: torch.Tensor, ms: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if rgb.shape != ms.shape:
            raise ValueError(f"GPTFusion2D expects same shapes, got rgb={rgb.shape} ms={ms.shape}")
        if rgb.ndim != 4:
            raise ValueError(f"GPTFusion2D expects BCHW tensors, got {rgb.shape}")

        b, c, h, w = rgb.shape
        if c != self.in_channels:
            raise ValueError(f"GPTFusion2D expects C={self.in_channels}, got {c}")

        rgb_pooled = self.avgpool(rgb)
        ms_pooled = self.avgpool(ms)
        rgb_pooled = self.in_proj(rgb_pooled)
        ms_pooled = self.in_proj(ms_pooled)

        # (B, D, Va, Ha) -> (B, 2*Va*Ha, D)
        rgb_tokens = rgb_pooled.flatten(2)
        ms_tokens = ms_pooled.flatten(2)
        tokens = torch.cat([rgb_tokens, ms_tokens], dim=2).transpose(1, 2).contiguous()

        x = self.drop(tokens + self.pos_emb)
        x = self.trans_blocks(x)
        x = self.final_norm(x)

        # (B, 2*Va*Ha, D) -> (B, 2, D, Va, Ha)
        x = x.view(b, 2, self.vert_anchors, self.horz_anchors, self.d_model).permute(0, 1, 4, 2, 3).contiguous()
        rgb_out = x[:, 0]
        ms_out = x[:, 1]

        rgb_out = self.out_proj(rgb_out)
        ms_out = self.out_proj(ms_out)

        rgb_out = F.interpolate(rgb_out, size=(h, w), mode="bilinear", align_corners=False)
        ms_out = F.interpolate(ms_out, size=(h, w), mode="bilinear", align_corners=False)

        if self.writeback_merge == "wavg":
            assert self.merge_gate_rgb is not None and self.merge_gate_ms is not None
            w_rgb = torch.sigmoid(self.merge_gate_rgb).to(dtype=rgb.dtype, device=rgb.device)
            w_ms = torch.sigmoid(self.merge_gate_ms).to(dtype=ms.dtype, device=ms.device)
            rgb = rgb * (1.0 - w_rgb) + rgb_out * w_rgb
            ms = ms * (1.0 - w_ms) + ms_out * w_ms
        elif self.writeback_merge == "concat1x1":
            assert self.concat_proj_rgb is not None and self.concat_proj_ms is not None
            rgb = rgb + self.concat_proj_rgb(torch.cat([rgb, rgb_out], dim=1))
            ms = ms + self.concat_proj_ms(torch.cat([ms, ms_out], dim=1))
        elif self.writeback_merge == "add":
            rgb = rgb + rgb_out
            ms = ms + ms_out
        else:
            raise RuntimeError(f"Unsupported writeback_merge: {self.writeback_merge}")
        return rgb, ms


__all__ = ["GPTFusion2D"]
