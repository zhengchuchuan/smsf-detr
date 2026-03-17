from __future__ import annotations

import torch
import torch.nn as nn

from .deform_align import DeformableAlign2D

__all__ = ["FixedBandCMDA"]


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


class _AnchorAwareFusion(nn.Module):
    """
    Fuse an aligned support feature with the fixed anchor feature.

    The block is initialized near identity so v1 behaves like "align first" before learning
    stronger anchor-conditioned fusion.
    """

    def __init__(self, channels: int, *, hidden_channels: int | None = None) -> None:
        super().__init__()
        c = int(channels)
        hidden = int(hidden_channels) if hidden_channels is not None else c
        if c <= 0:
            raise ValueError(f"channels must be > 0, got {channels}")
        if hidden <= 0:
            raise ValueError(f"hidden_channels must be > 0, got {hidden_channels}")

        self.query_proj = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=1, bias=False),
            _make_gn(c),
            nn.ReLU(inplace=True),
        )
        self.gate = nn.Conv2d(c * 2, c, kernel_size=1, bias=True)
        self.mix = nn.Sequential(
            nn.Conv2d(c * 2, hidden, kernel_size=1, bias=False),
            _make_gn(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, c, kernel_size=1, bias=True),
        )

        # Start from identity: output ~= aligned support at initialization.
        last = self.mix[-1]
        assert isinstance(last, nn.Conv2d)
        nn.init.constant_(last.weight, 0.0)
        if last.bias is not None:
            nn.init.constant_(last.bias, 0.0)

    def forward(self, anchor: torch.Tensor, aligned_support: torch.Tensor) -> torch.Tensor:
        if anchor.shape != aligned_support.shape:
            raise ValueError(f"Anchor/support shape mismatch: anchor={anchor.shape} support={aligned_support.shape}")
        anchor_query = self.query_proj(anchor)
        gate = torch.sigmoid(self.gate(torch.cat([anchor_query, aligned_support], dim=1)))
        delta = self.mix(torch.cat([aligned_support, anchor_query * gate], dim=1))
        return aligned_support + delta


class FixedBandCMDA(nn.Module):
    """
    Single-scale, fixed-anchor CMDA for explicit multi-band features.

    Input:
        x: (B, N, C, H, W), where N is the band count.

    Behavior:
        - choose one fixed anchor band
        - align every support band to the anchor grid with shared deformable alignment
        - fuse aligned support with anchor in anchor coordinates
        - keep the anchor band explicit in the output
    """

    def __init__(
        self,
        *,
        in_channels: int,
        anchor_band_index: int | str | None = None,
        num_iters: int = 1,
        anchor_detach: bool = False,
        num_keypoints: int = 5,
        offset_scale: float = 6.0,
        offset_enabled: bool = True,
        attention_norm: str = "sigmoid",
        padding_mode: str = "border",
        align_corners: bool = True,
        loss_type: str = "cosine",
        loss_downsample: float | None = None,
        nce_num_patches: int = 64,
        nce_patch_size: int = 5,
        nce_tau: float = 0.2,
        infonce_weight: float = 1.0,
        lncc_weight: float = 0.2,
        lncc_window_size: int = 5,
        lncc_eps: float = 1e-6,
        affine_enabled: bool = False,
        affine_scale: float = 0.1,
        affine_init_identity: bool = True,
        affine_type: str = "affine",
        loss_weight: float = 1.0,
        loss_offset_weight: float = 0.0,
        loss_attn_norm_weight: float = 0.0,
        loss_attn_entropy_weight: float = 0.0,
        fuse_hidden_channels: int | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        if self.in_channels <= 0:
            raise ValueError(f"in_channels must be > 0, got {in_channels}")

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

        self.num_iters = int(num_iters)
        if self.num_iters <= 0:
            raise ValueError(f"num_iters must be > 0, got {num_iters}")
        self.anchor_detach = bool(anchor_detach)
        self.loss_weight = float(loss_weight)
        self.loss_offset_weight = float(loss_offset_weight)
        self.loss_attn_norm_weight = float(loss_attn_norm_weight)
        self.loss_attn_entropy_weight = float(loss_attn_entropy_weight)
        if self.loss_weight < 0:
            raise ValueError(f"loss_weight must be >= 0, got {loss_weight}")
        if self.loss_offset_weight < 0:
            raise ValueError(f"loss_offset_weight must be >= 0, got {loss_offset_weight}")
        if self.loss_attn_norm_weight < 0:
            raise ValueError(f"loss_attn_norm_weight must be >= 0, got {loss_attn_norm_weight}")
        if self.loss_attn_entropy_weight < 0:
            raise ValueError(f"loss_attn_entropy_weight must be >= 0, got {loss_attn_entropy_weight}")

        self.aligner = DeformableAlign2D(
            in_channels=self.in_channels,
            num_keypoints=int(num_keypoints),
            offset_scale=float(offset_scale),
            offset_enabled=bool(offset_enabled),
            per_channel_offset=False,
            attention_norm=str(attention_norm),
            padding_mode=str(padding_mode),
            align_corners=bool(align_corners),
            loss_type=str(loss_type),
            loss_downsample=loss_downsample,
            nce_num_patches=int(nce_num_patches),
            nce_patch_size=int(nce_patch_size),
            nce_tau=float(nce_tau),
            infonce_weight=float(infonce_weight),
            lncc_weight=float(lncc_weight),
            lncc_window_size=int(lncc_window_size),
            lncc_eps=float(lncc_eps),
            affine_enabled=bool(affine_enabled),
            affine_scale=float(affine_scale),
            affine_init_identity=bool(affine_init_identity),
            affine_per_channel=False,
            affine_type=str(affine_type),
        )
        self.fuse = _AnchorAwareFusion(self.in_channels, hidden_channels=fuse_hidden_channels)

    @staticmethod
    def _safe_normalize_attention(attn: torch.Tensor) -> torch.Tensor:
        if attn.ndim != 4:
            raise ValueError(f"Expected attention tensor shaped (B,K,H,W), got {attn.shape}")
        denom = attn.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return attn / denom

    def _resolve_anchor_index(self, num_bands: int) -> int:
        if num_bands <= 0:
            raise ValueError(f"num_bands must be > 0, got {num_bands}")
        if self.anchor_band_index is None:
            return int(num_bands // 2)
        idx = self.anchor_band_index if self.anchor_band_index >= 0 else num_bands + self.anchor_band_index
        if idx < 0 or idx >= num_bands:
            raise ValueError(f"anchor_band_index out of range: idx={idx}, num_bands={num_bands}")
        return int(idx)

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if x.ndim != 5:
            raise ValueError(f"FixedBandCMDA expects (B,N,C,H,W) tensor, got shape={tuple(x.shape)}")
        _, n, c, _, _ = x.shape
        if c != self.in_channels:
            raise ValueError(f"Channel mismatch: expected C={self.in_channels}, got C={c}")
        if n < 2:
            return x

        anchor_idx = self._resolve_anchor_index(n)
        base_anchor = x[:, anchor_idx]
        cur = x

        aux_losses: dict[str, torch.Tensor] = {}
        it_align_losses = []
        it_offset_losses = []
        it_attn_norm_losses = []
        it_attn_entropy_losses = []

        for _ in range(self.num_iters):
            anchor = base_anchor
            anchor_pred = anchor.detach() if self.anchor_detach else anchor
            aligned_bands: list[torch.Tensor] = []
            band_align_losses = []
            band_offset_losses = []
            band_attn_norm_losses = []
            band_attn_entropy_losses = []

            for band_idx in range(n):
                if band_idx == anchor_idx:
                    aligned_bands.append(anchor)
                    continue

                src = cur[:, band_idx]
                pred = self.aligner.predict(anchor_pred, src)
                if self.aligner.affine_enabled and len(pred) == 4:
                    offset_x, offset_y, attn_weights, affine_theta = pred
                elif len(pred) == 3:
                    offset_x, offset_y, attn_weights = pred
                    affine_theta = None
                else:
                    raise RuntimeError(f"Unexpected predict output length: {len(pred)}")

                aligned, sampled_features, attn_exp = self.aligner.deform_with_attention(
                    src,
                    offset_x=offset_x,
                    offset_y=offset_y,
                    attention_weights=attn_weights,
                    affine_theta=affine_theta,
                )
                fused = self.fuse(anchor, aligned)
                aligned_bands.append(fused)

                if self.training and self.loss_weight > 0:
                    loss_dict = self.aligner.loss_calculate(
                        anchor,
                        offset_x,
                        offset_y,
                        aligned,
                        attn_exp,
                        affine_theta=affine_theta,
                        sampled_features=sampled_features,
                    )
                    band_align_losses.append(loss_dict["loss_deform_align"])

                if self.training and (
                    self.loss_offset_weight > 0
                    or self.loss_attn_norm_weight > 0
                    or self.loss_attn_entropy_weight > 0
                ):
                    _, k, hh, ww = attn_weights.shape
                    if self.loss_offset_weight > 0:
                        denom_x = max(int(ww) - 1, 1) / 2.0
                        denom_y = max(int(hh) - 1, 1) / 2.0
                        offset_x_px = offset_x * float(denom_x)
                        offset_y_px = offset_y * float(denom_y)
                        p = self._safe_normalize_attention(attn_weights)
                        fused_x = (p * offset_x_px).sum(dim=1)
                        fused_y = (p * offset_y_px).sum(dim=1)
                        band_offset_losses.append(torch.sqrt(fused_x ** 2 + fused_y ** 2 + 1e-8).mean())

                    if self.loss_attn_norm_weight > 0:
                        attn_sum = attn_weights.sum(dim=1)
                        band_attn_norm_losses.append(((attn_sum - 1.0) ** 2).mean())

                    if self.loss_attn_entropy_weight > 0:
                        p = self._safe_normalize_attention(attn_weights)
                        ent = -(p * torch.log(p.clamp_min(1e-8))).sum(dim=1).mean()
                        band_attn_entropy_losses.append(ent)

            cur = torch.stack(aligned_bands, dim=1)
            if band_align_losses:
                it_align_losses.append(torch.stack(band_align_losses).mean())
            if band_offset_losses:
                it_offset_losses.append(torch.stack(band_offset_losses).mean())
            if band_attn_norm_losses:
                it_attn_norm_losses.append(torch.stack(band_attn_norm_losses).mean())
            if band_attn_entropy_losses:
                it_attn_entropy_losses.append(torch.stack(band_attn_entropy_losses).mean())

        if self.training:
            if it_align_losses and self.loss_weight > 0:
                aux_losses["loss_ms_group_align"] = torch.stack(it_align_losses).mean() * self.loss_weight
            if it_offset_losses and self.loss_offset_weight > 0:
                aux_losses["loss_ms_group_offset"] = torch.stack(it_offset_losses).mean() * self.loss_offset_weight
            if it_attn_norm_losses and self.loss_attn_norm_weight > 0:
                aux_losses["loss_ms_group_attn"] = (
                    torch.stack(it_attn_norm_losses).mean() * self.loss_attn_norm_weight
                )
            if it_attn_entropy_losses and self.loss_attn_entropy_weight > 0:
                aux_losses["loss_ms_group_attn_entropy"] = (
                    torch.stack(it_attn_entropy_losses).mean() * self.loss_attn_entropy_weight
                )
            if aux_losses:
                return cur, aux_losses
        return cur
