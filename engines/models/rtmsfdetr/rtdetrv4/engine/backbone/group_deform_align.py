from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from .deform_align import DeformableAlign2D
from ..core import register

__all__ = [
    # Paper name: Canonical Reference Guided Groupwise Alignment (CRGGA).
    "CRGGA",
    "ProjectedCRGGA",
    # Backwards compatible names (keep existing imports/configs working).
    "GroupwiseDeformableAlign2D",
    "ProjectedGroupwiseDeformableAlign2D",
]


@register()
@register(name="GroupwiseDeformableAlign2D")
class CRGGA(nn.Module):
    """
    Canonical Reference Guided Groupwise Alignment (CRGGA).

    This module was previously named `GroupwiseDeformableAlign2D`. The implementation is unchanged; only the
    class name is updated for clearer "paper naming". A backwards compatible alias is provided.

    This is an extension of MRT-DETR's EDA idea: instead of aligning a single source to a fixed reference
    (e.g. IR -> RGB), we build a *learned canonical reference* from all bands, then align every band to it.

    Input:
        x: Tensor shaped (B, N, C, H, W), where:
            - N: number of bands (e.g. 7 for MS)
            - C: channels per band feature (can be 1 for raw bands, or >1 for per-band features)

    Output:
        - during eval: aligned x, shaped (B, N, C, H, W)
        - during train: (aligned x, aux_losses)

    Notes:
        - This module does NOT magically "unmix" bands if you have already fused them with a regular conv stem.
          It is intended for tensors where the band dimension is still explicit (N axis).
        - Set `ref_mode="spatial_weighted"` when different bands dominate at different spatial locations.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        ref_mode: str = "spatial_weighted",
        ref_band_index: int | str | None = None,
        num_iters: int = 1,
        ref_detach: bool = False,
        # DeformableAlign2D params (shared across bands)
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
        # Loss control (similarity)
        loss_weight: float = 1.0,
        # Regularizers (optional; set weights > 0 to enable)
        loss_offset_weight: float = 0.0,
        loss_attn_norm_weight: float = 0.0,
        loss_attn_entropy_weight: float = 0.0,
    ) -> None:
        super().__init__()
        in_channels = int(in_channels)
        if in_channels <= 0:
            raise ValueError(f"in_channels must be > 0, got {in_channels}")
        ref_mode_norm = str(ref_mode).strip().lower()
        if ref_mode_norm in {"fixed", "single_band", "band"}:
            ref_mode_norm = "fixed_band"
        if ref_mode_norm not in {"mean", "global_weighted", "spatial_weighted", "fixed_band"}:
            raise ValueError(
                f"Unsupported ref_mode={ref_mode} (supported: mean|global_weighted|spatial_weighted|fixed_band)"
            )
        self.in_channels = in_channels
        self.ref_mode = ref_mode_norm
        self.ref_band_index: int | None
        if ref_band_index is None:
            self.ref_band_index = None
        elif isinstance(ref_band_index, str):
            index_norm = ref_band_index.strip().lower()
            if index_norm in {"mid", "middle", "center", "centre", "auto"}:
                self.ref_band_index = None
            else:
                self.ref_band_index = int(ref_band_index)
        else:
            self.ref_band_index = int(ref_band_index)
        self.num_iters = int(num_iters)
        if self.num_iters <= 0:
            raise ValueError(f"num_iters must be > 0, got {num_iters}")
        self.ref_detach = bool(ref_detach)
        self.loss_weight = float(loss_weight)
        if self.loss_weight < 0:
            raise ValueError(f"loss_weight must be >= 0, got {loss_weight}")
        self.loss_offset_weight = float(loss_offset_weight)
        self.loss_attn_norm_weight = float(loss_attn_norm_weight)
        self.loss_attn_entropy_weight = float(loss_attn_entropy_weight)
        if self.loss_offset_weight < 0:
            raise ValueError(f"loss_offset_weight must be >= 0, got {loss_offset_weight}")
        if self.loss_attn_norm_weight < 0:
            raise ValueError(f"loss_attn_norm_weight must be >= 0, got {loss_attn_norm_weight}")
        if self.loss_attn_entropy_weight < 0:
            raise ValueError(f"loss_attn_entropy_weight must be >= 0, got {loss_attn_entropy_weight}")

        # Shared aligner applied band-by-band (same weights).
        self.aligner = DeformableAlign2D(
            in_channels=in_channels,
            num_keypoints=int(num_keypoints),
            offset_scale=float(offset_scale),
            offset_enabled=bool(offset_enabled),
            per_channel_offset=False,  # offsets are per-band; channels inside a band share the same warp
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

        # Build a canonical reference from all bands (no fixed "reference band").
        if self.ref_mode == "global_weighted":
            hidden = max(16, in_channels // 2)
            self.ref_mlp = nn.Sequential(
                nn.Linear(in_channels, hidden, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, 1, bias=True),
            )
        else:
            self.ref_mlp = None

        if self.ref_mode == "spatial_weighted":
            self.ref_conv = nn.Conv2d(in_channels, 1, kernel_size=1, bias=True)
        else:
            self.ref_conv = None

    @staticmethod
    def _safe_normalize_attention(attn: torch.Tensor) -> torch.Tensor:
        """
        Normalize attention weights across the keypoint dimension (K) to sum to 1.

        Expected input shape: (B, K, H, W).
        """
        if attn.ndim != 4:
            raise ValueError(f"Expected attention tensor shaped (B,K,H,W), got {attn.shape}")
        denom = attn.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return attn / denom

    def _compute_reference(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C, H, W) -> ref: (B, C, H, W)
        if self.ref_mode == "mean":
            return x.mean(dim=1)

        b, n, c, h, w = x.shape
        if self.ref_mode == "fixed_band":
            if self.ref_band_index is None:
                idx = n // 2
            else:
                idx = self.ref_band_index if self.ref_band_index >= 0 else n + self.ref_band_index
            if idx < 0 or idx >= n:
                raise ValueError(f"ref_band_index out of range: idx={idx}, num_bands={n}")
            return x[:, idx]

        if self.ref_mode == "global_weighted":
            assert self.ref_mlp is not None
            pooled = x.mean(dim=(3, 4))  # (B, N, C)
            logits = self.ref_mlp(pooled).squeeze(-1)  # (B, N)
            weights = torch.softmax(logits, dim=1).view(b, n, 1, 1, 1)
            return (x * weights).sum(dim=1)

        assert self.ref_mode == "spatial_weighted"
        assert self.ref_conv is not None
        scores = self.ref_conv(x.reshape(b * n, c, h, w)).reshape(b, n, h, w)  # (B, N, H, W)
        weights = torch.softmax(scores, dim=1).unsqueeze(2)  # (B, N, 1, H, W)
        return (x * weights).sum(dim=1)

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_debug: bool = False,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, dict[str, torch.Tensor]]
        | tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]
    ):
        if x.ndim != 5:
            raise ValueError(
                f"CRGGA expects (B,N,C,H,W) tensor, got shape={tuple(x.shape)}"
            )
        b, n, c, h, w = x.shape
        if c != self.in_channels:
            raise ValueError(f"Channel mismatch: expected C={self.in_channels}, got C={c}")
        if n < 2:
            if return_debug:
                return x, {}, {}
            return x

        aux_losses: dict[str, torch.Tensor] = {}
        debug: dict[str, torch.Tensor] | None = None
        cur = x
        it_align_losses = []
        it_offset_losses = []
        it_attn_norm_losses = []
        it_attn_entropy_losses = []
        for it_idx in range(self.num_iters):
            ref = self._compute_reference(cur)  # (B,C,H,W)
            ref_pred = ref.detach() if self.ref_detach else ref

            aligned_list = []
            band_align_losses = []
            band_offset_losses = []
            band_attn_norm_losses = []
            band_attn_entropy_losses = []
            capture_debug = return_debug and (it_idx == self.num_iters - 1)
            offset_x_list: list[torch.Tensor] = []
            offset_y_list: list[torch.Tensor] = []
            attn_list: list[torch.Tensor] = []
            affine_list: list[torch.Tensor] = []
            for i in range(n):
                src = cur[:, i]
                pred = self.aligner.predict(ref_pred, src)
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
                aligned_list.append(aligned)
                if capture_debug:
                    offset_x_list.append(offset_x.detach())
                    offset_y_list.append(offset_y.detach())
                    attn_list.append(attn_weights.detach())
                    if affine_theta is not None:
                        affine_list.append(affine_theta.detach())

                if self.training and (self.loss_weight > 0):
                    loss_dict = self.aligner.loss_calculate(
                        ref,
                        offset_x,
                        offset_y,
                        aligned,
                        attn_exp,
                        affine_theta=affine_theta,
                        sampled_features=sampled_features,
                    )
                    band_align_losses.append(loss_dict["loss_deform_align"])

                if self.training and (self.loss_offset_weight > 0 or self.loss_attn_norm_weight > 0 or self.loss_attn_entropy_weight > 0):
                    # Regularizers follow MRT-DETR's EDA intuition:
                    # - penalize large offsets (avoid destructive warps)
                    # - control attention distribution (avoid unstable scaling / degenerate sampling)
                    bsz, k, hh, ww = attn_weights.shape
                    # Offset magnitude in *pixel* units (consistent with MRT-DETR's offset_loss).
                    if self.loss_offset_weight > 0:
                        denom_x = max(int(ww) - 1, 1) / 2.0
                        denom_y = max(int(hh) - 1, 1) / 2.0
                        offset_x_px = offset_x * float(denom_x)
                        offset_y_px = offset_y * float(denom_y)
                        p = self._safe_normalize_attention(attn_weights)
                        fused_x = (p * offset_x_px).sum(dim=1)
                        fused_y = (p * offset_y_px).sum(dim=1)
                        band_offset_losses.append(torch.sqrt(fused_x ** 2 + fused_y ** 2 + 1e-8).mean())

                    # Encourage attention weights to have a stable sum (important when attention_norm='sigmoid').
                    if self.loss_attn_norm_weight > 0:
                        attn_sum = attn_weights.sum(dim=1)
                        band_attn_norm_losses.append(((attn_sum - 1.0) ** 2).mean())

                    # Attention entropy penalty (lower entropy => more selective keypoint).
                    if self.loss_attn_entropy_weight > 0:
                        p = self._safe_normalize_attention(attn_weights)
                        ent = -(p * torch.log(p.clamp_min(1e-8))).sum(dim=1).mean()
                        band_attn_entropy_losses.append(ent)

            cur = torch.stack(aligned_list, dim=1)
            if band_align_losses:
                it_align_losses.append(torch.stack(band_align_losses).mean())
            if band_offset_losses:
                it_offset_losses.append(torch.stack(band_offset_losses).mean())
            if band_attn_norm_losses:
                it_attn_norm_losses.append(torch.stack(band_attn_norm_losses).mean())
            if band_attn_entropy_losses:
                it_attn_entropy_losses.append(torch.stack(band_attn_entropy_losses).mean())

            if capture_debug:
                debug = {
                    "ref": ref.detach(),
                    "aligned": cur.detach(),
                    "offset_x": torch.stack(offset_x_list, dim=1),
                    "offset_y": torch.stack(offset_y_list, dim=1),
                    "attn": torch.stack(attn_list, dim=1),
                }
                if affine_list:
                    debug["affine_theta"] = torch.stack(affine_list, dim=1)

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
                if return_debug:
                    return cur, aux_losses, debug or {}
                return cur, aux_losses
        if return_debug:
            return cur, aux_losses, debug or {}
        return cur


@register()
@register(name="ProjectedGroupwiseDeformableAlign2D")
class ProjectedCRGGA(nn.Module):
    """
    Projected variant of CRGGA.

    This is a pragmatic way to use groupwise alignment at feature stages where the original band dimension
    is no longer explicit. We project (B,C,H,W) -> (B,N,Cg,H,W), align the N groups to a canonical reference,
    then project back to C channels. A residual connection keeps the module near-identity at init.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        num_groups: int,
        group_channels: int,
        ref_mode: str = "spatial_weighted",
        ref_band_index: int | str | None = None,
        num_iters: int = 1,
        ref_detach: bool = False,
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
        affine_enabled: bool = False,
        affine_scale: float = 0.1,
        affine_init_identity: bool = True,
        affine_type: str = "affine",
        loss_weight: float = 1.0,
        loss_offset_weight: float = 0.0,
        loss_attn_norm_weight: float = 0.0,
        loss_attn_entropy_weight: float = 0.0,
    ) -> None:
        super().__init__()
        in_channels = int(in_channels)
        num_groups = int(num_groups)
        group_channels = int(group_channels)
        if in_channels <= 0:
            raise ValueError(f"in_channels must be > 0, got {in_channels}")
        if num_groups <= 1:
            raise ValueError(f"num_groups must be > 1, got {num_groups}")
        if group_channels <= 0:
            raise ValueError(f"group_channels must be > 0, got {group_channels}")

        self.in_channels = in_channels
        self.num_groups = num_groups
        self.group_channels = group_channels

        proj_dim = int(num_groups * group_channels)
        self.proj_in = nn.Conv2d(in_channels, proj_dim, kernel_size=1, bias=False)
        self.group_aligner = GroupwiseDeformableAlign2D(
            in_channels=group_channels,
            ref_mode=ref_mode,
            ref_band_index=ref_band_index,
            num_iters=int(num_iters),
            ref_detach=bool(ref_detach),
            num_keypoints=int(num_keypoints),
            offset_scale=float(offset_scale),
            offset_enabled=bool(offset_enabled),
            attention_norm=str(attention_norm),
            padding_mode=str(padding_mode),
            align_corners=bool(align_corners),
            loss_type=str(loss_type),
            loss_downsample=loss_downsample,
            nce_num_patches=int(nce_num_patches),
            nce_patch_size=int(nce_patch_size),
            nce_tau=float(nce_tau),
            affine_enabled=bool(affine_enabled),
            affine_scale=float(affine_scale),
            affine_init_identity=bool(affine_init_identity),
            affine_type=str(affine_type),
            loss_weight=float(loss_weight),
            loss_offset_weight=float(loss_offset_weight),
            loss_attn_norm_weight=float(loss_attn_norm_weight),
            loss_attn_entropy_weight=float(loss_attn_entropy_weight),
        )
        self.proj_out = nn.Conv2d(proj_dim, in_channels, kernel_size=1, bias=True)
        # Start from identity (via residual): proj_out=0 => out=x.
        nn.init.constant_(self.proj_out.weight, 0.0)
        if self.proj_out.bias is not None:
            nn.init.constant_(self.proj_out.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if x.ndim != 4:
            raise ValueError(f"ProjectedCRGGA expects BCHW tensor, got {x.shape}")
        b, c, h, w = x.shape
        if c != self.in_channels:
            raise ValueError(f"Channel mismatch: expected C={self.in_channels}, got C={c}")

        z = self.proj_in(x).view(b, self.num_groups, self.group_channels, h, w)
        out = self.group_aligner(z)
        if self.training and isinstance(out, tuple) and len(out) >= 2 and torch.is_tensor(out[0]) and isinstance(out[1], dict):
            z_aligned = out[0]
            aux_losses = out[1]
        else:
            z_aligned = out[0] if isinstance(out, tuple) and len(out) > 0 else out
            aux_losses = {}
        if not torch.is_tensor(z_aligned):
            raise RuntimeError("Unexpected aligner output type in ProjectedCRGGA")
        delta = self.proj_out(z_aligned.reshape(b, self.num_groups * self.group_channels, h, w))
        y = x + delta
        if self.training and aux_losses:
            return y, aux_losses
        return y


# Backwards compatible aliases: keep old class names importable and keep registry/create() lookups working.
GroupwiseDeformableAlign2D = CRGGA
ProjectedGroupwiseDeformableAlign2D = ProjectedCRGGA
