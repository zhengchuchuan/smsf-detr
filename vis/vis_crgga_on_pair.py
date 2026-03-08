#!/usr/bin/env python
"""
Visualize CRGGA (GroupwiseDeformableAlign2D) alignment effects on a single RGB+MS sample.

This script runs ONLY the MSBandSeparatedStemAlign module (per-band embed + CRGGA),
and saves feature-level visualizations:
  - per-band embedded feature before/after alignment (channel-mean)
  - cosine similarity to canonical reference (before/after + delta)
  - fused offset flow magnitude (expected dx/dy under attention)
  - attention entropy
  - (optional) spatial_weighted reference band selection map

Notes:
  - CRGGA alignment happens on stride=4 feature grid (after two stride=2 convs), not on raw pixels.
  - The "image-level warp" is NOT produced here, because feature offsets are learned in feature space.
    You can still approximate it by upsampling the fused flow and multiplying by stride, but that may
    be misleading if interpreted as raw-pixel registration.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    # Allow running the script via absolute path: `python /abs/path/tools/vis_crgga_on_pair.py`.
    sys.path.insert(0, str(REPO_ROOT))

from engines.models.rtmsfdetr.rtdetrv4.engine.backbone.ms_band_sep import MSBandSeparatedStemAlign


def _load_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0  # (H,W,3) in [0,1]
    return arr


def _load_ms(path: Path, *, expected_bands: int, ms_fixed_scale: float) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        import tifffile

        arr = tifffile.imread(str(path))
    elif suffix == ".npy":
        arr = np.load(str(path))
    elif suffix == ".npz":
        obj = np.load(str(path))
        # Heuristic: pick the first array entry.
        if len(obj.files) == 0:
            raise ValueError(f"Empty npz: {path}")
        arr = obj[obj.files[0]]
    else:
        raise ValueError(f"Unsupported MS format: {path} (expected tif/tiff/npy/npz)")

    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[None, ...]  # (1,H,W)
    elif arr.ndim == 3:
        # Accept (N,H,W) or (H,W,N).
        if arr.shape[0] <= 16 and arr.shape[1] > 16 and arr.shape[2] > 16:
            # (N,H,W)
            pass
        elif arr.shape[2] <= 16 and arr.shape[0] > 16 and arr.shape[1] > 16:
            # (H,W,N) -> (N,H,W)
            arr = arr.transpose(2, 0, 1)
        else:
            # Ambiguous: prefer treating the last dim as band.
            if arr.shape[2] <= 16:
                arr = arr.transpose(2, 0, 1)
            # else keep as-is (N,H,W)
    else:
        raise ValueError(f"MS array must be 2D/3D, got shape={arr.shape} from {path}")

    n, h, w = arr.shape
    if n < expected_bands:
        raise ValueError(f"MS bands < expected: got {n}, expected {expected_bands} ({path})")
    if n != expected_bands:
        arr = arr[:expected_bands]

    # Normalize to [0,1] (match oil config: fixed_scale=65535).
    arr = arr.astype(np.float32)
    if ms_fixed_scale > 0:
        arr = arr / float(ms_fixed_scale)
    arr = np.clip(arr, 0.0, 1.0)
    return arr  # (N,H,W) float32


def _safe_cosine(a: torch.Tensor, b: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    # a/b: (C,H,W) -> out: (H,W)
    dot = (a * b).sum(dim=0)
    na = torch.sqrt((a * a).sum(dim=0).clamp_min(eps))
    nb = torch.sqrt((b * b).sum(dim=0).clamp_min(eps))
    return dot / (na * nb).clamp_min(eps)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _normalize_for_vis(x: np.ndarray, *, lo: float | None = None, hi: float | None = None) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if lo is None:
        lo = float(np.nanpercentile(x, 1.0))
    if hi is None:
        hi = float(np.nanpercentile(x, 99.0))
    if hi <= lo:
        hi = lo + 1e-6
    x = (x - lo) / (hi - lo)
    return np.clip(x, 0.0, 1.0)


def _plot_grid(
    grid: np.ndarray,
    *,
    out_path: Path,
    title: str,
    ncols: int,
    cmap: str = "gray",
    vmin: float | None = None,
    vmax: float | None = None,
    labels: list[str] | None = None,
) -> None:
    import matplotlib.pyplot as plt

    assert grid.ndim == 3, f"Expected (N,H,W), got {grid.shape}"
    n, _, _ = grid.shape
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(n / float(ncols)))

    fig_w = 3.0 * ncols
    fig_h = 3.0 * nrows + 0.6
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
    fig.suptitle(title)
    for i in range(nrows * ncols):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        ax.axis("off")
        if i >= n:
            continue
        im = ax.imshow(grid[i], cmap=cmap, vmin=vmin, vmax=vmax)
        if labels is None:
            ax.set_title(f"#{i}")
        else:
            ax.set_title(str(labels[i]))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_grid_rgb(
    grid_rgb: np.ndarray,
    *,
    out_path: Path,
    title: str,
    ncols: int,
    labels: list[str] | None = None,
) -> None:
    import matplotlib.pyplot as plt

    assert grid_rgb.ndim == 4 and grid_rgb.shape[-1] == 3, f"Expected (N,H,W,3), got {grid_rgb.shape}"
    n = int(grid_rgb.shape[0])
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(n / float(ncols)))

    fig_w = 3.0 * ncols
    fig_h = 3.0 * nrows + 0.6
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
    fig.suptitle(title)
    for i in range(nrows * ncols):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        ax.axis("off")
        if i >= n:
            continue
        ax.imshow(np.clip(grid_rgb[i], 0.0, 1.0))
        if labels is None:
            ax.set_title(f"#{i}")
        else:
            ax.set_title(str(labels[i]))
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_before_after_pairs(
    before: np.ndarray,
    after: np.ndarray,
    *,
    out_path: Path,
    title: str,
    cmap: str = "viridis",
    row_labels: list[str] | None = None,
) -> None:
    """
    before/after: (N,H,W). For each band i, use a shared vmin/vmax computed from both before_i+after_i.
    """
    import matplotlib.pyplot as plt

    assert before.shape == after.shape and before.ndim == 3, f"Expected same (N,H,W), got {before.shape} vs {after.shape}"
    n = int(before.shape[0])
    if row_labels is not None and len(row_labels) != n:
        raise ValueError(f"row_labels length mismatch: got {len(row_labels)} expected {n}")
    fig, axes = plt.subplots(n, 2, figsize=(7.0, 2.4 * n), squeeze=False)
    fig.suptitle(title)
    for i in range(n):
        vals = np.concatenate([before[i].ravel(), after[i].ravel()], axis=0)
        vmin = float(np.nanpercentile(vals, 1.0))
        vmax = float(np.nanpercentile(vals, 99.0))
        if vmax <= vmin:
            vmax = vmin + 1e-6
        for j, (img, name) in enumerate([(before[i], "before"), (after[i], "after")]):
            ax = axes[i][j]
            ax.axis("off")
            im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
            prefix = f"band #{i}" if row_labels is None else str(row_labels[i])
            ax.set_title(f"{prefix} {name} (p1..p99)")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_flow_quiver(
    *,
    dx: np.ndarray,  # (H,W) in feature pixels
    dy: np.ndarray,  # (H,W) in feature pixels
    mag: np.ndarray | None,  # (H,W)
    out_path: Path,
    title: str,
    step: int = 8,
    mag_cmap: str = "viridis",
) -> None:
    """
    Visualize a dense displacement field with direction using arrows (quiver).
    Note: dx/dy are "sampling displacements" used by grid_sample (backward warping).
    """
    import matplotlib.pyplot as plt

    dx = np.asarray(dx, dtype=np.float32)
    dy = np.asarray(dy, dtype=np.float32)
    h, w = int(dx.shape[0]), int(dx.shape[1])
    if dy.shape != (h, w):
        raise ValueError(f"dx/dy shape mismatch: dx={dx.shape} dy={dy.shape}")

    if mag is None:
        mag = np.sqrt(dx * dx + dy * dy)
    else:
        mag = np.asarray(mag, dtype=np.float32)

    # Background scale.
    mag_vmax = float(np.nanpercentile(mag, 99.0))
    mag_vmax = max(mag_vmax, 1e-6)

    # Subsample for readability.
    step = max(1, int(step))
    ys, xs = np.mgrid[0:h:step, 0:w:step]
    u = dx[::step, ::step]
    v = dy[::step, ::step]

    # Make arrows visible: scale so that ~p95 magnitude becomes ~2 feature pixels.
    p95 = float(np.nanpercentile(np.sqrt(u * u + v * v), 95.0))
    scale = max(p95, 1e-6) / 2.0

    fig, ax = plt.subplots(1, 1, figsize=(6.2, 6.2))
    ax.axis("off")
    ax.imshow(mag, cmap=mag_cmap, vmin=0.0, vmax=mag_vmax)
    ax.quiver(
        xs,
        ys,
        u,
        v,
        color="white",
        angles="xy",
        scale_units="xy",
        scale=scale,
        width=0.003,
        headwidth=3.5,
        headlength=4.5,
        headaxislength=4.0,
        minlength=0.0,
    )
    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _flow_to_hsv(dx: np.ndarray, dy: np.ndarray, *, mag_clip: float | None = None) -> np.ndarray:
    """
    dx/dy: (H,W) in feature-pixel units.
    Return RGB image (H,W,3) in [0,1].
    """
    # Hue: direction, Value: magnitude.
    ang = np.arctan2(dy, dx)  # [-pi,pi]
    hue = (ang + np.pi) / (2.0 * np.pi)
    mag = np.sqrt(dx * dx + dy * dy)
    if mag_clip is None:
        mag_clip = float(np.nanpercentile(mag, 99.0))
    mag_clip = max(float(mag_clip), 1e-6)
    val = np.clip(mag / mag_clip, 0.0, 1.0)
    sat = np.ones_like(val)

    hsv = np.stack([hue, sat, val], axis=-1).astype(np.float32)
    # Manual HSV->RGB (avoid extra deps).
    h = hsv[..., 0] * 6.0
    c = hsv[..., 1] * hsv[..., 2]
    x = c * (1.0 - np.abs((h % 2.0) - 1.0))
    m = hsv[..., 2] - c

    z = np.zeros_like(h)
    r, g, b = z.copy(), z.copy(), z.copy()
    conds = [
        (0 <= h) & (h < 1),
        (1 <= h) & (h < 2),
        (2 <= h) & (h < 3),
        (3 <= h) & (h < 4),
        (4 <= h) & (h < 5),
        (5 <= h) & (h <= 6),
    ]
    vals = [
        (c, x, z),
        (x, c, z),
        (z, c, x),
        (z, x, c),
        (x, z, c),
        (c, z, x),
    ]
    for cond, (rr, gg, bb) in zip(conds, vals):
        r[cond] = rr[cond]
        g[cond] = gg[cond]
        b[cond] = bb[cond]
    rgb = np.stack([r + m, g + m, b + m], axis=-1)
    return np.clip(rgb, 0.0, 1.0)


def _save_paper_figures(
    *,
    out_dir: Path,
    ref_mean: np.ndarray,  # (H4,W4)
    ref_choice: np.ndarray | None,  # (H4,W4) in [0..6]
    ms_raw: np.ndarray | None,  # (7,H,W) in [0,1]
    z_before_mean: np.ndarray,  # (7,H4,W4)
    z_after_mean: np.ndarray,  # (7,H4,W4)
    cos_before: np.ndarray,  # (7,H4,W4)
    cos_after: np.ndarray,  # (7,H4,W4)
    cos_delta: np.ndarray,  # (7,H4,W4)
    flow_mag: np.ndarray,  # (7,H4,W4) feature-pixel units
    flow_rgb: np.ndarray,  # (7,H4,W4,3) HSV->RGB
) -> None:
    """
    Save paper-friendly composite figures:
    - paper_overview.{png,pdf}: ref + (before/after/delta + cos_delta) for all bands
    - paper_cosine_triplet.{png,pdf}: cos_before/cos_after/cos_delta
    - paper_flow_hsv.{png,pdf}: flow direction (HSV)
    - paper_grid_top3.{png,pdf}: raw+ref+before/after+cos+flow in one grid (good as a main-paper figure)
    - paper_grid_all.{png,pdf}: same grid for all bands (usually supplementary)

    Design goals:
    - put before/after in the same figure
    - include canonical reference for comparison (explicitly in overview)
    - use percentile-based contrast stretching to make small changes visible
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    _ensure_dir(out_dir)

    # Global scaling for cosine-delta (keep consistent across all bands).
    cd_abs_p99 = float(np.nanpercentile(np.abs(cos_delta), 99.0))
    cd_v = max(cd_abs_p99, 1e-6)

    def _imshow(ax, img, *, cmap, vmin=None, vmax=None, title=None, title_size: int = 10):
        ax.axis("off")
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        if title:
            ax.set_title(title, fontsize=title_size)
        return im

    def _save_grid(indices: list[int], *, stem: str, include_raw: bool = False) -> None:
        """
        Paper-friendly grid that puts ref/before/after/cos/flow side-by-side.

        Rows: selected bands.
        Cols: ref | before | after | Δfeat | cos(before,ref) | cos(after,ref) | cosΔ | |flow|

        Contrast policy:
          - ref/before/after share a per-row p1..p99 scale computed from all 3 maps
          - Δfeat uses symmetric p99(|Δ|)
          - cosine maps share a per-row p1..p99 scale computed from cos_before+cos_after
          - cosΔ uses symmetric global p99(|cosΔ|)
        """
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        col_titles = [
            "ref",
            "before",
            "after",
            "Δfeat",
            "cos(before, ref)",
            "cos(after, ref)",
            "cosΔ",
            "|flow| (yellow=large)",
        ]
        if include_raw:
            col_titles = ["raw"] + col_titles
        ncols = len(col_titles)
        nrows = len(indices)

        fig = plt.figure(figsize=(2.0 * ncols, 2.0 * max(nrows, 1)))
        gs = GridSpec(nrows=nrows, ncols=ncols, figure=fig, wspace=0.015, hspace=0.06)

        mag_vmax = float(np.nanpercentile(flow_mag[indices], 99.0))
        mag_vmax = max(mag_vmax, 1e-6)

        row_label_artists = []
        label_x_axes = -0.06
        label_fontsize = 11
        col_title_fontsize = label_fontsize
        for r, band_i in enumerate(indices):
            before = z_before_mean[band_i]
            after = z_after_mean[band_i]

            # Shared scale for ref/before/after per row.
            vals = np.concatenate([ref_mean.ravel(), before.ravel(), after.ravel()], axis=0)
            vmin = float(np.nanpercentile(vals, 1.0))
            vmax = float(np.nanpercentile(vals, 99.0))
            if vmax <= vmin:
                vmax = vmin + 1e-6

            # Δfeat scaling.
            delta = after - before
            dv = float(np.nanpercentile(np.abs(delta), 99.0))
            dv = max(dv, 1e-6)

            # Cosine scaling (dynamic range is usually narrow if plotted with fixed [-1,1]).
            cb = cos_before[band_i]
            ca = cos_after[band_i]
            cvals = np.concatenate([cb.ravel(), ca.ravel()], axis=0)
            cvmin = float(np.nanpercentile(cvals, 1.0))
            cvmax = float(np.nanpercentile(cvals, 99.0))
            if cvmax <= cvmin:
                cvmax = cvmin + 1e-6

            col_offset = 0
            label_anchor_ax = None
            if include_raw:
                ax = fig.add_subplot(gs[r, 0])
                raw_band = ms_raw[band_i] if ms_raw is not None else np.zeros_like(ref_mean)
                raw_norm = _normalize_for_vis(raw_band)
                _imshow(
                    ax,
                    raw_norm,
                    cmap="gray",
                    vmin=0.0,
                    vmax=1.0,
                    title=col_titles[0] if r == 0 else None,
                    title_size=col_title_fontsize,
                )
                label_anchor_ax = ax
                label_x_axes = -0.10
                col_offset = 1

            ax = fig.add_subplot(gs[r, 0 + col_offset])
            _imshow(
                ax,
                ref_mean,
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                title=col_titles[0 + col_offset] if r == 0 else None,
                title_size=col_title_fontsize,
            )
            if label_anchor_ax is None:
                label_anchor_ax = ax
            # Place band label just outside the first panel (axes coords) and keep it in the tight bbox.
            row_label_artists.append(
                label_anchor_ax.text(
                    label_x_axes,
                    0.5,
                    f"band {band_i}",
                    transform=label_anchor_ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=label_fontsize,
                    color="black",
                    clip_on=False,
                )
            )

            ax = fig.add_subplot(gs[r, 1 + col_offset])
            _imshow(
                ax,
                before,
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                title=col_titles[1 + col_offset] if r == 0 else None,
                title_size=col_title_fontsize,
            )
            ax = fig.add_subplot(gs[r, 2 + col_offset])
            _imshow(
                ax,
                after,
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                title=col_titles[2 + col_offset] if r == 0 else None,
                title_size=col_title_fontsize,
            )

            ax = fig.add_subplot(gs[r, 3 + col_offset])
            _imshow(
                ax,
                delta,
                cmap="coolwarm",
                vmin=-dv,
                vmax=dv,
                title=col_titles[3 + col_offset] if r == 0 else None,
                title_size=col_title_fontsize,
            )

            ax = fig.add_subplot(gs[r, 4 + col_offset])
            _imshow(
                ax,
                cb,
                cmap="coolwarm",
                vmin=cvmin,
                vmax=cvmax,
                title=col_titles[4 + col_offset] if r == 0 else None,
                title_size=col_title_fontsize,
            )
            ax = fig.add_subplot(gs[r, 5 + col_offset])
            _imshow(
                ax,
                ca,
                cmap="coolwarm",
                vmin=cvmin,
                vmax=cvmax,
                title=col_titles[5 + col_offset] if r == 0 else None,
                title_size=col_title_fontsize,
            )
            ax = fig.add_subplot(gs[r, 6 + col_offset])
            _imshow(
                ax,
                cos_delta[band_i],
                cmap="coolwarm",
                vmin=-cd_v,
                vmax=cd_v,
                title=col_titles[6 + col_offset] if r == 0 else None,
                title_size=col_title_fontsize,
            )

            ax = fig.add_subplot(gs[r, 7 + col_offset])
            _imshow(
                ax,
                flow_mag[band_i],
                cmap="viridis",
                vmin=0.0,
                vmax=mag_vmax,
                title=col_titles[7 + col_offset] if r == 0 else None,
                title_size=col_title_fontsize,
            )

        suptitle = fig.suptitle(
            "CRGGA: ref vs before/after (feature) + cosine + |flow| @ stride=4",
            fontsize=12,
            y=0.985,
        )
        fig.subplots_adjust(left=0.07, right=0.995, bottom=0.03, top=0.92)
        fig.savefig(
            out_dir / f"{stem}.png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.02,
            bbox_extra_artists=row_label_artists + [suptitle],
        )
        fig.savefig(
            out_dir / f"{stem}.pdf",
            bbox_inches="tight",
            pad_inches=0.02,
            bbox_extra_artists=row_label_artists + [suptitle],
        )
        plt.close(fig)

    def _save_overview(indices: list[int], *, stem: str) -> None:
        n = len(indices)
        fig = plt.figure(figsize=(14.0, 2.0 * (n + 1)))
        gs = GridSpec(nrows=n + 1, ncols=4, figure=fig, wspace=0.02, hspace=0.18)

        # Row 0: canonical reference and aggregated diagnostics.
        r0_vmin = float(np.nanpercentile(ref_mean, 1.0))
        r0_vmax = float(np.nanpercentile(ref_mean, 99.0))
        if r0_vmax <= r0_vmin:
            r0_vmax = r0_vmin + 1e-6
        _imshow(
            fig.add_subplot(gs[0, 0]),
            ref_mean,
            cmap="viridis",
            vmin=r0_vmin,
            vmax=r0_vmax,
            title="ref (mean over C)",
        )

        if ref_choice is not None:
            _imshow(fig.add_subplot(gs[0, 1]), ref_choice, cmap="tab10", vmin=0.0, vmax=6.0, title="ref band argmax")
        else:
            fig.add_subplot(gs[0, 1]).axis("off")

        mag_mean = flow_mag.mean(axis=0)
        mag_vmax = float(np.nanpercentile(mag_mean, 99.0))
        _imshow(
            fig.add_subplot(gs[0, 2]),
            mag_mean,
            cmap="viridis",
            vmin=0.0,
            vmax=max(mag_vmax, 1e-6),
            title="|flow| mean over bands",
        )

        cd_mean = cos_delta.mean(axis=0)
        _imshow(fig.add_subplot(gs[0, 3]), cd_mean, cmap="coolwarm", vmin=-cd_v, vmax=cd_v, title="cosΔ mean over bands")

        # Per-band rows.
        for row_i, band_i in enumerate(indices):
            before = z_before_mean[band_i]
            after = z_after_mean[band_i]
            # Shared scale for before/after per band.
            vals = np.concatenate([before.ravel(), after.ravel()], axis=0)
            vmin = float(np.nanpercentile(vals, 1.0))
            vmax = float(np.nanpercentile(vals, 99.0))
            if vmax <= vmin:
                vmax = vmin + 1e-6

            delta = after - before
            dv = float(np.nanpercentile(np.abs(delta), 99.0))
            dv = max(dv, 1e-6)

            _imshow(fig.add_subplot(gs[row_i + 1, 0]), before, cmap="viridis", vmin=vmin, vmax=vmax, title=f"band#{band_i} before")
            _imshow(fig.add_subplot(gs[row_i + 1, 1]), after, cmap="viridis", vmin=vmin, vmax=vmax, title=f"band#{band_i} after")
            _imshow(fig.add_subplot(gs[row_i + 1, 2]), delta, cmap="coolwarm", vmin=-dv, vmax=dv, title="Δ=after-before")
            _imshow(fig.add_subplot(gs[row_i + 1, 3]), cos_delta[band_i], cmap="coolwarm", vmin=-cd_v, vmax=cd_v, title="cosΔ to ref")

        fig.suptitle("CRGGA feature alignment @ stride=4 (paper overview)", fontsize=14)
        fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
        fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

    n_bands = int(z_before_mean.shape[0])
    all_idx = list(range(n_bands))
    _save_overview(all_idx, stem="paper_overview")

    # A compact version for the main paper: pick top-3 bands by mean cos_delta.
    per_band_gain = cos_delta.reshape(n_bands, -1).mean(axis=1)
    topk = min(3, n_bands)
    top_idx = [int(i) for i in np.argsort(-per_band_gain)[:topk]]
    _save_overview(top_idx, stem="paper_overview_top3")
    _save_grid(top_idx, stem="paper_grid_top3", include_raw=True)
    _save_grid(all_idx, stem="paper_grid_all", include_raw=False)

    # ---- Figure 2: cosine maps (before/after/delta) ----
    def _save_cos_triplet(indices: list[int], *, stem: str) -> None:
        n = len(indices)
        fig = plt.figure(figsize=(12.5, 2.0 * n))
        gs = GridSpec(nrows=n, ncols=3, figure=fig, wspace=0.02, hspace=0.18)
        for row_i, band_i in enumerate(indices):
            _imshow(fig.add_subplot(gs[row_i, 0]), cos_before[band_i], cmap="coolwarm", vmin=-1.0, vmax=1.0, title=f"band#{band_i} cos(before, ref)")
            _imshow(fig.add_subplot(gs[row_i, 1]), cos_after[band_i], cmap="coolwarm", vmin=-1.0, vmax=1.0, title=f"band#{band_i} cos(after, ref)")
            _imshow(fig.add_subplot(gs[row_i, 2]), cos_delta[band_i], cmap="coolwarm", vmin=-cd_v, vmax=cd_v, title="Δ")
        fig.suptitle("Cosine similarity to canonical reference (before/after/Δ)", fontsize=14)
        fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
        fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

    _save_cos_triplet(all_idx, stem="paper_cosine_triplet")
    _save_cos_triplet(top_idx, stem="paper_cosine_triplet_top3")

    # ---- Figure 3: flow direction (HSV) ----
    def _save_flow(indices: list[int], *, stem: str) -> None:
        n = len(indices)
        fig = plt.figure(figsize=(7.0, 2.0 * n))
        gs = GridSpec(nrows=n, ncols=1, figure=fig, wspace=0.02, hspace=0.18)
        for row_i, band_i in enumerate(indices):
            ax = fig.add_subplot(gs[row_i, 0])
            ax.axis("off")
            ax.imshow(flow_rgb[band_i])
            ax.set_title(f"band#{band_i} flow (HSV: hue=dir, value=|flow|)", fontsize=10)
        fig.suptitle("Expected flow under attention (feature-pixel units, stride=4)", fontsize=14)
        fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
        fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

    _save_flow(all_idx, stem="paper_flow_hsv")
    _save_flow(top_idx, stem="paper_flow_hsv_top3")


def _plot_rgb(rgb: np.ndarray, *, out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(6.0, 6.0))
    plt.imshow(np.clip(rgb, 0.0, 1.0))
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _upsample_map(x: np.ndarray, *, out_hw: tuple[int, int]) -> np.ndarray:
    """
    x: (H,W) float32/float64
    out_hw: (H_out, W_out)
    """
    x = np.asarray(x, dtype=np.float32)
    xt = torch.from_numpy(x)[None, None]  # 1,1,H,W
    yt = F.interpolate(xt, size=out_hw, mode="bilinear", align_corners=False)
    return yt[0, 0].cpu().numpy()


def _rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float32)
    g = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return np.stack([g, g, g], axis=-1)


def _normalize_01(x: np.ndarray, *, p_lo: float = 1.0, p_hi: float = 99.0, clip_neg: bool = False) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if clip_neg:
        x = np.clip(x, 0.0, None)
    lo = float(np.nanpercentile(x, p_lo))
    hi = float(np.nanpercentile(x, p_hi))
    if hi <= lo:
        hi = lo + 1e-6
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0)


def _save_rgb_overlays(
    *,
    out_dir: Path,
    rgb: np.ndarray,  # (H,W,3) in [0,1]
    gain_map_h4w4: np.ndarray,  # (H4,W4), higher => larger alignment gain
    mag_map_h4w4: np.ndarray,  # (H4,W4), higher => larger offsets (feature pixels)
) -> None:
    """
    Paper-friendly overlays:
      - alignment gain: keep original RGB color where gain is high; desaturate to gray where gain is low
      - offset magnitude: larger offsets become more "yellow"
    """
    import matplotlib.pyplot as plt

    _ensure_dir(out_dir)

    h, w = int(rgb.shape[0]), int(rgb.shape[1])
    # ---- Gain: only positive improvements matter. Use a high-percentile stretch to highlight "where it helps". ----
    gain_raw = np.asarray(gain_map_h4w4, dtype=np.float32)
    gain_raw = np.clip(gain_raw, 0.0, None)
    g_lo = float(np.nanpercentile(gain_raw, 80.0))
    g_hi = float(np.nanpercentile(gain_raw, 99.0))
    if g_hi <= g_lo:
        g_hi = g_lo + 1e-6
    gain = np.clip((gain_raw - g_lo) / (g_hi - g_lo), 0.0, 1.0)

    # ---- Offset magnitude: highlight only large offsets (top tail) to avoid washing the whole image yellow. ----
    mag_raw = np.asarray(mag_map_h4w4, dtype=np.float32)
    mag_raw = np.clip(mag_raw, 0.0, None)
    m_lo = float(np.nanpercentile(mag_raw, 80.0))
    m_hi = float(np.nanpercentile(mag_raw, 99.0))
    if m_hi <= m_lo:
        m_hi = m_lo + 1e-6
    mag = np.clip((mag_raw - m_lo) / (m_hi - m_lo), 0.0, 1.0)

    gain_up = _upsample_map(gain, out_hw=(h, w))
    mag_up = _upsample_map(mag, out_hw=(h, w))

    # Gamma: suppress small values further, keep only confident regions.
    gain_a = np.clip(gain_up, 0.0, 1.0) ** 0.5
    mag_a = np.clip(mag_up, 0.0, 1.0) ** 1.6

    gray = _rgb_to_gray(rgb)
    keep_color = gray * (1.0 - gain_a[..., None]) + rgb * gain_a[..., None]

    yellow = np.array([1.0, 1.0, 0.0], dtype=np.float32)
    alpha = 0.85 * mag_a[..., None]  # cap to keep content visible
    offset_yellow = rgb * (1.0 - alpha) + yellow[None, None, :] * alpha

    combo_base = keep_color
    combo = combo_base * (1.0 - alpha) + yellow[None, None, :] * alpha

    # Compact horizontal layout for paper figure.
    fig, axes = plt.subplots(1, 4, figsize=(12.0, 3.2), squeeze=False)
    panels = [
        ("RGB (original)", rgb),
        ("Gain: keep color when high", keep_color),
        ("Offset magnitude: yellow when large", offset_yellow),
        ("Combined", combo),
    ]
    for ax, (t, img) in zip(axes[0], panels):
        ax.axis("off")
        ax.imshow(np.clip(img, 0.0, 1.0))
        ax.set_title(t, fontsize=9, pad=2)
    fig.suptitle("CRGGA overlays (gain & offset magnitude)", fontsize=11, y=0.98)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.90, wspace=0.02)
    fig.savefig(out_dir / "paper_rgb_overlays.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_dir / "paper_rgb_overlays.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _default_checkpoint() -> Path:
    # Keep a stable default for the "final" postblock + msbandsep config (oil).
    return Path(
        "outputs/oil_rgb_msi_20260115/rtmsfdetr/"
        "rtv4_hgnetv2_m_distill_det_dualstream_c2former_postblock_add_wbadd_c3c4c5_msbandsep_c2align_infonce_reg/"
        "260122-000531-rtmsfdetr_oil_rgb_msi_20260115_det_rtv4_hgnetv2_m_distill_dualstream_"
        "c2former_postblock_add_wbadd_c3c4c5_msbandsep_c2align_infonce_reg(915-780)/checkpoint_best.pth"
    )


def _build_msbandsep_from_final_cfg() -> MSBandSeparatedStemAlign:
    # Mirror the "final" YAML config defaults (oil, ms_fixed_scale=65535).
    align_cfg = {
        "enabled": True,
        "ref_mode": "spatial_weighted",
        "ref_detach": True,
        "num_iters": 1,
        "num_keypoints": 9,
        "offset_enabled": True,
        "offset_scale": 3.0,
        "attention_norm": "softmax",
        "padding_mode": "border",
        "align_corners": True,
        "loss_type": "infonce",
        "loss_downsample": 0.5,
        "nce_patch_size": 5,
        "nce_num_patches": 64,
        "nce_tau": 0.2,
        "loss_weight": 0.02,
        "loss_offset_weight": 0.01,
        "loss_attn_entropy_weight": 0.001,
    }
    return MSBandSeparatedStemAlign(
        ms_in_chs=7,
        c2_in_channels=32,  # HGNetv2-B2 stem out channels
        embed_channels=16,
        embed_use_bn=True,
        align_cfg=align_cfg,
    )


def _load_msbandsep_weights(stem: MSBandSeparatedStemAlign, ckpt_path: Path) -> None:
    obj = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = obj.get("model", obj.get("state_dict", obj))
    if not isinstance(state, dict):
        raise TypeError(f"Unexpected checkpoint format: {type(state)}")

    prefix = "model.backbone.ms_band_sep_stem."
    filtered = {}
    for k, v in state.items():
        kk = str(k)
        if kk.startswith("module."):
            kk = kk[len("module.") :]
        if kk.startswith(prefix):
            filtered[kk[len(prefix) :]] = v
    missing, unexpected = stem.load_state_dict(filtered, strict=False)
    if missing:
        raise RuntimeError(f"Missing keys when loading ms_band_sep_stem: {missing[:20]} (total={len(missing)})")
    if unexpected:
        raise RuntimeError(
            f"Unexpected keys when loading ms_band_sep_stem: {unexpected[:20]} (total={len(unexpected)})"
        )


def _compute_ref_weights(stem: MSBandSeparatedStemAlign, z: torch.Tensor) -> torch.Tensor | None:
    # z: (B,N,C,H4,W4)
    aligner = stem.aligner
    if aligner is None:
        return None
    if getattr(aligner, "ref_mode", None) != "spatial_weighted":
        return None
    ref_conv = getattr(aligner, "ref_conv", None)
    if ref_conv is None:
        return None
    b, n, c, h, w = z.shape
    scores = ref_conv(z.reshape(b * n, c, h, w)).reshape(b, n, h, w)
    return torch.softmax(scores, dim=1)  # (B,N,H,W)


def _run(
    *,
    rgb_path: Path | None,
    ms_path: Path,
    ckpt_path: Path,
    out_dir: Path,
    device: str,
    ms_fixed_scale: float,
) -> None:
    _ensure_dir(out_dir)

    rgb = None
    if rgb_path is not None:
        rgb = _load_rgb(rgb_path)
        _plot_rgb(rgb, out_path=out_dir / "rgb.png", title=f"RGB: {rgb_path.name}")

    ms = _load_ms(ms_path, expected_bands=7, ms_fixed_scale=ms_fixed_scale)  # (7,H,W) float32

    # Save raw MS quicklook.
    _plot_grid(
        np.stack([_normalize_for_vis(ms[i]) for i in range(ms.shape[0])], axis=0),
        out_path=out_dir / "ms_raw_bands.png",
        title=f"MS raw bands (normalized for display): {ms_path.name}",
        ncols=7,
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
    )

    stem = _build_msbandsep_from_final_cfg()
    _load_msbandsep_weights(stem, ckpt_path=ckpt_path)
    stem.eval()
    stem.to(device)

    ms_t = torch.from_numpy(ms).unsqueeze(0).to(device=device, dtype=torch.float32)  # (1,7,H,W)

    with torch.no_grad():
        z = stem.embed(ms_t)  # (1,7,16,H4,W4)
        assert stem.aligner is not None, "aligner is disabled in MSBandSeparatedStemAlign"
        z_aligned, _, debug = stem.aligner(z, return_debug=True)  # type: ignore[misc]

    # Feature quicklooks (channel-mean per band).
    z_before_mean = z.mean(dim=2)[0].detach().cpu().numpy()  # (7,H4,W4)
    z_after_mean = z_aligned.mean(dim=2)[0].detach().cpu().numpy()  # (7,H4,W4)

    _plot_grid(
        np.stack([_normalize_for_vis(z_before_mean[i]) for i in range(7)], axis=0),
        out_path=out_dir / "embed_before_mean.png",
        title="Per-band embedded feature mean (before CRGGA) @ stride=4",
        ncols=7,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    _plot_grid(
        np.stack([_normalize_for_vis(z_after_mean[i]) for i in range(7)], axis=0),
        out_path=out_dir / "embed_after_mean.png",
        title="Per-band embedded feature mean (after CRGGA) @ stride=4",
        ncols=7,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    _plot_before_after_pairs(
        z_before_mean,
        z_after_mean,
        out_path=out_dir / "embed_mean_before_after_shared_scale.png",
        title="Embedded feature mean before/after (shared scale per band, p1..p99)",
        cmap="viridis",
    )
    diff = z_after_mean - z_before_mean
    vmax = float(np.nanpercentile(np.abs(diff), 99.0))
    vmax = max(vmax, 1e-6)
    _plot_grid(
        diff,
        out_path=out_dir / "embed_mean_delta.png",
        title="Embedded feature mean delta: after - before (raw values)",
        ncols=7,
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
    )

    # Cosine similarity to canonical reference (before/after + delta).
    ref = debug["ref"][0]  # (C,H4,W4)
    ref_mean = ref.mean(dim=0).detach().cpu().numpy()  # (H4,W4)
    cos_before = []
    cos_after = []
    for i in range(7):
        cos_before.append(_safe_cosine(z[0, i], ref).detach().cpu().numpy())
        cos_after.append(_safe_cosine(z_aligned[0, i], ref).detach().cpu().numpy())
    cos_before = np.stack(cos_before, axis=0)
    cos_after = np.stack(cos_after, axis=0)
    cos_delta = cos_after - cos_before

    _plot_grid(
        cos_before,
        out_path=out_dir / "cos_to_ref_before.png",
        title="Cosine(x_band, ref) before alignment (raw values)",
        ncols=7,
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
    )
    _plot_grid(
        cos_after,
        out_path=out_dir / "cos_to_ref_after.png",
        title="Cosine(x_band, ref) after alignment (raw values)",
        ncols=7,
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
    )
    _plot_grid(
        cos_delta,
        out_path=out_dir / "cos_to_ref_delta.png",
        title="Cosine improvement: after - before",
        ncols=7,
        cmap="coolwarm",
        vmin=float(np.nanpercentile(cos_delta, 1.0)),
        vmax=float(np.nanpercentile(cos_delta, 99.0)),
    )

    # Offsets: compute attention-weighted expected flow in feature-pixel units.
    offset_x = debug["offset_x"][0].detach().cpu().numpy()  # (7,K,H4,W4), normalized grid delta
    offset_y = debug["offset_y"][0].detach().cpu().numpy()  # (7,K,H4,W4)
    attn = debug["attn"][0].detach().cpu().numpy()  # (7,K,H4,W4)

    k = int(attn.shape[1])
    h4, w4 = int(attn.shape[2]), int(attn.shape[3])
    denom_x = max(w4 - 1, 1) / 2.0
    denom_y = max(h4 - 1, 1) / 2.0

    # Normalize attention across K (softmax already sums to 1, but keep safe for sigmoid mode).
    attn_sum = np.clip(attn.sum(axis=1, keepdims=True), 1e-8, None)
    p = attn / attn_sum

    offset_x_px = offset_x * float(denom_x)
    offset_y_px = offset_y * float(denom_y)
    dx = (p * offset_x_px).sum(axis=1)  # (7,H4,W4) in feature pixels
    dy = (p * offset_y_px).sum(axis=1)  # (7,H4,W4)
    mag = np.sqrt(dx * dx + dy * dy)

    _plot_grid(
        mag,
        out_path=out_dir / "flow_mag_feature_px.png",
        title="Expected offset magnitude (feature-pixel units, stride=4)",
        ncols=7,
        cmap="viridis",
        vmin=0.0,
        vmax=float(np.nanpercentile(mag, 99.0)),
    )
    # Flow direction visualization (HSV: hue=direction, value=magnitude).
    mag_clip = float(np.nanpercentile(mag, 99.0))
    flow_rgb = np.stack([_flow_to_hsv(dx[i], dy[i], mag_clip=mag_clip) for i in range(7)], axis=0)
    _plot_grid_rgb(
        flow_rgb,
        out_path=out_dir / "flow_hsv.png",
        title="Flow direction (HSV): hue=direction, value=|flow| (feature-pixel units)",
        ncols=7,
    )
    # Quiver (vector field) summary: mean expected sampling displacement across bands.
    dx_mean = dx.mean(axis=0)
    dy_mean = dy.mean(axis=0)
    mag_mean = mag.mean(axis=0)
    _plot_flow_quiver(
        dx=dx_mean,
        dy=dy_mean,
        mag=mag_mean,
        out_path=out_dir / "paper_flow_quiver_mean.png",
        title="Mean expected sampling displacement (quiver) @ stride=4",
        step=8,
        mag_cmap="viridis",
    )
    _plot_flow_quiver(
        dx=dx_mean,
        dy=dy_mean,
        mag=mag_mean,
        out_path=out_dir / "paper_flow_quiver_mean.pdf",
        title="Mean expected sampling displacement (quiver) @ stride=4",
        step=8,
        mag_cmap="viridis",
    )

    # Attention entropy (after normalization).
    ent = -(p * np.log(np.clip(p, 1e-8, 1.0))).sum(axis=1)  # (7,H4,W4)
    _plot_grid(
        ent,
        out_path=out_dir / "attn_entropy.png",
        title="Attention entropy over K (lower => more selective)",
        ncols=7,
        cmap="viridis",
        vmin=float(np.nanpercentile(ent, 1.0)),
        vmax=float(np.nanpercentile(ent, 99.0)),
    )

    # Reference band selection (only for spatial_weighted).
    w_ref = _compute_ref_weights(stem, z)
    if w_ref is not None:
        w_ref_np = w_ref[0].detach().cpu().numpy()  # (7,H4,W4)
        _plot_grid(
            w_ref_np,
            out_path=out_dir / "ref_weights.png",
            title="Spatial reference weights w_n(h,w) (softmax over bands)",
            ncols=7,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
        )
        arg = w_ref_np.argmax(axis=0).astype(np.float32)  # (H4,W4)
        _plot_grid(
            arg[None, ...],
            out_path=out_dir / "ref_choice_argmax.png",
            title="Reference band choice argmax_n w_n(h,w) (0..6)",
            ncols=1,
            cmap="tab10",
            vmin=0.0,
            vmax=6.0,
        )
        ref_choice = arg
    else:
        ref_choice = None

    # Save raw arrays for later analysis.
    np.savez_compressed(
        out_dir / "crgga_debug_arrays.npz",
        ms=ms,
        z_before_mean=z_before_mean,
        z_after_mean=z_after_mean,
        cos_before=cos_before,
        cos_after=cos_after,
        cos_delta=cos_delta,
        dx_feature_px=dx,
        dy_feature_px=dy,
        mag_feature_px=mag,
        attn_entropy=ent,
    )

    # Save a quick numeric summary for debugging.
    def _fmt(x: float) -> str:
        return f"{x:.6f}"

    cos_before_mean = float(cos_before.mean())
    cos_after_mean = float(cos_after.mean())
    cos_delta_mean = float(cos_delta.mean())
    mag_mean = float(mag.mean())
    mag_p95 = float(np.nanpercentile(mag, 95.0))
    mag_p99 = float(np.nanpercentile(mag, 99.0))
    mag_max = float(mag.max())

    with (out_dir / "stats.txt").open("w", encoding="utf-8") as f:
        f.write("CRGGA visualization stats\n")
        f.write(f"ms_path: {ms_path}\n")
        f.write(f"checkpoint: {ckpt_path}\n")
        f.write(f"device: {device}\n")
        f.write("\n")
        h4, w4 = int(z.shape[-2]), int(z.shape[-1])
        stride_y = float(ms.shape[1]) / float(h4)
        stride_x = float(ms.shape[2]) / float(w4)
        offset_scale = float(getattr(stem.aligner.aligner, "offset_scale", 0.0))  # type: ignore[union-attr]
        f.write(f"feature_grid: H4={h4} W4={w4}\n")
        f.write(f"feature_stride: sy={stride_y:.3f} sx={stride_x:.3f}\n")
        f.write(f"offset_scale: +/-{offset_scale:.3f} feature_px  (theoretical bound per keypoint)\n")
        f.write(f"offset_scale: +/-{offset_scale * stride_x:.3f} input_px (x)  +/-{offset_scale * stride_y:.3f} input_px (y)\n")
        f.write("\n")
        f.write(
            "cos_to_ref mean: "
            f"before={_fmt(cos_before_mean)} "
            f"after={_fmt(cos_after_mean)} "
            f"delta={_fmt(cos_delta_mean)}\n"
        )
        f.write(
            "flow_mag_feature_px: "
            f"mean={_fmt(mag_mean)} "
            f"p95={_fmt(mag_p95)} "
            f"p99={_fmt(mag_p99)} "
            f"max={_fmt(mag_max)}\n"
        )
        f.write(
            "flow_mag_input_px (approx stride*x): "
            f"mean={_fmt(mag_mean * stride_x)} "
            f"p95={_fmt(mag_p95 * stride_x)} "
            f"p99={_fmt(mag_p99 * stride_x)} "
            f"max={_fmt(mag_max * stride_x)}\n"
        )
        f.write("\n")
        f.write("per-band mean flow_mag_feature_px:\n")
        for i, v in enumerate(mag.reshape(7, -1).mean(axis=1)):
            f.write(f"  band#{i}: {_fmt(float(v))}\n")
        f.write("per-band mean cos_delta:\n")
        for i, v in enumerate(cos_delta.reshape(7, -1).mean(axis=1)):
            f.write(f"  band#{i}: {_fmt(float(v))}\n")

    # ---- Keypoint-level (K=9) sampling diagnostics for the "most improved" band ----
    per_band_gain = cos_delta.reshape(7, -1).mean(axis=1)
    band_kp = int(np.argsort(-per_band_gain)[0])
    k = int(attn.shape[1])
    kp_labels = [f"k{k_i}" for k_i in range(k)]

    # Keypoint offsets (direction+mag) in HSV.
    mag_clip = float(getattr(stem.aligner.aligner, "offset_scale", 1.0))  # type: ignore[union-attr]
    flow_k_rgb = np.stack([_flow_to_hsv(offset_x_px[band_kp, kk], offset_y_px[band_kp, kk], mag_clip=mag_clip) for kk in range(k)], axis=0)
    _plot_grid_rgb(
        flow_k_rgb,
        out_path=out_dir / f"paper_keypoints_offset_hsv_band{band_kp}.png",
        title=f"Band#{band_kp}: per-keypoint sampling offsets (HSV) @ stride=4 (mag_clip={mag_clip:.2f} feat_px)",
        ncols=3,
        labels=kp_labels,
    )
    _plot_grid_rgb(
        flow_k_rgb,
        out_path=out_dir / f"paper_keypoints_offset_hsv_band{band_kp}.pdf",
        title=f"Band#{band_kp}: per-keypoint sampling offsets (HSV) @ stride=4 (mag_clip={mag_clip:.2f} feat_px)",
        ncols=3,
        labels=kp_labels,
    )

    # Keypoint attention maps p_k(h,w).
    attn_k = p[band_kp]  # (K,H4,W4)
    attn_vmax = float(np.nanpercentile(attn_k, 99.0))
    _plot_grid(
        attn_k,
        out_path=out_dir / f"paper_keypoints_attn_band{band_kp}.png",
        title=f"Band#{band_kp}: attention over keypoints p_k(h,w) (softmax)",
        ncols=3,
        cmap="viridis",
        vmin=0.0,
        vmax=max(attn_vmax, 1e-6),
        labels=kp_labels,
    )
    _plot_grid(
        attn_k,
        out_path=out_dir / f"paper_keypoints_attn_band{band_kp}.pdf",
        title=f"Band#{band_kp}: attention over keypoints p_k(h,w) (softmax)",
        ncols=3,
        cmap="viridis",
        vmin=0.0,
        vmax=max(attn_vmax, 1e-6),
        labels=kp_labels,
    )

    # Keypoint sampled feature maps (channel-mean): what each keypoint reads before weighting.
    with torch.no_grad():
        src = z[:, band_kp]  # (1,C,H4,W4)
        offx_t = debug["offset_x"][:, band_kp].to(device=device)  # (1,K,H4,W4)
        offy_t = debug["offset_y"][:, band_kp].to(device=device)
        attn_t = debug["attn"][:, band_kp].to(device=device)
        fused_t, sampled_t, _ = stem.aligner.aligner.deform_with_attention(  # type: ignore[union-attr]
            src,
            offset_x=offx_t,
            offset_y=offy_t,
            attention_weights=attn_t,
            affine_theta=None,
        )
        sampled_mean_k = sampled_t.mean(dim=2)[0].detach().cpu().numpy()  # (K,H4,W4)
        before_mean_1 = src.mean(dim=1)[0].detach().cpu().numpy()
        fused_mean_1 = fused_t.mean(dim=1)[0].detach().cpu().numpy()

    # Use a shared scale across K sampled maps to make differences comparable.
    vals = np.concatenate([sampled_mean_k.ravel(), before_mean_1.ravel(), fused_mean_1.ravel()], axis=0)
    svmin = float(np.nanpercentile(vals, 1.0))
    svmax = float(np.nanpercentile(vals, 99.0))
    if svmax <= svmin:
        svmax = svmin + 1e-6
    _plot_grid(
        sampled_mean_k,
        out_path=out_dir / f"paper_keypoints_sampled_mean_band{band_kp}.png",
        title=f"Band#{band_kp}: sampled features per keypoint (mean over C) @ stride=4",
        ncols=3,
        cmap="viridis",
        vmin=svmin,
        vmax=svmax,
        labels=kp_labels,
    )
    _plot_grid(
        sampled_mean_k,
        out_path=out_dir / f"paper_keypoints_sampled_mean_band{band_kp}.pdf",
        title=f"Band#{band_kp}: sampled features per keypoint (mean over C) @ stride=4",
        ncols=3,
        cmap="viridis",
        vmin=svmin,
        vmax=svmax,
        labels=kp_labels,
    )
    _plot_before_after_pairs(
        before_mean_1[None, ...],
        fused_mean_1[None, ...],
        out_path=out_dir / f"paper_band_before_fused_mean_band{band_kp}.png",
        title=f"Band#{band_kp}: before vs fused-aligned (mean over C)",
        cmap="viridis",
        row_labels=["meanC"],
    )
    _plot_before_after_pairs(
        before_mean_1[None, ...],
        fused_mean_1[None, ...],
        out_path=out_dir / f"paper_band_before_fused_mean_band{band_kp}.pdf",
        title=f"Band#{band_kp}: before vs fused-aligned (mean over C)",
        cmap="viridis",
        row_labels=["meanC"],
    )
    d = fused_mean_1 - before_mean_1
    dv = float(np.nanpercentile(np.abs(d), 99.0))
    dv = max(dv, 1e-6)
    _plot_grid(
        d[None, ...],
        out_path=out_dir / f"paper_band_delta_mean_band{band_kp}.png",
        title=f"Band#{band_kp}: Δ=fused-before (mean over C)",
        ncols=1,
        cmap="coolwarm",
        vmin=-dv,
        vmax=dv,
        labels=["Δ"],
    )
    _plot_grid(
        d[None, ...],
        out_path=out_dir / f"paper_band_delta_mean_band{band_kp}.pdf",
        title=f"Band#{band_kp}: Δ=fused-before (mean over C)",
        ncols=1,
        cmap="coolwarm",
        vmin=-dv,
        vmax=dv,
        labels=["Δ"],
    )

    # Channel-level view: channels share the same sampling grid, but responses differ by channel.
    with torch.no_grad():
        delta_c = (z_aligned[:, band_kp] - z[:, band_kp]).abs().mean(dim=(0, 2, 3))  # (C,)
        topc = int(min(4, delta_c.numel()))
        ch_idx = [int(i) for i in torch.topk(delta_c, k=topc).indices.detach().cpu().tolist()]
        before_ch = z[0, band_kp][ch_idx].detach().cpu().numpy()
        after_ch = z_aligned[0, band_kp][ch_idx].detach().cpu().numpy()
    _plot_before_after_pairs(
        before_ch,
        after_ch,
        out_path=out_dir / f"paper_channels_before_after_band{band_kp}_top{topc}.png",
        title=f"Band#{band_kp}: top-{topc} channels by mean |Δ| (channels share same sampling grid)",
        cmap="viridis",
        row_labels=[f"ch#{i}" for i in ch_idx],
    )
    _plot_before_after_pairs(
        before_ch,
        after_ch,
        out_path=out_dir / f"paper_channels_before_after_band{band_kp}_top{topc}.pdf",
        title=f"Band#{band_kp}: top-{topc} channels by mean |Δ| (channels share same sampling grid)",
        cmap="viridis",
        row_labels=[f"ch#{i}" for i in ch_idx],
    )

    # Paper-friendly composites (png+pdf).
    _save_paper_figures(
        out_dir=out_dir,
        ref_mean=ref_mean,
        ref_choice=ref_choice,
        ms_raw=ms,
        z_before_mean=z_before_mean,
        z_after_mean=z_after_mean,
        cos_before=cos_before,
        cos_after=cos_after,
        cos_delta=cos_delta,
        flow_mag=mag,
        flow_rgb=flow_rgb,
    )

    # Additional paper-style overlays on the original RGB for easier interpretation.
    if rgb is not None:
        # If RGB/MS sizes mismatch, resize RGB to MS size to keep overlays spatially consistent.
        ms_h, ms_w = int(ms.shape[1]), int(ms.shape[2])
        rgb_vis = rgb
        if int(rgb.shape[0]) != ms_h or int(rgb.shape[1]) != ms_w:
            from PIL import Image

            rgb_u8 = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
            rgb_vis = np.asarray(Image.fromarray(rgb_u8).resize((ms_w, ms_h), resample=Image.BILINEAR)).astype(np.float32) / 255.0

        gain_mean = cos_delta.mean(axis=0)  # (H4,W4)
        mag_mean = mag.mean(axis=0)  # (H4,W4)
        _save_rgb_overlays(out_dir=out_dir, rgb=rgb_vis, gain_map_h4w4=gain_mean, mag_map_h4w4=mag_mean)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--rgb", type=str, default=None, help="RGB image path (optional)")
    p.add_argument("--ms", type=str, required=True, help="MS image path (tif/tiff/npy/npz)")
    p.add_argument("--checkpoint", type=str, default=str(_default_checkpoint()), help="checkpoint_best.pth")
    p.add_argument("--outdir", type=str, default=None, help="Output directory (default: outputs/vis_crgga/<ms_stem>)")
    p.add_argument("--device", type=str, default="auto", help="auto/cpu/cuda")
    p.add_argument("--ms_fixed_scale", type=float, default=65535.0, help="MS fixed scale for normalization")
    return p.parse_args()


def _resolve_device(req: str) -> str:
    req = str(req).strip().lower()
    if req == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return req


def main() -> None:
    args = _parse_args()
    rgb_path = None if args.rgb is None else Path(args.rgb).expanduser()
    ms_path = Path(args.ms).expanduser()
    ckpt_path = Path(args.checkpoint).expanduser()
    if args.outdir is None:
        tag = ms_path.stem
        # Avoid overwriting by default: append current YYYYMMDD-HHMM suffix.
        ts = datetime.now().strftime("%Y%m%d-%H%M")
        out_dir = Path("outputs") / "vis_crgga" / f"{tag}_{ts}"
    else:
        out_dir = Path(args.outdir).expanduser()

    if rgb_path is not None and not rgb_path.is_file():
        raise FileNotFoundError(rgb_path)
    if not ms_path.is_file():
        raise FileNotFoundError(ms_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)

    device = _resolve_device(args.device)
    _run(
        rgb_path=rgb_path,
        ms_path=ms_path,
        ckpt_path=ckpt_path,
        out_dir=out_dir,
        device=device,
        ms_fixed_scale=float(args.ms_fixed_scale),
    )
    print(f"[OK] Saved CRGGA visualizations to: {out_dir}")


if __name__ == "__main__":
    main()


"""
  python /home/ubuntu/Documents/newdisk_22T/zcc/msifp-detr/vis/vis_crgga_on_pair.py \
    --rgb tmp/crgga_vis/MAX_20240612_MAX_0201_Color_D_slice_0_0.jpg \
    --ms  tmp/crgga_vis/MAX_20240612_MAX_0201_Color_D_slice_0_0.tif \
    --checkpoint "outputs/oil_rgb_msi_20260202_3cls/rtmsfdetr/rtv4_hgnetv2_m_distill_det_dualstream_c2former_postblock_add_wbadd_c3c4c5_msbandsep_c2align_infonce_reg_globalkv_pos2d/baseline5/260203-230950-rtmsfdetr_oil_rgb_msi_20260202_det_rtv4_hgnetv2_m_distill_dualstream_c2former_postblock_add_wbadd_c3c4c5_msbandsep_c2align_infonce_reg_globalkv_pos2d/checkpoint_best.pth" \
    --device cpu
"""
