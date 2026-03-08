#!/usr/bin/env python3
import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import OmegaConf

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from engines.core.parse_config import get_config, load_config
from engines.trainer.base_trainer import _filter_compatible_state_dict, _remap_ultralytics_state_dict


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "module"):
        model = model.module
    if hasattr(model, "model"):
        model = model.model
    return model


def _find_backbone(model: torch.nn.Module) -> torch.nn.Module:
    model = _unwrap_model(model)
    if hasattr(model, "backbone"):
        return model.backbone
    return model


def _unwrap_dataset(ds):
    while hasattr(ds, "dataset"):
        try:
            ds = ds.dataset
        except Exception:
            break
    return ds


def _safe_minmax(x: np.ndarray) -> np.ndarray:
    vmin = float(np.nanmin(x))
    vmax = float(np.nanmax(x))
    if not math.isfinite(vmin) or not math.isfinite(vmax) or (vmax - vmin) < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - vmin) / (vmax - vmin)).astype(np.float32)


def _normalize_metrics(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean()
    std = x.std()
    return (x - mean) / (std + 1e-6)


def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    denom = (a_flat.norm() * b_flat.norm()).clamp_min(1e-8)
    return float((a_flat * b_flat).sum() / denom)


def _normalize_attention(attn: torch.Tensor, *, mode: str) -> torch.Tensor:
    if mode == "softmax":
        return attn
    denom = attn.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return attn / denom


def _create_base_grid(h: int, w: int, *, device: torch.device, dtype: torch.dtype, batch: int) -> torch.Tensor:
    ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack((grid_x, grid_y), dim=-1)  # (H,W,2)
    return grid.unsqueeze(0).expand(batch, -1, -1, -1)


def _warp_single_channel(
    x: torch.Tensor,
    *,
    offset_x: torch.Tensor,
    offset_y: torch.Tensor,
    attn: torch.Tensor,
    align_corners: bool,
    padding_mode: str,
    attention_norm: str,
) -> torch.Tensor:
    if x.ndim != 4 or x.shape[1] != 1:
        raise ValueError(f"Expected x shape (B,1,H,W), got {x.shape}")
    b, _, h, w = x.shape
    k = int(attn.shape[1])
    base = _create_base_grid(h, w, device=x.device, dtype=x.dtype, batch=b)
    grid_x = base[..., 0].unsqueeze(1).expand(b, k, h, w) + offset_x.to(dtype=x.dtype)
    grid_y = base[..., 1].unsqueeze(1).expand(b, k, h, w) + offset_y.to(dtype=x.dtype)
    sampling_grid = torch.stack((grid_x, grid_y), dim=-1).clamp(-1, 1)

    x_expanded = x.unsqueeze(1).expand(b, k, 1, h, w).reshape(b * k, 1, h, w)
    sampling_grid = sampling_grid.reshape(b * k, h, w, 2)
    sampled = F.grid_sample(
        x_expanded,
        sampling_grid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=align_corners,
    ).reshape(b, k, 1, h, w)

    attn_norm = _normalize_attention(attn, mode=attention_norm)
    fused = (sampled * attn_norm.unsqueeze(2)).sum(dim=1)
    return fused


def _compute_offset_mag(
    offset_x: torch.Tensor,
    offset_y: torch.Tensor,
    attn: torch.Tensor,
    *,
    attention_norm: str,
) -> torch.Tensor:
    # offset_x/y: (B,K,H,W)
    b, k, h, w = offset_x.shape
    denom_x = max(int(w) - 1, 1) / 2.0
    denom_y = max(int(h) - 1, 1) / 2.0
    offset_x_px = offset_x * float(denom_x)
    offset_y_px = offset_y * float(denom_y)
    attn_norm = _normalize_attention(attn, mode=attention_norm)
    fused_x = (attn_norm * offset_x_px).sum(dim=1)
    fused_y = (attn_norm * offset_y_px).sum(dim=1)
    mag = torch.sqrt(fused_x ** 2 + fused_y ** 2 + 1e-8)
    return mag


def _compute_attn_entropy(attn: torch.Tensor, *, attention_norm: str) -> torch.Tensor:
    attn_norm = _normalize_attention(attn, mode=attention_norm).clamp_min(1e-8)
    ent = -(attn_norm * torch.log(attn_norm)).sum(dim=1)
    return ent


def _sample_indices(length: int, *, num: int, seed: int, indices: Iterable[int] | None) -> List[int]:
    if indices:
        return [int(i) for i in indices if 0 <= int(i) < length]
    rng = random.Random(seed)
    if num >= length:
        return list(range(length))
    return rng.sample(range(length), num)


def _get_sample_meta(dataset, idx: int) -> Tuple[int | None, str | None]:
    ds = _unwrap_dataset(dataset)
    image_id = None
    file_name = None
    try:
        if hasattr(ds, "ids"):
            image_id = int(ds.ids[idx])
        if hasattr(ds, "coco") and image_id is not None:
            info = ds.coco.loadImgs([image_id])[0]
            file_name = info.get("file_name")
    except Exception:
        pass
    return image_id, file_name


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_panel(
    out_path: Path,
    *,
    raw_bands: np.ndarray,
    aligned_bands: np.ndarray,
    ref: np.ndarray,
    diff_pre: np.ndarray,
    diff_post: np.ndarray,
    title: str,
):
    n_bands = raw_bands.shape[0]
    ncols = n_bands + 1
    fig, axes = plt.subplots(4, ncols, figsize=(ncols * 2.4, 8.8))
    for i in range(4):
        for j in range(ncols):
            axes[i, j].axis("off")

    for band_idx in range(n_bands):
        axes[0, band_idx].imshow(raw_bands[band_idx], cmap="gray")
        axes[0, band_idx].set_title(f"raw b{band_idx}")
        axes[1, band_idx].imshow(aligned_bands[band_idx], cmap="gray")
        axes[1, band_idx].set_title(f"aligned b{band_idx}")
        axes[2, band_idx].imshow(diff_pre[band_idx], cmap="magma")
        axes[2, band_idx].set_title(f"diff pre b{band_idx}")
        axes[3, band_idx].imshow(diff_post[band_idx], cmap="magma")
        axes[3, band_idx].set_title(f"diff post b{band_idx}")

    axes[0, -1].imshow(ref, cmap="gray")
    axes[0, -1].set_title("ref")
    axes[1, -1].imshow(ref, cmap="gray")
    axes[1, -1].set_title("ref")
    axes[2, -1].imshow(_safe_minmax(ref), cmap="magma")
    axes[2, -1].set_title("ref")
    axes[3, -1].imshow(_safe_minmax(ref), cmap="magma")
    axes[3, -1].set_title("ref")

    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _save_heatmap_grid(out_path: Path, *, maps: np.ndarray, title: str, cmap: str):
    n_bands = maps.shape[0]
    fig, axes = plt.subplots(1, n_bands, figsize=(n_bands * 2.4, 2.6))
    if n_bands == 1:
        axes = [axes]
    for idx in range(n_bands):
        axes[idx].axis("off")
        axes[idx].imshow(maps[idx], cmap=cmap)
        axes[idx].set_title(f"b{idx}")
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _make_pseudo_bgr(bands: np.ndarray, *, bgr_indices: Tuple[int, int, int]) -> np.ndarray:
    if bands.ndim != 3:
        raise ValueError(f"Expected bands shape (C,H,W), got {bands.shape}")
    b_idx, g_idx, r_idx = bgr_indices
    b = _safe_minmax(bands[b_idx])
    g = _safe_minmax(bands[g_idx])
    r = _safe_minmax(bands[r_idx])
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb, 0.0, 1.0)


def _save_pseudo_match_vis(
    out_path: Path,
    *,
    raw_rgb: np.ndarray,
    aligned_rgb: np.ndarray,
    title: str,
    layout: str = "lr",
):
    if layout not in {"lr", "tb"}:
        raise ValueError(f"Unsupported layout: {layout} (expected 'lr' or 'tb').")
    if layout == "tb":
        fig, axes = plt.subplots(2, 1, figsize=(4.2, 7.4))
        axes = np.asarray(axes).reshape(2)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.2))
    axes[0].axis("off")
    axes[1].axis("off")
    axes[0].imshow(raw_rgb)
    axes[0].set_title("raw bgr")
    axes[1].imshow(aligned_rgb)
    axes[1].set_title("aligned bgr")
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _aggregate(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {}
    return {
        "mean": float(np.nanmean(arr)),
        "median": float(np.nanmedian(arr)),
        "p25": float(np.nanpercentile(arr, 25)),
        "p75": float(np.nanpercentile(arr, 75)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize MS band alignment (ms_band_sep) outputs.")
    parser.add_argument("--config", type=str, required=True, help="Config path (yaml).")
    parser.add_argument("--config-dir", type=str, default="configs", help="Hydra config root.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path.")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory.")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], help="Dataset split.")
    parser.add_argument("--num-samples", type=int, default=8, help="Number of samples to visualize.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument("--indices", type=str, default="", help="Comma-separated dataset indices.")
    parser.add_argument("--device", type=str, default="", help="Device (cuda/cpu).")
    parser.add_argument("--downsample", type=int, default=4, help="Downsample factor when aligner is absent.")
    parser.add_argument("--no-align", action="store_true", help="Skip alignment even if ms_band_sep is enabled.")
    parser.add_argument("--use-ema", action="store_true", help="Use EMA weights if present in checkpoint.")
    parser.add_argument("--pseudo-layout", type=str, default="lr", choices=["lr", "tb"], help="Pseudo BGR layout.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser()
    _ensure_dir(output_dir)

    cfg = load_config(args.config, config_dir=Path(args.config_dir))
    cfg.mode = "test"
    if args.device:
        runtime_cfg = get_config(cfg, "runtime", {}) or {}
        runtime_cfg.device = str(args.device)
        cfg.runtime = runtime_cfg

    trainer_cfg = getattr(cfg, "trainer", None)
    if trainer_cfg is None:
        raise KeyError("Missing trainer config in cfg.")
    trainer = instantiate(trainer_cfg, cfg)
    trainer.init_device(trainer.device)
    trainer.build_dataset()
    trainer.build_model()

    model = trainer.model
    device = trainer.device
    model.eval()

    ckpt_path = Path(args.checkpoint).expanduser()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if args.use_ema and isinstance(state, dict) and state.get("ema") is not None:
        model_state = state.get("ema")
        if hasattr(model_state, "state_dict"):
            model_state = model_state.state_dict()
    else:
        model_state = state.get("model", state)
        if hasattr(model_state, "state_dict"):
            model_state = model_state.state_dict()
    if not isinstance(model_state, dict):
        raise TypeError(f"Unexpected checkpoint payload: {type(model_state)}")
    model_state = _remap_ultralytics_state_dict(model_state)
    model_ref = model.module if hasattr(model, "module") else model
    compatible = _filter_compatible_state_dict(model_ref, model_state)
    model_ref.load_state_dict(compatible, strict=False)

    dataset = {
        "train": trainer.train_dataset,
        "val": trainer.validation_dataset,
        "test": trainer.test_dataset,
    }.get(args.split)
    if dataset is None:
        raise RuntimeError(f"Dataset split not available: {args.split}")

    backbone = _find_backbone(model)
    rgb_in_chs = int(getattr(backbone, "rgb_in_chs", 0) or 0)
    ms_in_chs = int(getattr(backbone, "ms_in_chs", 0) or 0)
    ms_band_sep = getattr(backbone, "ms_band_sep_stem", None)
    aligner = getattr(ms_band_sep, "aligner", None) if ms_band_sep is not None else None

    if ms_in_chs <= 0:
        raise RuntimeError("Model does not expose ms_in_chs; cannot run MS alignment visualization.")

    indices = []
    if args.indices:
        indices = [int(x.strip()) for x in args.indices.split(",") if x.strip() != ""]
    sample_indices = _sample_indices(len(dataset), num=args.num_samples, seed=args.seed, indices=indices)
    if not sample_indices:
        raise RuntimeError("No valid sample indices selected.")

    records = []
    sample_summaries = []
    per_band_metrics = {}
    attention_norm = str(getattr(getattr(aligner, "aligner", None), "attention_norm", "softmax"))
    align_corners = bool(getattr(getattr(aligner, "aligner", None), "align_corners", True))
    padding_mode = str(getattr(getattr(aligner, "aligner", None), "padding_mode", "border"))

    for idx in sample_indices:
        sample, _ = dataset[idx]
        if isinstance(sample, dict):
            ms = sample.get("ms")
            if ms is None:
                raise RuntimeError("Expected ms tensor in dataset sample but got None.")
        else:
            ms = sample[rgb_in_chs : rgb_in_chs + ms_in_chs]

        image_id, file_name = _get_sample_meta(dataset, idx)
        title = f"idx={idx}"
        if file_name:
            title = f"{title} file={file_name}"

        ms = ms.unsqueeze(0).to(device)
        with torch.no_grad():
            debug = {}
            z = None
            aligned_z = None
            if aligner is not None and not args.no_align:
                z = ms_band_sep.embed(ms)
                aligned_z, _, debug = aligner(z, return_debug=True)
                target_hw = z.shape[-2:]
            else:
                h, w = ms.shape[-2:]
                target_hw = (max(1, h // args.downsample), max(1, w // args.downsample))

            ms_ds = F.interpolate(ms, size=target_hw, mode="bilinear", align_corners=False).cpu()
            ref_raw = ms_ds.mean(dim=1, keepdim=True)

            aligned_raw = []
            offset_mag_maps = []
            attn_entropy_maps = []
            for band_idx in range(ms_in_chs):
                raw_band = ms_ds[:, band_idx : band_idx + 1]
                if debug:
                    offset_x = debug["offset_x"][:, band_idx]
                    offset_y = debug["offset_y"][:, band_idx]
                    attn = debug["attn"][:, band_idx]
                    if offset_x.ndim == 5:
                        offset_x = offset_x.mean(dim=1)
                        offset_y = offset_y.mean(dim=1)
                    aligned_band = _warp_single_channel(
                        raw_band.to(device),
                        offset_x=offset_x,
                        offset_y=offset_y,
                        attn=attn,
                        align_corners=align_corners,
                        padding_mode=padding_mode,
                        attention_norm=attention_norm,
                    ).cpu()
                    offset_mag = _compute_offset_mag(
                        offset_x.to(device),
                        offset_y.to(device),
                        attn.to(device),
                        attention_norm=attention_norm,
                    ).cpu()
                    attn_entropy = _compute_attn_entropy(attn.to(device), attention_norm=attention_norm).cpu()
                else:
                    aligned_band = raw_band.clone()
                    offset_mag = None
                    attn_entropy = None

                aligned_raw.append(aligned_band)
                if offset_mag is not None:
                    offset_mag_maps.append(offset_mag)
                if attn_entropy is not None:
                    attn_entropy_maps.append(attn_entropy)

                raw_norm = _normalize_metrics(raw_band)
                aligned_norm = _normalize_metrics(aligned_band)
                ref_norm = _normalize_metrics(ref_raw)

                record = {
                    "index": idx,
                    "image_id": image_id,
                    "file_name": file_name,
                    "band": band_idx,
                    "raw_l1": float((raw_norm - ref_norm).abs().mean()),
                    "aligned_l1": float((aligned_norm - ref_norm).abs().mean()),
                    "raw_cos": _cosine_sim(raw_norm, ref_norm),
                    "aligned_cos": _cosine_sim(aligned_norm, ref_norm),
                }

                if debug and z is not None and aligned_z is not None:
                    ref_embed = debug.get("ref")
                    if ref_embed is not None:
                        ref_embed = ref_embed.detach().cpu()
                        band_embed = z[:, band_idx].detach().cpu()
                        band_aligned = aligned_z[:, band_idx].detach().cpu()
                        record["raw_embed_cos"] = _cosine_sim(
                            _normalize_metrics(band_embed), _normalize_metrics(ref_embed)
                        )
                        record["aligned_embed_cos"] = _cosine_sim(
                            _normalize_metrics(band_aligned), _normalize_metrics(ref_embed)
                        )

                if offset_mag is not None:
                    record["offset_mag_mean"] = float(offset_mag.mean())
                    record["offset_mag_p95"] = float(torch.quantile(offset_mag.flatten(), 0.95))
                if attn_entropy is not None:
                    record["attn_entropy_mean"] = float(attn_entropy.mean())
                    record["attn_entropy_p95"] = float(torch.quantile(attn_entropy.flatten(), 0.95))

                records.append(record)
                per_band_metrics.setdefault(band_idx, []).append(record)

            aligned_raw = torch.cat(aligned_raw, dim=1)
            raw_np = ms_ds.squeeze(0).numpy()
            aligned_np = aligned_raw.squeeze(0).numpy()
            ref_np = ref_raw.squeeze(0).numpy()

            raw_vis = np.stack([_safe_minmax(raw_np[i]) for i in range(ms_in_chs)], axis=0)
            aligned_vis = np.stack([_safe_minmax(aligned_np[i]) for i in range(ms_in_chs)], axis=0)
            ref_vis = _safe_minmax(ref_np)
            diff_pre = np.stack([_safe_minmax(np.abs(raw_np[i] - ref_np)) for i in range(ms_in_chs)], axis=0)
            diff_post = np.stack([_safe_minmax(np.abs(aligned_np[i] - ref_np)) for i in range(ms_in_chs)], axis=0)

            viz_dir = output_dir / "viz"
            _save_panel(
                viz_dir / f"sample_{idx:05d}_panel.png",
                raw_bands=raw_vis,
                aligned_bands=aligned_vis,
                ref=ref_vis,
                diff_pre=diff_pre,
                diff_post=diff_post,
                title=title,
            )

            if ms_in_chs >= 3:
                raw_rgb = _make_pseudo_bgr(raw_np, bgr_indices=(0, 1, 2))
                aligned_rgb = _make_pseudo_bgr(aligned_np, bgr_indices=(0, 1, 2))
                _save_pseudo_match_vis(
                    viz_dir / f"sample_{idx:05d}_pseudo_bgr.png",
                    raw_rgb=raw_rgb,
                    aligned_rgb=aligned_rgb,
                    title=title,
                    layout=args.pseudo_layout,
                )

            if offset_mag_maps:
                offset_vis = np.stack(
                    [_safe_minmax(m.squeeze(0).numpy()) for m in offset_mag_maps], axis=0
                )
                _save_heatmap_grid(
                    viz_dir / f"sample_{idx:05d}_offset_mag.png",
                    maps=offset_vis,
                    title="offset magnitude",
                    cmap="viridis",
                )
            if attn_entropy_maps:
                attn_vis = np.stack(
                    [_safe_minmax(m.squeeze(0).numpy()) for m in attn_entropy_maps], axis=0
                )
                _save_heatmap_grid(
                    viz_dir / f"sample_{idx:05d}_attn_entropy.png",
                    maps=attn_vis,
                    title="attention entropy",
                    cmap="viridis",
                )

            sample_metrics = [r for r in records if r["index"] == idx]
            if sample_metrics:
                avg_raw_l1 = np.mean([r["raw_l1"] for r in sample_metrics])
                avg_aligned_l1 = np.mean([r["aligned_l1"] for r in sample_metrics])
                avg_raw_cos = np.mean([r["raw_cos"] for r in sample_metrics])
                avg_aligned_cos = np.mean([r["aligned_cos"] for r in sample_metrics])
                sample_summaries.append(
                    {
                        "index": idx,
                        "image_id": image_id,
                        "file_name": file_name,
                        "raw_l1_mean": float(avg_raw_l1),
                        "aligned_l1_mean": float(avg_aligned_l1),
                        "raw_cos_mean": float(avg_raw_cos),
                        "aligned_cos_mean": float(avg_aligned_cos),
                    }
                )

    summary = {
        "config": str(Path(args.config).expanduser()),
        "checkpoint": str(ckpt_path),
        "split": args.split,
        "num_samples": len(sample_indices),
        "ms_in_chs": ms_in_chs,
        "attention_norm": attention_norm,
        "aligner_enabled": bool(aligner is not None and not args.no_align),
    }

    if records:
        summary["raw_l1"] = _aggregate([r["raw_l1"] for r in records])
        summary["aligned_l1"] = _aggregate([r["aligned_l1"] for r in records])
        summary["raw_cos"] = _aggregate([r["raw_cos"] for r in records])
        summary["aligned_cos"] = _aggregate([r["aligned_cos"] for r in records])

        embed_raw = [r.get("raw_embed_cos") for r in records if r.get("raw_embed_cos") is not None]
        embed_aligned = [r.get("aligned_embed_cos") for r in records if r.get("aligned_embed_cos") is not None]
        if embed_raw and embed_aligned:
            summary["raw_embed_cos"] = _aggregate(embed_raw)
            summary["aligned_embed_cos"] = _aggregate(embed_aligned)

        offset_mean = [r.get("offset_mag_mean") for r in records if r.get("offset_mag_mean") is not None]
        if offset_mean:
            summary["offset_mag_mean"] = _aggregate(offset_mean)
        attn_entropy = [r.get("attn_entropy_mean") for r in records if r.get("attn_entropy_mean") is not None]
        if attn_entropy:
            summary["attn_entropy_mean"] = _aggregate(attn_entropy)

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "per_sample.json").write_text(json.dumps(sample_summaries, indent=2), encoding="utf-8")
    (output_dir / "per_band.json").write_text(json.dumps(per_band_metrics, indent=2), encoding="utf-8")
    (output_dir / "per_band_records.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records),
        encoding="utf-8",
    )

    print(f"Saved alignment visualization to: {output_dir}")


if __name__ == "__main__":
    main()
