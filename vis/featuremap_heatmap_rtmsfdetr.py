from __future__ import annotations

"""
RTMSF-DETR visualization: feature/encoder/attention heatmaps.

This file is primarily used by:
- scripts/vis_rtmsfdetr_featuremap_{val,test}_oil_20260115_baseline5.sh
- tools/cluster_analysis/run_featuremap_heatmap_panel.sh

Currently supported heat sources:
- feat: channel-reduced feature heatmap (backbone/fpn)
- attn_cam: class-conditioned decoder cross-attention "CAM" (no gradients)

Notes
-----
1) attn_cam is the recommended "class-related" visualization for DETR-like detectors.
   It aggregates deformable cross-attention sampling points weighted by query class scores.
2) This script intentionally uses manual image loading + normalization (RGB/MSI dirs),
   instead of the training dataset pipeline, to help sanity-check preprocessing.
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf, open_dict
from PIL import Image
from torchvision.transforms import functional as tvf

try:
    import cv2  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError("This visualization script requires opencv-python (cv2).") from exc

try:
    import tifffile  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError("This visualization script requires tifffile.") from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.misc import NestedTensor  # noqa: E402


@dataclass(frozen=True)
class PairItem:
    rgb: Path
    msi: Path | None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RTMSF-DETR heatmap visualization (feature/attn_cam).")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", type=str, help="Hydra config path (configs/**.yaml or any yaml).")
    group.add_argument("--resolved-config", type=str, help="Resolved outputs/**/config.yaml.")

    parser.add_argument("--config-dir", type=str, default="configs", help="Hydra config root (default: configs).")
    parser.add_argument("--opts", nargs="*", default=None, help="Optional dotlist overrides.")

    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path (best/last).")
    parser.add_argument(
        "--weights-only",
        action="store_true",
        help="Load checkpoint with weights_only=True (PyTorch >=2.6 safer mode). "
        "Default is weights_only=False for backward compatibility with our training checkpoints.",
    )
    parser.add_argument("--use-ema", action="store_true", help="Prefer EMA weights if present in checkpoint.")

    parser.add_argument("--rgb-dir", type=str, required=True, help="RGB image directory for the split.")
    parser.add_argument("--msi-dir", type=str, required=True, help="MSI tif directory for the split.")
    parser.add_argument("--split", type=str, default="val", help="Split name for metadata/logging.")

    parser.add_argument("--device", type=str, default="cuda", help="cpu/cuda/cuda:0 ...")
    parser.add_argument("--amp", action="store_true", help="Enable autocast (cuda only).")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (default: 1).")
    parser.add_argument("--img-size", type=int, default=0, help="Square resize size (0 => auto/640).")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of samples (0 => no limit).")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan rgb-dir for images.")
    parser.add_argument("--strict-pairs", action="store_true", help="Require RGB/MSI to be present for every sample.")

    parser.add_argument(
        "--heat-source",
        type=str,
        default="feat",
        choices=["feat", "attn_cam"],
        help="feat=feature heatmap; attn_cam=decoder cross-attention CAM (class related).",
    )

    # Feature heatmap settings.
    parser.add_argument("--feat-source", type=str, default="fpn", choices=["fpn", "backbone"], help="Feature source.")
    parser.add_argument("--feat-level", type=int, default=0, help="Feature pyramid level index (0/1/2...).")
    parser.add_argument("--reduce", type=str, default="l2", choices=["l2", "meanabs"], help="Channel reduce method.")

    # attn_cam settings.
    parser.add_argument("--class-name", type=str, default="oil", help="Class name for attn_cam.")
    parser.add_argument("--class-index", type=int, default=-1, help="Explicit class index (0-based); -1 => infer by name.")
    parser.add_argument("--attn-layer", type=int, default=-1, help="Decoder layer index (-1 => last).")
    parser.add_argument("--attn-topk", type=int, default=50, help="Use top-k queries by class score (0 => all).")
    parser.add_argument("--attn-score-thr", type=float, default=0.2, help="Filter queries by sigmoid score.")

    # Compatibility args: kept so existing .sh wrappers can pass them.
    # (This file currently does not implement grad_cam/det_heatmap.)
    parser.add_argument("--gradcam-topk", type=int, default=100)
    parser.add_argument("--gradcam-score-thr", type=float, default=0.0)
    parser.add_argument("--gradcam-source", type=str, default="enc_score")
    parser.add_argument("--gradcam-score-mode", type=str, default="logit")
    parser.add_argument("--gradcam-backward-type", type=str, default="class")
    parser.add_argument("--gradcam-conf-thr", type=float, default=0.01)

    parser.add_argument("--det-score-thr", type=float, default=0.25)
    parser.add_argument("--det-topk", type=int, default=100)
    parser.add_argument("--det-agg", type=str, default="max")
    parser.add_argument("--det-score-mode", type=str, default="sigmoid")

    # Output post-processing.
    parser.add_argument("--normalize", type=str, default="minmax", choices=["none", "minmax", "clip01"], help="Heatmap normalization.")
    parser.add_argument("--blur", type=int, default=5, help="Gaussian blur kernel (odd; 0 disables).")
    parser.add_argument("--alpha", type=float, default=0.45, help="Overlay alpha.")

    parser.add_argument("--output-dir", type=str, default="outputs/repr/rtmsfdetr", help="Output root directory.")
    parser.add_argument("--run-name", type=str, default="", help="Run directory name (empty => auto).")
    return parser.parse_args()


def get_config(node: Any, key: str, default: Any | None = None) -> Any:
    if node is None:
        return default
    if hasattr(node, "get"):
        return node.get(key, default)
    return getattr(node, key, default)


def _sanitize_dirname(name: str) -> str:
    name = str(name).strip().replace(" ", "_")
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in name)


def _pick_device(device: str) -> torch.device:
    d = str(device).strip().lower()
    if d.startswith("cuda") and not torch.cuda.is_available():
        logging.warning("CUDA requested but not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(d)


def _load_config_any(
    config: str | None,
    resolved_config: str | None,
    *,
    config_dir: str,
    overrides: list[str] | None,
) -> Any:
    if resolved_config:
        cfg = OmegaConf.load(resolved_config)
        OmegaConf.set_struct(cfg, False)
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
            OmegaConf.set_struct(cfg, False)
        return cfg
    if not config:
        raise ValueError("Either --config or --resolved-config must be provided.")
    from engines.core.parse_config import load_config

    cfg = load_config(config, config_dir=Path(config_dir), overrides=overrides)
    OmegaConf.set_struct(cfg, False)
    return cfg


def _extract_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(checkpoint, Mapping):
        model_state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    else:
        model_state = checkpoint
    if hasattr(model_state, "state_dict"):
        model_state = model_state.state_dict()
    if not isinstance(model_state, Mapping):
        raise TypeError(f"Cannot extract state_dict from checkpoint type={type(checkpoint)}")
    return model_state  # type: ignore[return-value]


def _filter_compatible_state_dict(model: torch.nn.Module, state_dict: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
    model_sd = model.state_dict()
    filtered: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if k not in model_sd:
            continue
        if model_sd[k].shape != v.shape:
            continue
        filtered[k] = v
    return filtered


def _rgb_normalize(x: torch.Tensor, *, mode: str, rgb_mean: tuple[float, float, float], rgb_std: tuple[float, float, float]) -> torch.Tensor:
    mode = str(mode or "imagenet").lower()
    if mode == "imagenet":
        return tvf.normalize(x, mean=list(rgb_mean), std=list(rgb_std))
    if mode == "linear":
        return x
    if mode == "image_max":
        return x / x.amax().clamp_min(1e-6)
    if mode == "per_channel_minmax":
        mins = x.amin(dim=(1, 2), keepdim=True)
        maxs = x.amax(dim=(1, 2), keepdim=True)
        return (x - mins) / (maxs - mins).clamp_min(1e-6)
    raise ValueError(f"Unsupported rgb_normalize_mode={mode}")


def _load_msi_as_tensor(path: Path, *, expected_channels: int) -> torch.Tensor:
    arr = tifffile.imread(str(path))
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3:
        raise ValueError(f"Unexpected MSI shape={arr.shape} for {path}")
    # tifffile usually returns HWC
    if arr.shape[2] == expected_channels:
        hwc = arr
    elif arr.shape[0] == expected_channels and arr.shape[2] != expected_channels:
        # CHW
        hwc = np.transpose(arr, (1, 2, 0))
    else:
        raise ValueError(f"MSI channels mismatch: expected={expected_channels}, got={arr.shape} for {path}")
    ms = torch.from_numpy(hwc).permute(2, 0, 1).contiguous().float()  # [C,H,W]
    return ms


def _resize_ms_tensor(ms: torch.Tensor, *, size_hw: tuple[int, int]) -> torch.Tensor:
    ms4 = ms.unsqueeze(0)
    out = F.interpolate(ms4, size=size_hw, mode="bilinear", align_corners=False)
    return out.squeeze(0)


def _normalize_ms_tensor(ms: torch.Tensor, *, mode: str, scale_value: float | None) -> torch.Tensor:
    mode = str(mode or "fixed_scale").lower()
    if mode == "none":
        return ms
    if mode == "linear":
        if scale_value is None:
            raise ValueError("ms_normalize_mode=linear requires scale_value.")
        return ms / float(scale_value)
    if mode == "per_channel_minmax":
        mins = ms.amin(dim=(1, 2), keepdim=True)
        maxs = ms.amax(dim=(1, 2), keepdim=True)
        return (ms - mins) / (maxs - mins).clamp_min(1e-6)
    if mode == "tensor_minmax":
        lo = ms.amin()
        hi = ms.amax()
        return (ms - lo) / (hi - lo).clamp_min(1e-6)
    if mode == "image_max":
        return ms / ms.amax().clamp_min(1e-6)
    if mode == "fixed_scale":
        if scale_value is None:
            raise ValueError("ms_normalize_mode=fixed_scale requires scale_value.")
        return torch.clamp(ms / float(scale_value), 0.0, 1.0)
    raise ValueError(f"Unsupported ms_normalize_mode={mode}")


def _normalize_map(x: np.ndarray, mode: str) -> np.ndarray:
    mode = str(mode or "minmax").lower()
    if mode == "none":
        return x.astype(np.float32, copy=False)
    if mode == "clip01":
        return np.clip(x, 0.0, 1.0).astype(np.float32)
    if mode == "minmax":
        lo = float(np.nanmin(x))
        hi = float(np.nanmax(x))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-12:
            return np.zeros_like(x, dtype=np.float32)
        return ((x - lo) / (hi - lo)).astype(np.float32)
    raise ValueError(f"Unsupported normalize={mode}")


def _blend(a: np.ndarray, b: np.ndarray, *, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    out = (1.0 - alpha) * a.astype(np.float32) + alpha * b.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def _resolve_class_index(num_c: int, *, data_cfg: Any, class_name: str, class_index: int) -> int:
    if int(class_index) >= 0:
        idx = int(class_index)
        if idx >= num_c:
            raise IndexError(f"class-index out of range: {idx} >= {num_c}")
        return idx
    if num_c == 1:
        return 0
    raw_names = get_config(data_cfg, "class_names", None)
    names_all = [str(x).strip() for x in (list(raw_names) if raw_names is not None else [])]
    lookup = {n.lower(): i for i, n in enumerate(names_all) if n}
    key = str(class_name).strip().lower()
    if key not in lookup:
        raise ValueError(f"Cannot infer class index for class-name={class_name}. class_names={names_all}")
    idx = int(lookup[key])
    if idx >= num_c:
        raise ValueError(f"class-name={class_name} -> idx={idx} but num_classes={num_c}; use --class-index.")
    return idx


def _patch_ms_deformable_attn_for_vis(cross_attn: Any) -> None:
    """
    Runtime-only patch for MSDeformableAttention to capture attention weights/locations.
    This avoids touching training/inference code paths outside visualization.
    """
    if cross_attn is None:
        return
    if getattr(cross_attn, "_vis_attn_patched", False):
        return

    orig_forward = cross_attn.forward

    def _forward_patched(*args, **kwargs):  # noqa: ANN001 - runtime patch
        if len(args) < 4:
            return orig_forward(*args, **kwargs)
        query = args[0]
        reference_points = args[1]
        value_spatial_shapes = args[3]

        # Default to "not captured" to make failure explicit.
        cross_attn.last_attn_weights = None
        cross_attn.last_sampling_locations = None
        cross_attn.last_value_spatial_shapes = None

        try:
            with torch.no_grad():
                bs, len_q = int(query.shape[0]), int(query.shape[1])
                num_heads = int(getattr(cross_attn, "num_heads"))
                num_points_list = [int(x) for x in getattr(cross_attn, "num_points_list", [])]
                if not num_points_list:
                    num_levels = int(getattr(cross_attn, "num_levels"))
                    num_points = int(getattr(cross_attn, "num_points", 4))
                    num_points_list = [num_points for _ in range(num_levels)]
                sum_points = int(sum(num_points_list))

                sampling_offsets = cross_attn.sampling_offsets(query).reshape(bs, len_q, num_heads, sum_points, 2)
                attn = cross_attn.attention_weights(query).reshape(bs, len_q, num_heads, sum_points)
                attn = torch.softmax(attn, dim=-1)

                sampling_locations = None
                if reference_points.shape[-1] == 2:
                    spatial = torch.as_tensor(value_spatial_shapes, dtype=query.dtype, device=query.device).flip([1])  # (W,H)
                    splits = torch.split(sampling_offsets, num_points_list, dim=3)
                    locs = []
                    for lvl, off_l in enumerate(splits):
                        ref_l = reference_points[:, :, lvl, :].unsqueeze(2).unsqueeze(3)  # [B,Q,1,1,2]
                        norm_l = spatial[lvl].view(1, 1, 1, 1, 2)
                        locs.append(ref_l + off_l / norm_l)
                    sampling_locations = torch.cat(locs, dim=3)
                elif reference_points.shape[-1] == 4:
                    # reference_points in this repo: [B,Q,1,4] (cx,cy,w,h)
                    if hasattr(cross_attn, "num_points_scale"):
                        nps = cross_attn.num_points_scale.to(dtype=query.dtype, device=query.device).unsqueeze(-1)
                    else:
                        tmp = []
                        for n in num_points_list:
                            tmp.extend([1.0 / float(n)] * int(n))
                        nps = torch.tensor(tmp, dtype=query.dtype, device=query.device).unsqueeze(-1)
                    offset_scale = float(getattr(cross_attn, "offset_scale", 0.5))
                    offset = sampling_offsets * nps * reference_points[:, :, None, :, 2:] * offset_scale
                    sampling_locations = reference_points[:, :, None, :, :2] + offset

                cross_attn.last_attn_weights = attn.detach()
                cross_attn.last_sampling_locations = sampling_locations.detach() if sampling_locations is not None else None
                cross_attn.last_value_spatial_shapes = value_spatial_shapes
        except Exception:
            # Best-effort: do not fail model forward, just keep None and let caller raise a clear error.
            pass

        return orig_forward(*args, **kwargs)

    cross_attn.forward = _forward_patched  # type: ignore[method-assign]
    cross_attn._vis_attn_patched = True
    cross_attn._vis_attn_orig_forward = orig_forward


def _attn_cam_from_decoder(
    *,
    scores: torch.Tensor,
    attn_weights: torch.Tensor,
    sampling_locations: torch.Tensor,
    spatial_shapes: list[tuple[int, int]],
    num_points_list: list[int],
    topk: int,
    score_thr: float,
    out_hw: tuple[int, int],
) -> np.ndarray:
    scores_np = scores.detach().float().cpu().numpy()
    if float(score_thr) > 0:
        keep = scores_np >= float(score_thr)
    else:
        keep = np.ones_like(scores_np, dtype=bool)

    if int(topk) > 0 and keep.any():
        idx_all = np.where(keep)[0]
        if idx_all.size > int(topk):
            sel = idx_all[np.argsort(scores_np[idx_all])[-int(topk) :]]
        else:
            sel = idx_all
    else:
        sel = np.where(keep)[0]

    if sel.size == 0:
        return np.zeros(out_hw, dtype=np.float32)

    attn = attn_weights.detach().float().cpu().numpy()[sel]  # [K, n_head, sum_points]
    locs = sampling_locations.detach().float().cpu().numpy()[sel]  # [K, n_head, sum_points, 2]
    score_sel = scores_np[sel].astype(np.float32)

    out_h, out_w = int(out_hw[0]), int(out_hw[1])
    total_map = np.zeros((out_h, out_w), dtype=np.float32)

    start = 0
    for (h_l, w_l), npts in zip(spatial_shapes, num_points_list):
        npts = int(npts)
        if npts <= 0:
            continue
        attn_l = attn[:, :, start : start + npts]
        loc_l = locs[:, :, start : start + npts, :]
        start += npts

        weights = attn_l * score_sel[:, None, None]
        loc_l = np.clip(loc_l, 0.0, 1.0)
        xs = np.rint(loc_l[..., 0] * max(int(w_l) - 1, 1)).astype(np.int64)
        ys = np.rint(loc_l[..., 1] * max(int(h_l) - 1, 1)).astype(np.int64)
        xs = np.clip(xs, 0, max(int(w_l) - 1, 0))
        ys = np.clip(ys, 0, max(int(h_l) - 1, 0))

        level_map = np.zeros((int(h_l), int(w_l)), dtype=np.float32)
        np.add.at(level_map, (ys.reshape(-1), xs.reshape(-1)), weights.reshape(-1))

        if (int(h_l), int(w_l)) != (out_h, out_w):
            level_map = cv2.resize(level_map, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        total_map += level_map

    return total_map


def _scan_pairs(rgb_dir: Path, msi_dir: Path, *, recursive: bool, strict_pairs: bool) -> list[PairItem]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    if recursive:
        rgb_files = [p for p in rgb_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts and not p.name.startswith(".")]
    else:
        rgb_files = [p for p in rgb_dir.iterdir() if p.is_file() and p.suffix.lower() in exts and not p.name.startswith(".")]
    rgb_files = sorted(rgb_files)

    msi_map = {}
    if msi_dir.is_dir():
        for p in msi_dir.iterdir():
            if p.is_file() and not p.name.startswith("."):
                msi_map[p.stem] = p

    pairs: list[PairItem] = []
    for rgb in rgb_files:
        msi = msi_map.get(rgb.stem)
        if strict_pairs and msi is None:
            continue
        pairs.append(PairItem(rgb=rgb, msi=msi))
    return pairs


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = _load_config_any(
        config=str(args.config) if args.config else None,
        resolved_config=str(args.resolved_config) if args.resolved_config else None,
        config_dir=str(args.config_dir),
        overrides=list(args.opts or []),
    )

    device = _pick_device(str(args.device))
    amp_enabled = bool(args.amp) and device.type == "cuda"

    # Make it safe for visualization-only: do not build teacher by default.
    with open_dict(cfg):
        cfg.mode = "test"
        if "runtime" not in cfg:
            cfg.runtime = {}
        cfg.runtime.device = str(device)
        if "model" not in cfg:
            cfg.model = {}
        cfg.model.disable_distill = True
        cfg.model.force_no_pretrain = True

    from hydra.utils import instantiate

    trainer = instantiate(cfg.trainer, cfg)
    model = trainer.build_model()
    model.eval().to(device)

    checkpoint = torch.load(str(args.checkpoint), map_location="cpu", weights_only=bool(args.weights_only))
    model_state = checkpoint
    if bool(args.use_ema) and isinstance(checkpoint, Mapping) and checkpoint.get("ema") is not None:
        model_state = checkpoint["ema"]
    state_dict = _extract_state_dict(model_state)
    compatible = _filter_compatible_state_dict(model, state_dict)
    skipped = len(state_dict) - len(compatible)
    if skipped:
        logging.warning("checkpoint has %d incompatible keys; skipped loading.", skipped)
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    if missing or unexpected:
        logging.info("state_dict loaded: missing=%d unexpected=%d", len(missing), len(unexpected))

    rgb_dir = Path(args.rgb_dir).expanduser()
    msi_dir = Path(args.msi_dir).expanduser()
    pairs = _scan_pairs(rgb_dir, msi_dir, recursive=bool(args.recursive), strict_pairs=bool(args.strict_pairs))
    if int(args.limit) > 0:
        pairs = pairs[: int(args.limit)]
    if not pairs:
        raise RuntimeError("No image pairs found.")

    img_size = int(args.img_size)
    if img_size <= 0:
        img_size = int(get_config(cfg.data, "img_size", 640) or 640)
    img_size = int(img_size)

    out_root = Path(args.output_dir).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)
    run_name = str(args.run_name).strip()
    if not run_name:
        stamp = datetime.now().strftime("%Y%m%d-%H%M")
        run_name = f"{_sanitize_dirname(str(args.split))}-attncam-{stamp}"
    run_dir = out_root / run_name
    out_heat = run_dir / "selection" / "heatmap"
    out_overlay = run_dir / "selection" / "overlay_heatmap"
    out_heat.mkdir(parents=True, exist_ok=True)
    out_overlay.mkdir(parents=True, exist_ok=True)

    blur_k = int(args.blur)
    if blur_k <= 0 or blur_k % 2 == 0:
        blur_k = 0

    # Config-driven preprocessing settings.
    data_cfg = cfg.data
    use_rgb = bool(get_config(data_cfg, "use_rgb_input", True))
    use_msi = bool(get_config(data_cfg, "use_msi_input", True))
    rgb_mode = str(get_config(data_cfg, "rgb_normalize_mode", "imagenet"))
    rgb_mean = tuple(get_config(data_cfg, "rgb_mean", (0.485, 0.456, 0.406)))
    rgb_std = tuple(get_config(data_cfg, "rgb_std", (0.229, 0.224, 0.225)))
    rgb_ch = int(get_config(data_cfg, "rgb_input_channels", 3) or 3)
    ms_ch = int(get_config(data_cfg, "ms_input_channels", 0) or 0)
    ms_mode = str(get_config(data_cfg, "ms_normalize_mode", "fixed_scale"))
    ms_fixed_scale = get_config(data_cfg, "ms_fixed_scale", None)
    ms_center_to_rgb_range = bool(get_config(data_cfg, "ms_center_to_rgb_range", False))

    heat_source = str(args.heat_source).lower().strip()
    logging.info("heat_source=%s class=%s class_index=%d", heat_source, args.class_name, int(args.class_index))
    logging.info("rgb_dir=%s msi_dir=%s N=%d img_size=%d device=%s amp=%s", rgb_dir, msi_dir, len(pairs), img_size, device, amp_enabled)
    logging.info("out=%s", run_dir)

    # Process in mini-batches.
    bs = max(1, int(args.batch_size))
    index_meta: list[dict[str, Any]] = []

    for i0 in range(0, len(pairs), bs):
        batch = pairs[i0 : i0 + bs]

        tensors = []
        vis_rgbs = []
        for it in batch:
            rgb_img = Image.open(it.rgb).convert("RGB")
            rgb_img = rgb_img.resize((img_size, img_size), resample=Image.BILINEAR)
            rgb_np = np.asarray(rgb_img, dtype=np.uint8)
            vis_rgbs.append(rgb_np)

            modalities = []
            if use_rgb:
                x = tvf.to_tensor(rgb_img)  # [3,H,W] in [0,1]
                x = _rgb_normalize(x, mode=rgb_mode, rgb_mean=rgb_mean, rgb_std=rgb_std)
                if x.shape[0] != rgb_ch:
                    raise ValueError(f"Unexpected RGB channels: expected={rgb_ch}, got={x.shape[0]} file={it.rgb}")
                modalities.append(x)

            if use_msi:
                if it.msi is None:
                    raise RuntimeError(f"Missing MSI for {it.rgb} (use --strict-pairs to pre-filter).")
                ms = _load_msi_as_tensor(it.msi, expected_channels=ms_ch)
                ms = _resize_ms_tensor(ms, size_hw=(img_size, img_size))
                ms = _normalize_ms_tensor(ms, mode=ms_mode, scale_value=float(ms_fixed_scale) if ms_fixed_scale is not None else None)
                if ms_center_to_rgb_range and ms_mode.lower() in {"per_channel_minmax", "tensor_minmax", "image_max", "fixed_scale"}:
                    # Map [0,1] -> [-1,1] to roughly match RGB normalization range.
                    ms = (ms - 0.5) / 0.5
                if ms.shape[0] != ms_ch:
                    raise ValueError(f"Unexpected MSI channels: expected={ms_ch}, got={ms.shape[0]} file={it.msi}")
                modalities.append(ms)

            if not modalities:
                raise RuntimeError("Empty modalities (both use_rgb_input/use_msi_input are False).")
            tensors.append(torch.cat(modalities, dim=0))
            index_meta.append({"image": str(it.rgb), "msi": str(it.msi) if it.msi is not None else ""})

        batch_tensor = torch.stack(tensors, dim=0)
        mask = torch.zeros((batch_tensor.shape[0], batch_tensor.shape[2], batch_tensor.shape[3]), dtype=torch.bool)
        samples = NestedTensor(batch_tensor.to(device), mask.to(device))

        detector = model.module if hasattr(model, "module") else model
        if not hasattr(detector, "_prepare_images") or not hasattr(detector, "model"):
            raise TypeError(f"Expected RTDETRv4Detector wrapper, got {type(detector)}")

        with torch.inference_mode():
            if amp_enabled:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    images = detector._prepare_images(samples)  # type: ignore[attr-defined]
                    raw = detector.model  # type: ignore[attr-defined]
                    feats_backbone = raw.backbone(images)
                    feats_fpn = raw.encoder(feats_backbone)
            else:
                images = detector._prepare_images(samples)  # type: ignore[attr-defined]
                raw = detector.model  # type: ignore[attr-defined]
                feats_backbone = raw.backbone(images)
                feats_fpn = raw.encoder(feats_backbone)

        if isinstance(feats_fpn, tuple) and len(feats_fpn) >= 1:
            feats_fpn = feats_fpn[0]

        if heat_source == "feat":
            feat_maps = feats_fpn if str(args.feat_source) == "fpn" else feats_backbone
            if not isinstance(feat_maps, (list, tuple)):
                raise TypeError(f"Expected feature list/tuple, got {type(feat_maps)}")
            level = int(args.feat_level)
            if level < 0 or level >= len(feat_maps):
                raise IndexError(f"feat-level out of range: {level} not in [0,{len(feat_maps)-1}]")
            selected = feat_maps[level]  # [B,C,H,W]
            if selected.ndim != 4:
                raise TypeError(f"Expected selected feature [B,C,H,W], got {tuple(selected.shape)}")
            if str(args.reduce) == "meanabs":
                heat = selected.detach().abs().mean(dim=1)
            else:
                heat = selected.detach().float().pow(2).mean(dim=1).sqrt()
            heat_np = heat.cpu().numpy()
            for j in range(heat_np.shape[0]):
                hmap = _normalize_map(heat_np[j], mode=str(args.normalize))
                h_in, w_in = vis_rgbs[j].shape[:2]
                if hmap.shape != (h_in, w_in):
                    hmap = cv2.resize(hmap.astype(np.float32), (w_in, h_in), interpolation=cv2.INTER_LINEAR)
                if blur_k > 0:
                    hmap = cv2.GaussianBlur(hmap.astype(np.float32), (blur_k, blur_k), sigmaX=0)
                heat_u8 = (np.clip(hmap, 0.0, 1.0) * 255.0).astype(np.uint8)
                heat_bgr = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
                heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
                overlay = _blend(vis_rgbs[j], heat_rgb, alpha=float(args.alpha))
                stem = Path(batch[j].rgb).stem
                Image.fromarray(heat_rgb).save(out_heat / f"{stem}.png")
                Image.fromarray(overlay).save(out_overlay / f"{stem}.png")
            continue

        if heat_source != "attn_cam":
            raise ValueError(f"Unsupported heat_source={heat_source}")

        # attn_cam: run decoder once to populate cross-attention weights
        if not isinstance(feats_fpn, (list, tuple)):
            raise TypeError(f"Expected fpn feature list/tuple, got {type(feats_fpn)}")
        decoder = getattr(raw, "decoder", None)
        if decoder is None:
            raise AttributeError("Raw model has no decoder; cannot use heat_source=attn_cam.")
        dec = getattr(decoder, "decoder", None)
        dec_layers = getattr(dec, "layers", None) if dec is not None else None
        if dec_layers is None or len(dec_layers) == 0:
            raise AttributeError("Decoder has no layers; cannot use heat_source=attn_cam.")

        for layer in dec_layers:
            cross_attn = getattr(layer, "cross_attn", None)
            if cross_attn is not None:
                _patch_ms_deformable_attn_for_vis(cross_attn)

        with torch.inference_mode():
            out = decoder(list(feats_fpn), targets=None)
        if not isinstance(out, dict) or "pred_logits" not in out:
            raise RuntimeError("Decoder output missing pred_logits; cannot build attn_cam.")

        pred_logits = out["pred_logits"]  # [B,Q,C]
        num_c = int(pred_logits.shape[-1])
        class_idx = _resolve_class_index(
            num_c,
            data_cfg=data_cfg,
            class_name=str(args.class_name),
            class_index=int(args.class_index),
        )
        prob_q = torch.sigmoid(pred_logits[..., class_idx])  # [B,Q]

        layer_idx = int(args.attn_layer)
        if layer_idx < 0:
            layer_idx = len(dec_layers) + layer_idx
        if layer_idx < 0 or layer_idx >= len(dec_layers):
            raise IndexError(f"attn-layer out of range: {layer_idx} not in [0,{len(dec_layers)-1}]")
        cross_attn = getattr(dec_layers[layer_idx], "cross_attn", None)
        if cross_attn is None:
            raise AttributeError(f"Decoder layer[{layer_idx}] has no cross_attn.")

        attn_w = getattr(cross_attn, "last_attn_weights", None)
        locs = getattr(cross_attn, "last_sampling_locations", None)
        spatial_shapes = getattr(cross_attn, "last_value_spatial_shapes", None)
        if attn_w is None or locs is None or spatial_shapes is None:
            raise RuntimeError("Attention not captured. Ensure the runtime patch is applied and decoder forward ran.")
        if torch.is_tensor(spatial_shapes):
            spatial_list = [(int(h), int(w)) for h, w in spatial_shapes.detach().cpu().tolist()]
        else:
            spatial_list = [(int(h), int(w)) for h, w in list(spatial_shapes)]
        num_points_list = [int(x) for x in getattr(cross_attn, "num_points_list", [])]
        if not num_points_list:
            raise RuntimeError("cross_attn has empty num_points_list; cannot split points by level.")

        for j in range(int(attn_w.shape[0])):
            heat = _attn_cam_from_decoder(
                scores=prob_q[j],
                attn_weights=attn_w[j],
                sampling_locations=locs[j],
                spatial_shapes=spatial_list,
                num_points_list=num_points_list,
                topk=int(args.attn_topk),
                score_thr=float(args.attn_score_thr),
                out_hw=(int(vis_rgbs[j].shape[0]), int(vis_rgbs[j].shape[1])),
            )
            heat = _normalize_map(heat, mode=str(args.normalize))
            if blur_k > 0:
                heat = cv2.GaussianBlur(heat.astype(np.float32), (blur_k, blur_k), sigmaX=0)
            heat_u8 = (np.clip(heat, 0.0, 1.0) * 255.0).astype(np.uint8)
            heat_bgr = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
            heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
            overlay = _blend(vis_rgbs[j], heat_rgb, alpha=float(args.alpha))
            stem = Path(batch[j].rgb).stem
            Image.fromarray(heat_rgb).save(out_heat / f"{stem}.png")
            Image.fromarray(overlay).save(out_overlay / f"{stem}.png")

    # Save minimal metadata for downstream tooling.
    meta = {
        "config": str(args.resolved_config or args.config),
        "checkpoint": str(args.checkpoint),
        "rgb_dir": str(rgb_dir),
        "msi_dir": str(msi_dir),
        "split": str(args.split),
        "img_size": int(img_size),
        "heat_source": str(args.heat_source),
        "class_name": str(args.class_name),
        "class_index": int(args.class_index),
        "index": index_meta,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    logging.info("Done: run_dir=%s", run_dir)
    logging.info("heatmap=%s overlay=%s", out_heat, out_overlay)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
