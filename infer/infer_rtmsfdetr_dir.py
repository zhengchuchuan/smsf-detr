from __future__ import annotations

"""
Directory inference for RTMSFDETR without annotations (RGB+MSI).

This script loads a trained checkpoint and runs inference on a folder
containing paired RGB/MSI images, then saves visualizations and JSON
predictions.
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
import torchvision.transforms.functional as TF
from omegaconf import OmegaConf, open_dict
from PIL import Image, ImageDraw, ImageFont
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from engines.core.parse_config import get_config, load_config
from engines.trainer.base_trainer import (
    _filter_compatible_state_dict,
    _register_yolo_pickle_alias,
    _remap_ultralytics_state_dict,
)
from datasets.multispectral_coco import (
    _letterbox_ms_tensor,
    _load_msi_as_tensor,
    _normalize_ms_tensor,
    _resize_ms_tensor,
    select_annotation_file,
)
from hydra.utils import instantiate


def _as_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def _load_config_any(
    config: str | Path, *, config_dir: str | Path, overrides: list[str] | None
) -> tuple[Any, Path | None]:
    """
    Supports:
      1) Hydra config under configs/
      2) outputs/**/config.yaml
    """
    config_path = _as_path(config).expanduser()
    if config_path.is_file():
        cfg = OmegaConf.load(str(config_path))
        OmegaConf.set_struct(cfg, False)
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
            OmegaConf.set_struct(cfg, False)
        return cfg, config_path
    cfg = load_config(config, config_dir=_as_path(config_dir), overrides=overrides)
    return cfg, None


def _resolve_checkpoint(checkpoint: str | None, config_path: Path | None) -> Path | None:
    if checkpoint:
        return _as_path(checkpoint).expanduser()
    if config_path is not None and config_path.is_file():
        for name in ("checkpoint_best.pth", "checkpoint.pth"):
            candidate = config_path.parent / name
            if candidate.is_file():
                return candidate
    return None


def _sanitize_dirname(name: str) -> str:
    name = str(name).strip().replace("/", "_").replace("\\", "_")
    name = "_".join([p for p in name.split() if p])
    return name or "data"


def _iter_images(input_dir: Path, *, recursive: bool) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    if recursive:
        paths = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    else:
        paths = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(paths)


def _find_pair_dirs(dataset_dir: Path, split: str) -> tuple[Path, Path]:
    candidates = [
        (dataset_dir / "rgb" / split, dataset_dir / "msi" / split),
        (dataset_dir / split / "rgb", dataset_dir / split / "msi"),
        (dataset_dir / "rgb", dataset_dir / "msi"),
    ]
    for rgb_dir, msi_dir in candidates:
        if rgb_dir.is_dir() and msi_dir.is_dir():
            return rgb_dir, msi_dir
    raise FileNotFoundError(
        f"Cannot locate rgb/msi directories under dataset_dir={dataset_dir} (split={split})."
    )


def _resolve_msi_path(msi_dir: Path, rgb_rel: Path, *, ms_suffix: str) -> Path | None:
    rel = Path(rgb_rel)
    stem = rel.stem
    if rel.suffix.lower() not in {".tif", ".tiff"}:
        rel = rel.with_suffix(ms_suffix)
    candidate = msi_dir / rel
    if candidate.is_file():
        return candidate
    fallback = msi_dir / rel.parent / f"{stem}{ms_suffix}"
    if fallback.is_file():
        return fallback
    same_name = msi_dir / rgb_rel
    if same_name.is_file():
        return same_name
    globbed = next(iter((msi_dir / rel.parent).glob(f"{stem}.*")), None)
    if globbed is not None and globbed.is_file():
        return globbed
    return None


def _normalize_rgb(
    rgb_tensor: torch.Tensor,
    *,
    mode: str,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> torch.Tensor:
    mode = str(mode).lower()
    if mode == "imagenet":
        return TF.normalize(rgb_tensor, mean=mean, std=std)
    if mode == "linear":
        return rgb_tensor
    if mode == "image_max":
        return rgb_tensor / rgb_tensor.amax().clamp_min(1e-6)
    if mode == "per_channel_minmax":
        mins = rgb_tensor.amin(dim=(1, 2), keepdim=True)
        maxs = rgb_tensor.amax(dim=(1, 2), keepdim=True)
        return (rgb_tensor - mins) / (maxs - mins).clamp_min(1e-6)
    raise ValueError(f"Unknown rgb_normalize_mode={mode}")


def _letterbox_rgb_msi(
    rgb_img: Image.Image,
    ms_tensor: torch.Tensor | None,
    *,
    out_size: int,
    rgb_fill: int,
    ms_fill: float,
) -> tuple[Image.Image, torch.Tensor | None, dict[str, float]]:
    w0, h0 = rgb_img.size
    if w0 <= 0 or h0 <= 0:
        raise ValueError(f"invalid image size: {(w0, h0)}")
    scale = min(out_size / float(w0), out_size / float(h0))
    new_w = int(round(w0 * scale))
    new_h = int(round(h0 * scale))
    new_w = max(1, min(new_w, out_size))
    new_h = max(1, min(new_h, out_size))

    pad_w = out_size - new_w
    pad_h = out_size - new_h
    pad_left = int(pad_w // 2)
    pad_right = int(pad_w - pad_left)
    pad_top = int(pad_h // 2)
    pad_bottom = int(pad_h - pad_top)

    resized_rgb = rgb_img.resize((new_w, new_h), resample=Image.BILINEAR)
    canvas = Image.new("RGB", (out_size, out_size), color=(int(rgb_fill), int(rgb_fill), int(rgb_fill)))
    canvas.paste(resized_rgb, (pad_left, pad_top))

    ms_out = None
    if ms_tensor is not None:
        ms_out = _letterbox_ms_tensor(
            ms_tensor,
            out_size=out_size,
            scale=scale,
            pad_left=pad_left,
            pad_top=pad_top,
            pad_right=pad_right,
            pad_bottom=pad_bottom,
            pad_value=float(ms_fill),
        )

    meta = {
        "scale": float(scale),
        "pad_left": float(pad_left),
        "pad_top": float(pad_top),
        "out_size": float(out_size),
    }
    return canvas, ms_out, meta


def _de_letterbox_xyxy(
    boxes: torch.Tensor, *, scale: float, pad_left: float, pad_top: float, orig_w: float, orig_h: float
) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes
    out = boxes.clone()
    out[:, 0::2] -= float(pad_left)
    out[:, 1::2] -= float(pad_top)
    out = out / float(max(scale, 1e-12))
    out[:, 0::2].clamp_(0.0, float(orig_w))
    out[:, 1::2].clamp_(0.0, float(orig_h))
    return out


def _normalize_class_names(names: Any) -> list[str]:
    if not names:
        return []
    try:
        from omegaconf import ListConfig  # type: ignore
    except Exception:
        ListConfig = ()  # type: ignore
    if isinstance(names, ListConfig):
        names = list(names)
    if not isinstance(names, (list, tuple)):
        return []
    normalized: list[str] = []
    for item in names:
        if item is None:
            continue
        s = str(item).strip()
        if not s:
            continue
        if s.lower() in {"background", "_background_", "bg"}:
            continue
        normalized.append(s)
    return normalized


@dataclass(frozen=True)
class DetItem:
    label: int
    score: float
    box_xyxy: list[float]
    class_name: str | None = None


def _load_font(font_size: int) -> ImageFont.ImageFont | None:
    if font_size <= 0:
        return None
    candidates = [
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=int(font_size))
        except Exception:
            continue
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def _draw_detections(
    image: Image.Image,
    dets: list[DetItem],
    *,
    font_size: int = 0,
    line_width: int = 3,
    show_score: bool = True,
    label_prefix: str | None = None,
    box_color: tuple[int, int, int] = (255, 0, 0),
    text_color: tuple[int, int, int] = (0, 255, 255),
) -> Image.Image:
    img = image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    if font_size <= 0:
        font_size = max(16, int(min(w, h) * 0.03))
    font = _load_font(font_size)
    pad = 2
    line_w = max(1, int(line_width))

    for det in dets:
        x1, y1, x2, y2 = det.box_xyxy
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=line_w)
        name = det.class_name if det.class_name else f"class_{det.label}"
        prefix = label_prefix or ""
        if show_score:
            title = f"{prefix}{name} {det.score:.3f}"
        else:
            title = f"{prefix}{name}"
        if font is not None:
            try:
                tbox0 = draw.textbbox((0, 0), title, font=font)
                text_w = float(tbox0[2] - tbox0[0])
                text_h = float(tbox0[3] - tbox0[1])
                tx = float(x1)
                tx = max(0.0, min(tx, float(w) - (text_w + 2 * pad)))
                # Prefer placing text above the box; add extra margin to avoid the box line.
                ty_out = float(y1) - (text_h + 2 * pad + line_w + 1)
                if ty_out >= 0:
                    ty = ty_out
                else:
                    # Fallback: place inside the box but offset from the border.
                    ty = float(y1) + line_w + 1
                ty = max(0.0, min(ty, float(h) - (text_h + 2 * pad)))
            except Exception:
                tx, ty = float(x1), max(0.0, float(y1))
            draw.text((int(tx + pad), int(ty + pad)), title, fill=text_color, font=font)
        else:
            tx = int(max(0.0, min(float(x1), float(w - 1))))
            ty = int(max(0.0, float(y1) - float(font_size) - 2 * pad - line_w - 1))
            if ty <= 0:
                ty = int(max(0.0, min(float(y1) + line_w + 1, float(h - 1))))
            draw.text((tx, ty), title, fill=text_color)
    return img


def _load_coco_gt(ann_path: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[int, str], dict[str, str]]:
    data = json.load(ann_path.open("r", encoding="utf-8"))
    images = data.get("images", []) or []
    annotations = data.get("annotations", []) or []
    categories = data.get("categories", []) or []

    id_to_file: dict[int, str] = {}
    for img in images:
        img_id = img.get("id")
        file_name = img.get("file_name")
        if img_id is None or file_name is None:
            continue
        id_to_file[int(img_id)] = str(file_name)

    cat_id_to_name = {int(c.get("id")): str(c.get("name", c.get("id"))) for c in categories if c.get("id") is not None}

    anns_by_file: dict[str, list[dict[str, Any]]] = {}
    for ann in annotations:
        img_id = ann.get("image_id")
        if img_id is None:
            continue
        file_name = id_to_file.get(int(img_id))
        if not file_name:
            continue
        anns_by_file.setdefault(file_name, []).append(ann)

    basename_index: dict[str, str] = {}
    dup = set()
    for file_name in anns_by_file.keys():
        base = Path(file_name).name
        if base in basename_index:
            dup.add(base)
        else:
            basename_index[base] = file_name
    for base in dup:
        basename_index.pop(base, None)

    return anns_by_file, cat_id_to_name, basename_index


def _find_gt_anns(
    anns_by_file: dict[str, list[dict[str, Any]]],
    basename_index: dict[str, str],
    *,
    rel_path: Path,
    split: str,
) -> list[dict[str, Any]] | None:
    rel_posix = rel_path.as_posix()
    candidates = [
        rel_posix,
        rel_path.name,
        f"{split}/{rel_posix}",
        f"rgb/{split}/{rel_posix}",
        f"rgb/{rel_posix}",
    ]
    for key in candidates:
        if key in anns_by_file:
            return anns_by_file[key]
    base = rel_path.name
    file_key = basename_index.get(base)
    if file_key:
        return anns_by_file.get(file_key)
    return None


def _msi_channel_to_pil(ms_tensor: torch.Tensor, channel_idx: int | None) -> tuple[Image.Image, int]:
    if ms_tensor.ndim != 3:
        raise ValueError(f"ms_tensor must have shape [C,H,W], got {tuple(ms_tensor.shape)}")
    c = int(ms_tensor.shape[0])
    if c <= 0:
        raise ValueError("ms_tensor has no channels.")
    if channel_idx is None or channel_idx < 0 or channel_idx >= c:
        channel_idx = c // 2
    ch = ms_tensor[int(channel_idx)].detach().cpu().numpy()
    ch = np.asarray(ch, dtype=np.float32)
    finite = np.isfinite(ch)
    if not finite.any():
        scaled = np.zeros_like(ch, dtype=np.uint8)
    else:
        vmin = float(np.min(ch[finite]))
        vmax = float(np.max(ch[finite]))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            scaled = np.zeros_like(ch, dtype=np.uint8)
        else:
            norm = (ch - vmin) / (vmax - vmin)
            scaled = (np.clip(norm, 0.0, 1.0) * 255.0).astype(np.uint8)
    img = Image.fromarray(scaled, mode="L").convert("RGB")
    return img, int(channel_idx)


def _build_sample(
    *,
    rgb_path: Path,
    msi_path: Path | None,
    img_size: int,
    shared_transform: str,
    rgb_normalize_mode: str,
    rgb_mean: tuple[float, float, float],
    rgb_std: tuple[float, float, float],
    ms_normalize_mode: str,
    ms_fixed_scale: float | None,
    ms_center_to_rgb_range: bool,
    ms_expected_channels: int | None,
    use_rgb_input: bool,
    use_msi_input: bool,
    msi_vis_channel: int | None,
) -> tuple[
    dict[str, torch.Tensor] | torch.Tensor,
    Image.Image,
    dict[str, Any],
    Image.Image | None,
]:
    rgb_img = Image.open(rgb_path).convert("RGB")
    orig_w, orig_h = rgb_img.size
    rgb_orig = rgb_img.copy()

    ms_tensor = None
    msi_vis = None
    msi_vis_channel_used = None
    if use_msi_input:
        if msi_path is None:
            raise FileNotFoundError(f"Missing MSI file for {rgb_path}")
        ms_tensor = _load_msi_as_tensor(msi_path, expected_channels=ms_expected_channels)
        if msi_vis_channel is not None:
            msi_vis, msi_vis_channel_used = _msi_channel_to_pil(ms_tensor, msi_vis_channel)
        if ms_tensor.shape[1:] != (orig_h, orig_w):
            raise ValueError(
                f"RGB/MSI size mismatch: rgb={(orig_h, orig_w)} msi={tuple(ms_tensor.shape[1:])} ({rgb_path})"
            )

    shared_transform = str(shared_transform).lower()
    letterbox_meta: dict[str, float] | None = None
    if shared_transform == "letterbox":
        rgb_fill = 114
        ms_fill = 0.0
        rgb_img, ms_tensor, meta = _letterbox_rgb_msi(
            rgb_img,
            ms_tensor,
            out_size=int(img_size),
            rgb_fill=rgb_fill,
            ms_fill=ms_fill,
        )
        letterbox_meta = meta
        target_size = (int(img_size), int(img_size))
    else:
        rgb_img = rgb_img.resize((int(img_size), int(img_size)), resample=Image.BILINEAR)
        if ms_tensor is not None:
            ms_tensor = _resize_ms_tensor(ms_tensor, (int(img_size), int(img_size)))
        target_size = (orig_h, orig_w)

    rgb_tensor = None
    if use_rgb_input:
        rgb_tensor = TF.to_tensor(rgb_img)
        rgb_tensor = _normalize_rgb(
            rgb_tensor,
            mode=rgb_normalize_mode,
            mean=rgb_mean,
            std=rgb_std,
        )

    if ms_tensor is not None:
        ms_tensor = _normalize_ms_tensor(ms_tensor, mode=ms_normalize_mode, scale_value=ms_fixed_scale)
        if ms_center_to_rgb_range and ms_normalize_mode in {
            "per_channel_minmax",
            "tensor_minmax",
            "image_max",
            "fixed_scale",
        }:
            ms_tensor = (ms_tensor - 0.5) / 0.5
        if rgb_tensor is not None and ms_tensor.shape[1:] != rgb_tensor.shape[1:]:
            ms_tensor = _resize_ms_tensor(ms_tensor, rgb_tensor.shape[1:])

    if use_rgb_input and use_msi_input:
        if rgb_tensor is None or ms_tensor is None:
            raise RuntimeError("RGB/MSI inputs requested but missing.")
        sample = {"rgb": rgb_tensor, "ms": ms_tensor}
    elif use_rgb_input:
        if rgb_tensor is None:
            raise RuntimeError("RGB input requested but missing.")
        sample = rgb_tensor
    elif use_msi_input:
        if ms_tensor is None:
            raise RuntimeError("MSI input requested but missing.")
        sample = ms_tensor
    else:
        raise ValueError("At least one modality must be enabled.")

    meta = {
        "orig_h": int(orig_h),
        "orig_w": int(orig_w),
        "target_h": int(target_size[0]),
        "target_w": int(target_size[1]),
        "letterbox": letterbox_meta,
        "msi_vis_channel": msi_vis_channel_used,
    }
    return sample, rgb_orig, meta, msi_vis


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RTMSFDETR directory inference (no labels)")
    parser.add_argument("--config", required=True, help="Hydra config or outputs/**/config.yaml")
    parser.add_argument("--config-dir", default="configs", help="Hydra config root (default: configs)")
    parser.add_argument("--opts", nargs="*", default=None, help="Optional overrides KEY=VALUE")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path (default: pick from config dir)")
    parser.add_argument("--dataset-dir", default=None, help="Dataset root containing rgb/msi folders")
    parser.add_argument("--rgb-dir", default=None, help="RGB image directory")
    parser.add_argument("--msi-dir", default=None, help="MSI image directory")
    parser.add_argument("--split", default="test", help="Split name under dataset-dir (default: test)")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan rgb-dir")
    parser.add_argument("--output-dir", default="outputs/infer/rtmsfdetr", help="Output directory")
    parser.add_argument("--device", default="cuda", help="cpu/cuda/cuda:0")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--score-thr", type=float, default=0.3, help="Score threshold")
    parser.add_argument("--max-dets", type=int, default=100, help="Max detections per image")
    parser.add_argument("--save-vis", action="store_true", help="Save visualization images")
    parser.add_argument("--save-msi-vis", action="store_true", help="Save MSI single-channel visualization")
    parser.add_argument(
        "--msi-channel",
        type=int,
        default=-1,
        help="MSI channel index (0-based). Default: middle channel.",
    )
    parser.add_argument("--save-gt-vis", action="store_true", help="Save GT-only visualization images (requires COCO annotations)")
    parser.add_argument("--ann-file", default=None, help="COCO annotation json for GT visualization")
    parser.add_argument("--vis-subdir", default="", help="Visualization subdir name (auto if empty)")
    parser.add_argument("--vis-font-size", type=int, default=0, help="Font size (0=auto)")
    parser.add_argument("--vis-line-width", type=int, default=3, help="BBox line width")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of images (0=all)")
    parser.add_argument("--use-ema", action="store_true", help="Use EMA weights if present in checkpoint")
    parser.add_argument("--weights-only", action="store_true", help="Load checkpoint with weights_only=True")
    parser.add_argument("--amp", action="store_true", help="Enable AMP (CUDA only)")
    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg, config_path = _load_config_any(args.config, config_dir=args.config_dir, overrides=list(args.opts or []))
    with open_dict(cfg):
        cfg.mode = "test"
        if "runtime" not in cfg:
            cfg.runtime = {}
        cfg.runtime.device = str(args.device)

    ckpt_path = _resolve_checkpoint(args.checkpoint, config_path)
    if ckpt_path is None or (not ckpt_path.is_file()):
        raise FileNotFoundError("No checkpoint found; please pass --checkpoint.")

    data_cfg = get_config(cfg, "data", {}) or {}
    train_cfg = get_config(cfg, "train", {}) or {}
    model_cfg = get_config(cfg, "model", {}) or {}

    rgb_dir = _as_path(args.rgb_dir).expanduser() if args.rgb_dir else None
    msi_dir = _as_path(args.msi_dir).expanduser() if args.msi_dir else None
    if rgb_dir is None or msi_dir is None:
        if args.dataset_dir is None:
            raise ValueError("Please provide --dataset-dir or both --rgb-dir/--msi-dir.")
        rgb_dir, msi_dir = _find_pair_dirs(_as_path(args.dataset_dir).expanduser(), args.split)
    if not rgb_dir.is_dir():
        raise NotADirectoryError(f"rgb-dir not found: {rgb_dir}")
    if not msi_dir.is_dir():
        raise NotADirectoryError(f"msi-dir not found: {msi_dir}")

    # Build model via trainer for consistency with RTMSFDETR config.
    trainer_cfg = getattr(cfg, "trainer", None)
    if trainer_cfg is None:
        raise KeyError("Missing trainer in config.")
    trainer = instantiate(trainer_cfg, cfg)
    trainer.init_device(str(args.device))
    model = trainer.build_model()
    trainer.build_criterion()
    postprocess = trainer.criterion_extras.get("postprocess")
    if postprocess is None:
        raise RuntimeError("Postprocess not available from trainer.criterion_extras")
    postprocess = postprocess.to(trainer.device)

    # Load checkpoint
    _register_yolo_pickle_alias()
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=bool(args.weights_only))
    if args.use_ema and isinstance(checkpoint, Mapping) and checkpoint.get("ema") is not None:
        model_state = checkpoint["ema"]
    else:
        model_state = checkpoint.get("model", checkpoint)
    if hasattr(model_state, "state_dict"):
        model_state = model_state.state_dict()
    if not isinstance(model_state, Mapping):
        raise TypeError(f"Failed to parse state_dict from checkpoint: {type(model_state)}")
    model_state = _remap_ultralytics_state_dict(model_state)
    model_ref = model.module if hasattr(model, "module") else model
    compatible = _filter_compatible_state_dict(model_ref, model_state)
    missing, unexpected = model_ref.load_state_dict(compatible, strict=False)
    if missing or unexpected:
        logging.warning("load_state_dict: missing=%d unexpected=%d", len(missing), len(unexpected))

    model.eval()
    model.to(trainer.device)
    torch.set_grad_enabled(False)

    img_size = int(get_config(data_cfg, "img_size", get_config(train_cfg, "img_size", get_config(model_cfg, "img_size", 640))))
    shared_transform = str(get_config(data_cfg, "shared_transform", "simple")).lower()
    rgb_normalize_mode = str(get_config(data_cfg, "rgb_normalize_mode", "imagenet"))
    rgb_mean = tuple(float(x) for x in get_config(data_cfg, "rgb_mean", (0.485, 0.456, 0.406)))
    rgb_std = tuple(float(x) for x in get_config(data_cfg, "rgb_std", (0.229, 0.224, 0.225)))
    ms_normalize_mode = str(get_config(data_cfg, "ms_normalize_mode", "per_channel_minmax"))
    ms_fixed_scale = get_config(data_cfg, "ms_fixed_scale", None)
    ms_center_to_rgb_range = bool(get_config(data_cfg, "ms_center_to_rgb_range", False))
    ms_expected_channels = get_config(data_cfg, "ms_expected_channels", get_config(data_cfg, "ms_input_channels", None))
    ms_suffix = str(get_config(data_cfg, "ms_suffix", ".tif"))
    use_rgb_input = bool(get_config(data_cfg, "use_rgb_input", True))
    use_msi_input = bool(get_config(data_cfg, "use_msi_input", True))

    class_names = _normalize_class_names(get_config(data_cfg, "class_names", None))
    num_obj_classes = int(get_config(train_cfg, "num_classes", 0) or 0)
    if num_obj_classes <= 0:
        num_obj_classes = len(class_names)

    rgb_paths = _iter_images(rgb_dir, recursive=bool(args.recursive))
    if args.limit and int(args.limit) > 0:
        rgb_paths = rgb_paths[: int(args.limit)]
    if not rgb_paths:
        raise FileNotFoundError(f"No images found in rgb-dir: {rgb_dir}")

    pairs: list[tuple[Path, Path]] = []
    for rgb_path in rgb_paths:
        rel = rgb_path.relative_to(rgb_dir)
        msi_path = _resolve_msi_path(msi_dir, rel, ms_suffix=ms_suffix)
        if msi_path is None:
            logging.warning("Skip (missing MSI): %s", rgb_path)
            continue
        pairs.append((rgb_path, msi_path))

    if not pairs:
        raise FileNotFoundError("No paired RGB/MSI images found.")

    gt_anns_by_file = None
    gt_cat_id_to_name: dict[int, str] | None = None
    gt_basename_index = None
    if args.save_gt_vis:
        ann_path = _as_path(args.ann_file).expanduser() if args.ann_file else None
        if ann_path is None:
            if args.dataset_dir is None:
                raise ValueError("save-gt-vis requires --ann-file or --dataset-dir with annotations.")
            ann_dir = _as_path(args.dataset_dir).expanduser() / "annotations"
            ann_path = select_annotation_file(ann_dir, args.split, prefer_bbox=True)
        if not ann_path.is_file():
            raise FileNotFoundError(f"Annotation file not found: {ann_path}")
        gt_anns_by_file, gt_cat_id_to_name, gt_basename_index = _load_coco_gt(ann_path)

    output_dir = _as_path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_subdir = str(args.vis_subdir).strip()
    if not vis_subdir:
        stamp = datetime.now().strftime("%Y%m%d-%H%M")
        vis_subdir = f"{_sanitize_dirname(rgb_dir.name)}-{stamp}"
    if args.save_vis:
        (output_dir / vis_subdir).mkdir(parents=True, exist_ok=True)

    logging.info("config=%s", args.config)
    logging.info("checkpoint=%s", ckpt_path)
    logging.info("rgb_dir=%s", rgb_dir)
    logging.info("msi_dir=%s", msi_dir)
    logging.info("output_dir=%s", output_dir)
    logging.info("shared_transform=%s img_size=%s", shared_transform, img_size)

    results: list[dict[str, Any]] = []
    batch_size = max(1, int(args.batch_size))
    score_thr = float(args.score_thr)
    max_dets = max(1, int(args.max_dets))
    amp_enabled = bool(args.amp) and trainer.device.type == "cuda"

    idx = 0
    while idx < len(pairs):
        batch_pairs = pairs[idx : idx + batch_size]
        idx += batch_size

        batch_samples = []
        batch_meta: list[dict[str, Any]] = []
        batch_rgb: list[Image.Image] = []
        batch_msi_vis: list[Image.Image | None] = []
        for rgb_path, msi_path in batch_pairs:
            sample, rgb_orig, meta, msi_vis = _build_sample(
                rgb_path=rgb_path,
                msi_path=msi_path,
                img_size=img_size,
                shared_transform=shared_transform,
                rgb_normalize_mode=rgb_normalize_mode,
                rgb_mean=rgb_mean,
                rgb_std=rgb_std,
                ms_normalize_mode=ms_normalize_mode,
                ms_fixed_scale=ms_fixed_scale,
                ms_center_to_rgb_range=ms_center_to_rgb_range,
                ms_expected_channels=ms_expected_channels,
                use_rgb_input=use_rgb_input,
                use_msi_input=use_msi_input,
                msi_vis_channel=None if not args.save_msi_vis else int(args.msi_channel),
            )
            batch_samples.append(sample)
            batch_meta.append(meta)
            batch_rgb.append(rgb_orig)
            batch_msi_vis.append(msi_vis)

        if isinstance(batch_samples[0], dict):
            batch_input = {}
            keys = batch_samples[0].keys()
            for key in keys:
                batch_input[key] = torch.stack([s[key] for s in batch_samples], dim=0).to(trainer.device)
        else:
            batch_input = torch.stack(batch_samples, dim=0).to(trainer.device)

        if shared_transform == "letterbox":
            target_sizes = torch.tensor(
                [[m["target_h"], m["target_w"]] for m in batch_meta],
                dtype=torch.float32,
                device=trainer.device,
            )
        else:
            target_sizes = torch.tensor(
                [[m["orig_h"], m["orig_w"]] for m in batch_meta],
                dtype=torch.float32,
                device=trainer.device,
            )

        with torch.inference_mode():
            if amp_enabled:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(batch_input)
            else:
                outputs = model(batch_input)
            batch_preds = postprocess(outputs, target_sizes)

        for (rgb_path, _), rgb_orig, meta, msi_vis, pred in zip(
            batch_pairs, batch_rgb, batch_meta, batch_msi_vis, batch_preds
        ):
            boxes = pred.get("boxes")
            labels = pred.get("labels")
            scores = pred.get("scores")
            if boxes is None or labels is None or scores is None:
                dets: list[DetItem] = []
            else:
                boxes_t = boxes.detach().float().cpu()
                labels_t = labels.detach().long().cpu()
                scores_t = scores.detach().float().cpu()

                if shared_transform == "letterbox" and meta.get("letterbox") is not None:
                    lb = meta["letterbox"]
                    boxes_t = _de_letterbox_xyxy(
                        boxes_t,
                        scale=float(lb["scale"]),
                        pad_left=float(lb["pad_left"]),
                        pad_top=float(lb["pad_top"]),
                        orig_w=float(meta["orig_w"]),
                        orig_h=float(meta["orig_h"]),
                    )

                keep = scores_t >= score_thr
                if num_obj_classes > 0:
                    keep = keep & (labels_t >= 0) & (labels_t < num_obj_classes)

                boxes_t = boxes_t[keep]
                labels_t = labels_t[keep]
                scores_t = scores_t[keep]

                if scores_t.numel() > max_dets:
                    order = torch.argsort(scores_t, descending=True)[:max_dets]
                    boxes_t = boxes_t[order]
                    labels_t = labels_t[order]
                    scores_t = scores_t[order]

                dets = []
                for b, l, s in zip(boxes_t.tolist(), labels_t.tolist(), scores_t.tolist()):
                    name = class_names[l] if 0 <= l < len(class_names) else None
                    dets.append(
                        DetItem(
                            label=int(l),
                            score=float(s),
                            box_xyxy=[float(x) for x in b],
                            class_name=name,
                        )
                    )

            results.append(
                {
                    "file": str(rgb_path),
                    "detections": [
                        {
                            "label": d.label,
                            "class_name": d.class_name,
                            "score": d.score,
                            "box_xyxy": d.box_xyxy,
                            "box_xywh": [
                                d.box_xyxy[0],
                                d.box_xyxy[1],
                                d.box_xyxy[2] - d.box_xyxy[0],
                                d.box_xyxy[3] - d.box_xyxy[1],
                            ],
                        }
                        for d in dets
                    ],
                }
            )

            if args.save_vis:
                vis = _draw_detections(
                    rgb_orig,
                    dets,
                    font_size=int(args.vis_font_size),
                    line_width=int(args.vis_line_width),
                    show_score=True,
                    label_prefix="",
                    box_color=(255, 0, 0),
                    text_color=(0, 255, 255),
                )
                out_path = output_dir / vis_subdir / f"{Path(rgb_path).stem}.png"
                vis.save(out_path)
            if args.save_msi_vis and msi_vis is not None:
                msi_out = _draw_detections(
                    msi_vis,
                    dets,
                    font_size=int(args.vis_font_size),
                    line_width=int(args.vis_line_width),
                    show_score=True,
                    label_prefix="",
                    box_color=(255, 0, 0),
                    text_color=(0, 255, 255),
                )
                ch_idx = meta.get("msi_vis_channel")
                ch_tag = f"_msi_c{int(ch_idx):02d}" if ch_idx is not None else "_msi"
                out_path = output_dir / vis_subdir / f"{Path(rgb_path).stem}{ch_tag}.png"
                msi_out.save(out_path)

            if args.save_gt_vis and gt_anns_by_file is not None:
                rel = Path(rgb_path).relative_to(rgb_dir)
                gt_anns = _find_gt_anns(
                    gt_anns_by_file,
                    gt_basename_index or {},
                    rel_path=rel,
                    split=args.split,
                )
                if gt_anns:
                    gt_dets: list[DetItem] = []
                    for ann in gt_anns:
                        if int(ann.get("iscrowd", 0)) == 1:
                            continue
                        bbox = ann.get("bbox")
                        if not bbox or len(bbox) < 4:
                            continue
                        x, y, w, h = [float(v) for v in bbox[:4]]
                        label_id = int(ann.get("category_id", -1))
                        name = None
                        if gt_cat_id_to_name:
                            name = gt_cat_id_to_name.get(label_id)
                        gt_dets.append(
                            DetItem(
                                label=label_id,
                                score=1.0,
                                box_xyxy=[x, y, x + w, y + h],
                                class_name=name,
                            )
                        )
                    if gt_dets:
                        gt_rgb = _draw_detections(
                            rgb_orig,
                            gt_dets,
                            font_size=int(args.vis_font_size),
                            line_width=int(args.vis_line_width),
                            show_score=False,
                            label_prefix="GT:",
                            box_color=(0, 255, 0),
                            text_color=(0, 255, 0),
                        )
                        out_path = output_dir / vis_subdir / f"{Path(rgb_path).stem}_GT.png"
                        gt_rgb.save(out_path)
                        if args.save_msi_vis and msi_vis is not None:
                            gt_msi = _draw_detections(
                                msi_vis,
                                gt_dets,
                                font_size=int(args.vis_font_size),
                                line_width=int(args.vis_line_width),
                                show_score=False,
                                label_prefix="GT:",
                                box_color=(0, 255, 0),
                                text_color=(0, 255, 0),
                            )
                            ch_idx = meta.get("msi_vis_channel")
                            ch_tag = f"_msi_c{int(ch_idx):02d}" if ch_idx is not None else "_msi"
                            out_path = output_dir / vis_subdir / f"{Path(rgb_path).stem}{ch_tag}_GT.png"
                            gt_msi.save(out_path)

    out_json = output_dir / "predictions.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "config": str(args.config),
                "checkpoint": str(ckpt_path),
                "rgb_dir": str(rgb_dir),
                "msi_dir": str(msi_dir),
                "score_thr": score_thr,
                "max_dets": max_dets,
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    logging.info("Done. images=%d json=%s", len(results), out_json)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
