from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
    
# python tools/cluster_analysis/repr_oil_heatmap_panel.py --run-dir outputs/repr/rtmsfdetr/oil_rgb_val_k32x16_20260104 --ann data/oil_20260101/annotations/val.json --show oil --draw-boxes-on-right

@dataclass
class RunIndexItem:
    image: str
    msi: str = ""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="把“原图+GT标注”和“oil heatmap overlay”拼成一张对比图，便于查看。"
    )
    parser.add_argument("--run-dir", type=str, required=True, help="repr run 目录（包含 meta.json / selection/overlay_heatmap）。")
    parser.add_argument(
        "--ann",
        type=str,
        default="",
        help="COCO annotations json（默认从 meta.json 记录的 config 推导 data.dataset_dir/annotations/<split>.json）。",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="",
        choices=["train", "val", "test"],
        help="当自动推导 ann 时使用的 split（默认优先 meta.json 的 split，否则 val）。",
    )
    parser.add_argument(
        "--oil-names",
        nargs="*",
        default=["oil"],
        help="在 COCO categories.name 中视作 oil 的类别名（默认 oil）。",
    )
    parser.add_argument(
        "--show",
        type=str,
        default="oil",
        choices=["oil", "all"],
        help="在原图上画哪些 GT bbox（默认 oil）。",
    )
    parser.add_argument("--box-width", type=int, default=3, help="bbox 线宽（默认 3）。")
    parser.add_argument("--font-size", type=int, default=0, help="字体大小（0=自动；默认 0）。")
    parser.add_argument("--gap", type=int, default=12, help="左右拼图间距（默认 12）。")
    parser.add_argument("--margin-top", type=int, default=36, help="顶部标题留白（默认 36）。")
    parser.add_argument("--title-left", type=str, default="RGB + GT", help="左图标题（默认 'RGB + GT'）。")
    parser.add_argument("--title-right", type=str, default="Oil Heatmap", help="右图标题（默认 'Oil Heatmap'）。")
    parser.add_argument("--draw-boxes-on-right", action="store_true", help="也在右侧 heatmap overlay 上画 GT bbox。")
    parser.add_argument("--draw-preds", action="store_true", help="额外叠加模型推理预测框（从 meta.json 的 config/checkpoint 推理）。")
    parser.add_argument("--pred-device", type=str, default="cuda", help="推理设备（cpu/cuda/cuda:0；默认 cuda）。")
    parser.add_argument("--pred-amp", action="store_true", help="推理使用 AMP（仅 CUDA 有效）。")
    parser.add_argument("--pred-score-thr", type=float, default=0.3, help="预测框 score 阈值（默认 0.3）。")
    parser.add_argument("--pred-max-dets", type=int, default=100, help="最多绘制多少个预测框（默认 100）。")
    parser.add_argument(
        "--pred-show",
        type=str,
        default="all",
        choices=["oil", "all"],
        help="绘制哪些预测框：oil=仅 oil 类；all=全部（默认 all）。",
    )
    parser.add_argument("--pred-box-width", type=int, default=2, help="预测框线宽（默认 2）。")
    parser.add_argument("--pred-color", type=str, default="0,255,255", help="预测框颜色 RGB，例如 0,255,255（默认青色）。")
    parser.add_argument("--pred-show-score", action="store_true", help="在预测框上显示 score/label。")
    parser.add_argument("--pred-batch-size", type=int, default=1, help="推理 batch size（默认 1）。")
    parser.add_argument("--pred-use-ema", action="store_true", help="若 ckpt 包含 ema 权重则优先使用。")
    parser.add_argument(
        "--pred-weights-only",
        action="store_true",
        help="torch.load(..., weights_only=True) 加载 ckpt（更安全，但部分 ckpt 可能不兼容）。",
    )
    parser.add_argument("--limit", type=int, default=0, help="最多生成多少张（0=不限制）。")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="输出目录（默认 <run-dir>/selection/panel）。",
    )
    return parser.parse_args()


def _load_run_meta(meta_json: Path) -> dict[str, Any]:
    obj = json.loads(meta_json.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise TypeError(f"meta.json 不是 dict: {meta_json}")
    return obj


def _load_run_index(meta_json: Path) -> list[RunIndexItem]:
    meta = _load_run_meta(meta_json)
    files = meta.get("files", []) or []
    items: list[RunIndexItem] = []
    for it in files:
        if not isinstance(it, dict):
            continue
        image = str(it.get("image", "")).strip()
        msi = str(it.get("msi", "")).strip()
        if not image:
            continue
        items.append(RunIndexItem(image=image, msi=msi))
    if not items:
        raise ValueError(f"meta.json 未包含有效 files: {meta_json}")
    return items


def _resolve_ann_path(run_meta: dict[str, Any], *, run_dir: Path, ann_arg: str, split_arg: str) -> Path:
    if str(ann_arg).strip():
        raw = Path(ann_arg).expanduser()
        tried: list[Path] = []
        if raw.is_absolute():
            tried.append(raw)
            if raw.is_file():
                return raw
            raise FileNotFoundError(f"--ann 不存在: {raw}")
        cand_cwd = (Path.cwd() / raw).resolve()
        tried.append(cand_cwd)
        if cand_cwd.is_file():
            return cand_cwd
        cand_repo = (REPO_ROOT / raw).resolve()
        tried.append(cand_repo)
        if cand_repo.is_file():
            return cand_repo
        cand_run = (run_dir / raw).resolve()
        tried.append(cand_run)
        if cand_run.is_file():
            return cand_run
        raise FileNotFoundError(f"--ann 不存在，尝试过：{' | '.join(str(p) for p in tried)}")

    config_path = str(run_meta.get("config", "")).strip()
    if not config_path:
        raise ValueError("meta.json 未记录 config 路径，无法自动推导 ann；请显式传 --ann。")
    cfg_file = Path(config_path).expanduser()
    if not cfg_file.is_absolute():
        cfg_file2 = (REPO_ROOT / cfg_file).resolve()
        cfg_file = cfg_file2 if cfg_file2.is_file() else cfg_file
    if not cfg_file.is_file():
        raise FileNotFoundError(f"无法读取 config 文件以推导 ann：{cfg_file}")

    cfg = OmegaConf.load(str(cfg_file))
    data_cfg = cfg.get("data", {}) or {}
    dataset_dir = data_cfg.get("dataset_dir", None)
    if not dataset_dir:
        raise ValueError(f"config 中缺少 data.dataset_dir，无法推导 ann：{cfg_file}")
    dataset_root = Path(str(dataset_dir)).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = (REPO_ROOT / dataset_root).resolve()

    split = str(split_arg).strip() or str(run_meta.get("split", "")).strip() or "val"
    ann = dataset_root / "annotations" / f"{split}.json"
    if not ann.is_file():
        raise FileNotFoundError(f"自动推导的 ann 不存在: {ann}；请显式传 --ann。")
    return ann


def _load_coco(ann_path: Path) -> dict[str, Any]:
    obj = json.loads(ann_path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise TypeError(f"COCO ann 不是 dict: {ann_path}")
    return obj


def _index_coco(coco: dict[str, Any]) -> tuple[dict[str, int], dict[int, list[dict[str, Any]]], dict[int, str]]:
    images = coco.get("images", []) or []
    anns = coco.get("annotations", []) or []
    cats = coco.get("categories", []) or []

    stem_to_img_id: dict[str, int] = {}
    for im in images:
        if not isinstance(im, dict):
            continue
        file_name = str(im.get("file_name", "")).strip()
        if not file_name:
            continue
        stem = Path(file_name).stem
        if stem in stem_to_img_id:
            continue
        try:
            stem_to_img_id[stem] = int(im.get("id"))
        except Exception:
            continue

    anns_by_img: dict[int, list[dict[str, Any]]] = {}
    for ann in anns:
        if not isinstance(ann, dict):
            continue
        try:
            img_id = int(ann.get("image_id"))
        except Exception:
            continue
        anns_by_img.setdefault(img_id, []).append(ann)

    cat_id_to_name: dict[int, str] = {}
    for cat in cats:
        if not isinstance(cat, dict):
            continue
        try:
            cid = int(cat.get("id"))
        except Exception:
            continue
        cat_id_to_name[cid] = str(cat.get("name", "")).strip() or str(cid)

    return stem_to_img_id, anns_by_img, cat_id_to_name


def _oil_cat_ids(cat_id_to_name: dict[int, str], oil_names: Iterable[str]) -> set[int]:
    names = {str(n).strip().lower() for n in oil_names if str(n).strip()}
    out = {cid for cid, name in cat_id_to_name.items() if str(name).strip().lower() in names}
    if not out:
        raise ValueError(f"未找到 oil 类别：oil_names={sorted(names)} categories={sorted(set(cat_id_to_name.values()))[:20]}...")
    return out


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if size <= 0:
        return ImageFont.load_default()
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def _draw_boxes(
    img: Image.Image,
    *,
    anns: list[dict[str, Any]],
    cat_id_to_name: dict[int, str],
    oil_cat_ids: set[int],
    show: str,
    box_width: int,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size

    for ann in anns:
        if not isinstance(ann, dict):
            continue
        if int(ann.get("iscrowd", 0) or 0) != 0:
            continue
        try:
            cid = int(ann.get("category_id"))
        except Exception:
            continue
        if show == "oil" and cid not in oil_cat_ids:
            continue
        bbox = ann.get("bbox", None)
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            continue
        x, y, bw, bh = [float(v) for v in bbox]
        if bw <= 0 or bh <= 0:
            continue
        x1 = max(0.0, min(float(w), x))
        y1 = max(0.0, min(float(h), y))
        x2 = max(0.0, min(float(w), x + bw))
        y2 = max(0.0, min(float(h), y + bh))
        if x2 <= x1 or y2 <= y1:
            continue

        is_oil = cid in oil_cat_ids
        color = (255, 0, 0) if is_oil else (0, 255, 0)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=max(1, int(box_width)))

        name = cat_id_to_name.get(cid, str(cid))
        label = f"{name}"
        try:
            tbox = draw.textbbox((0, 0), label, font=font)
            tw, th = tbox[2] - tbox[0], tbox[3] - tbox[1]
        except Exception:
            tw, th = (len(label) * 6, 11)

        tx = int(x1)
        ty = int(max(0, y1 - th - 4))
        bg = [tx, ty, tx + tw + 6, ty + th + 4]
        draw.rectangle(bg, fill=(0, 0, 0))
        draw.text((tx + 3, ty + 2), label, fill=(255, 255, 255), font=font)

    return out


def _parse_rgb_color(spec: str) -> tuple[int, int, int]:
    s = str(spec).strip()
    if not s:
        return (0, 255, 255)
    if s.startswith("#") and len(s) in {7, 9}:
        try:
            r = int(s[1:3], 16)
            g = int(s[3:5], 16)
            b = int(s[5:7], 16)
            return (r, g, b)
        except Exception:
            return (0, 255, 255)
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        return (0, 255, 255)
    try:
        r, g, b = [int(float(p)) for p in parts]
    except Exception:
        return (0, 255, 255)
    return (int(np.clip(r, 0, 255)), int(np.clip(g, 0, 255)), int(np.clip(b, 0, 255)))


def _draw_pred_boxes(
    img: Image.Image,
    *,
    preds: list[dict[str, Any]],
    box_width: int,
    color: tuple[int, int, int],
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    show_score: bool,
) -> Image.Image:
    if not preds:
        return img
    out = img.copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size

    for pred in preds:
        bbox = pred.get("bbox_xyxy", None)
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            continue
        x1, y1, x2, y2 = [float(v) for v in bbox]
        x1 = max(0.0, min(float(w), x1))
        y1 = max(0.0, min(float(h), y1))
        x2 = max(0.0, min(float(w), x2))
        y2 = max(0.0, min(float(h), y2))
        if x2 <= x1 or y2 <= y1:
            continue

        draw.rectangle([x1, y1, x2, y2], outline=tuple(color), width=max(1, int(box_width)))

        if not show_score:
            continue
        score = pred.get("score", None)
        label = pred.get("label_name", None) or pred.get("label", None)
        if score is None:
            continue
        try:
            s = float(score)
        except Exception:
            continue
        label_str = str(label) if label is not None else "pred"
        text = f"P {label_str} {s:.2f}"
        try:
            tbox = draw.textbbox((0, 0), text, font=font)
            tw, th = tbox[2] - tbox[0], tbox[3] - tbox[1]
        except Exception:
            tw, th = (len(text) * 6, 11)

        tx = int(x1)
        ty = int(max(0, y1 - th - 4))
        bg = [tx, ty, tx + tw + 6, ty + th + 4]
        draw.rectangle(bg, fill=(0, 0, 0))
        draw.text((tx + 3, ty + 2), text, fill=(255, 255, 255), font=font)

    return out


def _pick_device(device: str):
    device = str(device).strip().lower()
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("draw-preds 需要 torch。") from exc
    if device.startswith("cuda") and not torch.cuda.is_available():
        logging.warning("请求使用 CUDA，但当前不可用，自动回退到 CPU。")
        return torch.device("cpu")
    return torch.device(device)


def _normalize_class_names(names: Any) -> list[str]:
    if not names:
        return []
    try:
        from omegaconf import ListConfig  # type: ignore
    except Exception:  # pragma: no cover
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
        if s.lower() in {"background", "_background_", "bg", "no_object", "no-object"}:
            continue
        normalized.append(s)
    return normalized


def _run_rtmsfdetr_predictions(
    items: list[RunIndexItem],
    *,
    run_meta: dict[str, Any],
    run_dir: Path,
    oil_names: list[str],
    pred_show: str,
    pred_device: str,
    pred_amp: bool,
    pred_score_thr: float,
    pred_max_dets: int,
    pred_batch_size: int,
    pred_use_ema: bool,
    pred_weights_only: bool,
) -> dict[str, list[dict[str, Any]]]:
    import torch
    import torch.nn.functional as F_torch
    from torchvision.transforms import functional as tvf

    from utils.misc import NestedTensor

    def get_config(node: Any, key: str, default: Any | None = None) -> Any:
        if node is None:
            return default
        if hasattr(node, "get"):
            return node.get(key, default)
        return getattr(node, key, default)

    def _rgb_normalize(
        x: torch.Tensor, *, mode: str, rgb_mean: tuple[float, float, float], rgb_std: tuple[float, float, float]
    ) -> torch.Tensor:
        mode = str(mode or "imagenet").lower()
        if mode == "imagenet":
            return tvf.normalize(x, mean=list(rgb_mean), std=list(rgb_std))
        if mode == "linear":
            return x
        if mode == "image_max":
            denom = float(x.amax().clamp_min(1e-6))
            return x / denom
        if mode == "per_channel_minmax":
            mins = x.amin(dim=(1, 2), keepdim=True)
            maxs = x.amax(dim=(1, 2), keepdim=True)
            denom = (maxs - mins).clamp_min(1e-6)
            return (x - mins) / denom
        raise ValueError(f"未知的 rgb_normalize_mode: {mode}")

    def _load_msi_as_tensor(msi_path: Path, *, expected_channels: int | None) -> torch.Tensor:
        if not msi_path.is_file():
            raise FileNotFoundError(f"未找到多光谱文件：{msi_path}")

        try:
            import tifffile  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("需要 tifffile 才能读取多光谱 TIF。") from exc

        suffix = msi_path.suffix.lower()
        if suffix in {".tif", ".tiff"}:
            array = tifffile.imread(str(msi_path))
        else:
            with Image.open(msi_path) as img:
                img.load()
                array = np.array(img)

        if array.ndim == 2:
            array = array[..., None]
        if array.ndim != 3:
            raise ValueError(f"多光谱图像形状异常（需要 3 维）：{msi_path}, shape={array.shape}")

        dim0, dim1, dim2 = array.shape
        is_chw = dim0 <= 32 and dim0 < dim1 and dim0 < dim2
        is_hwc = dim2 <= 32 and dim2 < dim0 and dim2 < dim1
        if is_chw and is_hwc and expected_channels is not None:
            exp = int(expected_channels)
            if dim0 == exp and dim2 != exp:
                is_hwc = False
            elif dim2 == exp and dim0 != exp:
                is_chw = False

        if is_chw and not is_hwc:
            array_hwc = np.transpose(array, (1, 2, 0))
        else:
            array_hwc = array

        if expected_channels is not None:
            exp = int(expected_channels)
            c = int(array_hwc.shape[2])
            if c == exp:
                pass
            elif exp == 1:
                if c > 1:
                    array_hwc = array_hwc.mean(axis=2, keepdims=True)
            elif c == 1 and exp > 1:
                array_hwc = np.repeat(array_hwc, repeats=exp, axis=2)
            elif c > exp:
                array_hwc = array_hwc[..., :exp]
            else:
                raise ValueError(f"多光谱通道数不足：{msi_path}, channels={c}, expected={exp}")

        tensor = torch.from_numpy(np.transpose(array_hwc, (2, 0, 1)).astype(np.float32))
        return tensor

    def _normalize_ms_tensor(ms: torch.Tensor, *, mode: str, scale_value: float | None) -> torch.Tensor:
        mode = str(mode or "per_channel_minmax").lower()
        if mode == "none":
            return ms
        if mode == "linear":
            if scale_value is None:
                raise ValueError("ms_normalize_mode=linear 需要 scale_value（data.ms_fixed_scale）。")
            return ms / float(scale_value)
        if mode == "per_channel_minmax":
            mins = ms.amin(dim=(1, 2), keepdim=True)
            maxs = ms.amax(dim=(1, 2), keepdim=True)
            denom = (maxs - mins).clamp_min(1e-6)
            return (ms - mins) / denom
        if mode == "tensor_minmax":
            lo = ms.amin()
            hi = ms.amax()
            denom = (hi - lo).clamp_min(1e-6)
            return (ms - lo) / denom
        if mode == "image_max":
            denom = ms.amax().clamp_min(1e-6)
            return ms / denom
        if mode == "fixed_scale":
            if scale_value is None:
                raise ValueError("ms_normalize_mode=fixed_scale 需要 scale_value（data.ms_fixed_scale）。")
            return torch.clamp(ms / float(scale_value), 0.0, 1.0)
        raise ValueError(f"未知的 ms_normalize_mode: {mode}")

    def _resize_ms_tensor(ms: torch.Tensor, *, size_hw: tuple[int, int]) -> torch.Tensor:
        ms = ms.unsqueeze(0)
        out = F_torch.interpolate(ms, size=size_hw, mode="bilinear", align_corners=False)
        return out.squeeze(0)

    config_path = str(run_meta.get("config", "")).strip()
    ckpt_path = str(run_meta.get("checkpoint", "")).strip()
    if not config_path or not ckpt_path:
        raise ValueError("meta.json 缺少 config/checkpoint，无法 draw-preds。")

    cfg_file = Path(config_path).expanduser()
    if not cfg_file.is_absolute():
        cfg_repo = (REPO_ROOT / cfg_file).resolve()
        cfg_file = cfg_repo if cfg_repo.is_file() else cfg_file
    if not cfg_file.is_file():
        cfg_run = (run_dir / cfg_file).resolve()
        if cfg_run.is_file():
            cfg_file = cfg_run
    if not cfg_file.is_file():
        raise FileNotFoundError(f"无法读取 config：{cfg_file}")

    ckpt_file = Path(ckpt_path).expanduser()
    if not ckpt_file.is_absolute():
        ckpt_repo = (REPO_ROOT / ckpt_file).resolve()
        ckpt_file = ckpt_repo if ckpt_repo.is_file() else ckpt_file
    if not ckpt_file.is_file():
        ckpt_run = (run_dir / ckpt_file).resolve()
        if ckpt_run.is_file():
            ckpt_file = ckpt_run
    if not ckpt_file.is_file():
        raise FileNotFoundError(f"无法读取 checkpoint：{ckpt_file}")

    cfg = OmegaConf.load(str(cfg_file))
    OmegaConf.set_struct(cfg, False)

    data_cfg = getattr(cfg, "data", None)
    model_cfg = getattr(cfg, "model", None) or {}
    train_cfg = getattr(cfg, "train", None) or {}

    use_rgb = bool(get_config(data_cfg, "use_rgb_input", True))
    use_msi = bool(get_config(data_cfg, "use_msi_input", False))

    rgb_ch = int(get_config(data_cfg, "rgb_input_channels", 3) or 3)
    ms_ch = int(get_config(data_cfg, "ms_input_channels", 0) or 0)
    expected_in_ch = (rgb_ch if use_rgb else 0) + (ms_ch if use_msi else 0)
    if expected_in_ch <= 0:
        raise ValueError("use_rgb_input/use_msi_input 均为 False，无法推理。")
    if use_msi and ms_ch <= 0:
        raise ValueError("use_msi_input=True 但 ms_input_channels<=0，无法推理。")

    rgb_mean = tuple(get_config(data_cfg, "rgb_mean", (0.485, 0.456, 0.406)))
    rgb_std = tuple(get_config(data_cfg, "rgb_std", (0.229, 0.224, 0.225)))
    rgb_mode = str(get_config(data_cfg, "rgb_normalize_mode", "imagenet") or "imagenet")
    ms_mode = str(get_config(data_cfg, "ms_normalize_mode", "per_channel_minmax") or "per_channel_minmax")
    ms_fixed_scale = get_config(data_cfg, "ms_fixed_scale", None)
    ms_center_to_rgb_range = bool(get_config(data_cfg, "ms_center_to_rgb_range", False))
    ms_suffix = str(get_config(data_cfg, "ms_suffix", ".tif") or ".tif")

    img_size = int(
        get_config(data_cfg, "img_size", None)
        or get_config(train_cfg, "img_size", None)
        or int(run_meta.get("img_size", 0) or 0)
        or 640
    )

    device = _pick_device(pred_device)
    amp_enabled = bool(pred_amp) and device.type == "cuda"

    from argparse import Namespace
    from engines.models.rtmsfdetr.builder import build_model_and_processors

    num_classes = int(get_config(train_cfg, "num_classes", get_config(model_cfg, "num_classes", 1)) or 1)
    build_args = Namespace(
        device=str(device),
        num_classes=int(num_classes),
        img_size=int(img_size),
        rtdetrv4_config=get_config(model_cfg, "rtdetrv4_config", None),
        disable_distill=bool(get_config(model_cfg, "disable_distill", True)),
        teacher_repo_path=get_config(model_cfg, "teacher_repo_path", None),
        teacher_weights_path=get_config(model_cfg, "teacher_weights_path", None),
        hgnet_pretrained=bool(get_config(model_cfg, "hgnet_pretrained", False)),
        hgnet_local_model_dir=get_config(model_cfg, "hgnet_local_model_dir", None),
        hgnet_freeze_at=int(get_config(model_cfg, "hgnet_freeze_at", -1)),
        hgnet_freeze_norm=bool(get_config(model_cfg, "hgnet_freeze_norm", False)),
        input_denormalize=bool(get_config(model_cfg, "input_denormalize", True)),
        clamp_after_denormalize=bool(get_config(model_cfg, "clamp_after_denormalize", True)),
        rgb_mean=tuple(rgb_mean),
        rgb_std=tuple(rgb_std),
        rgb_input_channels=int(rgb_ch if use_rgb else 0),
        ms_input_channels=int(ms_ch if use_msi else 0),
        input_channels=int(expected_in_ch),
        dual_stream_backbone=bool(get_config(model_cfg, "dual_stream_backbone", False)),
        backbone_output_merge=str(get_config(model_cfg, "backbone_output_merge", "avg") or "avg"),
        backbone_fusion=get_config(model_cfg, "backbone_fusion", None),
    )
    model, _, postprocessor = build_model_and_processors(build_args)

    checkpoint = torch.load(ckpt_file, map_location="cpu", weights_only=bool(pred_weights_only))
    model_state = checkpoint
    if pred_use_ema and isinstance(checkpoint, dict) and checkpoint.get("ema") is not None:
        model_state = checkpoint["ema"]
    state_dict = model_state.get("model", model_state) if isinstance(model_state, dict) else model_state
    if hasattr(state_dict, "state_dict"):
        state_dict = state_dict.state_dict()
    if not isinstance(state_dict, dict):
        raise TypeError(f"无法从 checkpoint 解析 state_dict，得到类型={type(state_dict)}")

    # filter incompatible keys to avoid shape mismatch
    model_sd = model.state_dict()
    compatible = {k: v for k, v in state_dict.items() if k in model_sd and getattr(v, "shape", None) == model_sd[k].shape}
    skipped = len(state_dict) - len(compatible)
    if skipped:
        logging.warning("checkpoint 有 %d 个参数 shape 不匹配或不存在，已跳过加载。", skipped)
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    if missing or unexpected:
        logging.info("加载 state_dict：missing=%d unexpected=%d", len(missing), len(unexpected))

    model.eval()
    model.to(device)
    postprocessor.eval()
    postprocessor.to(device)

    # label -> name (best-effort)
    names_all = list(get_config(data_cfg, "class_names", None) or [])
    names_norm = _normalize_class_names(names_all)
    if len(names_norm) == num_classes:
        label_names = [str(x).strip() for x in names_norm]
    elif len(names_all) == num_classes:
        label_names = [str(x).strip() for x in names_all]
    else:
        label_names = []

    oil_lower = {str(n).strip().lower() for n in oil_names if str(n).strip()}
    oil_label_ids: set[int] = set()
    if label_names:
        for i, name in enumerate(label_names):
            if str(name).strip().lower() in oil_lower:
                oil_label_ids.add(int(i))

    preds_by_stem: dict[str, list[dict[str, Any]]] = {}
    batch_size = max(1, int(pred_batch_size))
    score_thr = float(pred_score_thr)
    max_dets = max(1, int(pred_max_dets))

    for i0 in range(0, len(items), batch_size):
        batch = items[i0 : i0 + batch_size]

        rgbs: list[Path] = []
        orig_sizes_hw: list[tuple[int, int]] = []
        tensors: list[torch.Tensor] = []

        for it in batch:
            rgb_p = Path(it.image).expanduser()
            if not rgb_p.is_file():
                logging.warning("pred: 跳过缺失原图: %s", rgb_p)
                continue
            with Image.open(rgb_p) as _im:
                rgb_img = _im.convert("RGB")
            orig_w, orig_h = rgb_img.size

            rgb_resized = rgb_img.resize((int(img_size), int(img_size)), Image.BILINEAR) if img_size > 0 else rgb_img
            modalities: list[torch.Tensor] = []
            if use_rgb:
                x = tvf.to_tensor(rgb_resized)
                x = _rgb_normalize(x, mode=rgb_mode, rgb_mean=rgb_mean, rgb_std=rgb_std)
                modalities.append(x)

            msi_p: Path | None = None
            if use_msi:
                raw = str(it.msi).strip()
                if raw:
                    cand = Path(raw).expanduser()
                    if not cand.is_absolute():
                        cand_repo = (REPO_ROOT / cand).resolve()
                        cand = cand_repo if cand_repo.is_file() else cand
                    msi_p = cand if cand.is_file() else None
                if msi_p is None:
                    msi_dir_raw = str(run_meta.get("msi_dir", "")).strip()
                    if msi_dir_raw:
                        msi_dir = Path(msi_dir_raw).expanduser()
                        if not msi_dir.is_absolute():
                            msi_dir = (REPO_ROOT / msi_dir).resolve()
                        cand = (msi_dir / f"{rgb_p.stem}{ms_suffix}").expanduser()
                        msi_p = cand if cand.is_file() else None
                if msi_p is None:
                    logging.warning("pred: 缺失 msi 配对文件，跳过: rgb=%s", rgb_p)
                    continue
                ms = _load_msi_as_tensor(msi_p, expected_channels=ms_ch)
                if ms.shape[1:] != (orig_h, orig_w):
                    ms = _resize_ms_tensor(ms, size_hw=(int(orig_h), int(orig_w)))
                if img_size > 0:
                    ms = _resize_ms_tensor(ms, size_hw=(int(img_size), int(img_size)))
                else:
                    ms = _resize_ms_tensor(ms, size_hw=(int(rgb_resized.size[1]), int(rgb_resized.size[0])))
                ms = _normalize_ms_tensor(ms, mode=ms_mode, scale_value=float(ms_fixed_scale) if ms_fixed_scale is not None else None)
                if ms_center_to_rgb_range and ms_mode.lower() in {
                    "per_channel_minmax",
                    "tensor_minmax",
                    "image_max",
                    "fixed_scale",
                }:
                    ms = (ms - 0.5) / 0.5
                modalities.append(ms)

            if not modalities:
                continue
            x_in = torch.cat(modalities, dim=0)
            tensors.append(x_in)
            rgbs.append(rgb_p)
            orig_sizes_hw.append((int(orig_h), int(orig_w)))

        if not tensors:
            continue
        batch_tensor = torch.stack(tensors, dim=0)
        mask = torch.zeros((batch_tensor.shape[0], batch_tensor.shape[2], batch_tensor.shape[3]), dtype=torch.bool)
        samples = NestedTensor(batch_tensor.to(device), mask.to(device))
        orig_target_sizes = torch.tensor(orig_sizes_hw, dtype=torch.int64, device=device)

        with torch.inference_mode():
            if amp_enabled:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(samples)
                    results = postprocessor(outputs, orig_target_sizes=orig_target_sizes)
            else:
                outputs = model(samples)
                results = postprocessor(outputs, orig_target_sizes=orig_target_sizes)

        for rgb_p, res, (orig_h, orig_w) in zip(rgbs, results, orig_sizes_hw[: len(results)]):
            stem = rgb_p.stem
            labels = res.get("labels")
            boxes = res.get("boxes")
            scores = res.get("scores")
            if labels is None or boxes is None or scores is None:
                preds_by_stem[stem] = []
                continue

            lab = labels.detach().to("cpu").tolist()
            box = boxes.detach().to("cpu").tolist()
            sco = scores.detach().to("cpu").tolist()

            rows: list[dict[str, Any]] = []
            for l, b, s in zip(lab, box, sco):
                try:
                    score = float(s)
                except Exception:
                    continue
                if score < score_thr:
                    continue
                try:
                    label_i = int(l)
                except Exception:
                    label_i = -1
                if pred_show == "oil" and oil_label_ids and label_i not in oil_label_ids:
                    continue
                if not (isinstance(b, (list, tuple)) and len(b) == 4):
                    continue
                x1, y1, x2, y2 = [float(v) for v in b]
                x1 = max(0.0, min(float(orig_w), x1))
                y1 = max(0.0, min(float(orig_h), y1))
                x2 = max(0.0, min(float(orig_w), x2))
                y2 = max(0.0, min(float(orig_h), y2))
                if x2 <= x1 or y2 <= y1:
                    continue
                label_name = label_names[label_i] if (0 <= label_i < len(label_names)) else str(label_i)
                rows.append(
                    {
                        "bbox_xyxy": [x1, y1, x2, y2],
                        "score": score,
                        "label": label_i,
                        "label_name": label_name,
                    }
                )

            rows.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)
            preds_by_stem[stem] = rows[:max_dets]

    if pred_show == "oil" and num_classes > 1 and not oil_label_ids:
        logging.warning(
            "pred-show=oil 但无法从 cfg.data.class_names 推导 oil label 索引，已回退为不过滤（绘制全部）。"
        )
    return preds_by_stem


def main() -> int:
    args = _parse_args()
    run_dir = Path(args.run_dir).expanduser()
    meta_json = run_dir / "meta.json"
    overlay_dir = run_dir / "selection" / "overlay_heatmap"
    if not meta_json.is_file():
        raise FileNotFoundError(f"缺少 meta.json: {meta_json}")
    if not overlay_dir.is_dir():
        raise FileNotFoundError(
            f"缺少 overlay_heatmap/: {overlay_dir}；请先运行 tools/repr_oil_heatmap.py 生成 heatmap。"
        )

    run_meta = _load_run_meta(meta_json)
    ann_path = _resolve_ann_path(run_meta, run_dir=run_dir, ann_arg=str(args.ann), split_arg=str(args.split))
    coco = _load_coco(ann_path)
    stem_to_img_id, anns_by_img, cat_id_to_name = _index_coco(coco)
    oil_cat_ids = _oil_cat_ids(cat_id_to_name, args.oil_names)

    items = _load_run_index(meta_json)
    if args.limit and int(args.limit) > 0:
        items = items[: int(args.limit)]

    out_dir = Path(args.output_dir).expanduser() if str(args.output_dir).strip() else (run_dir / "selection" / "panel")
    out_dir.mkdir(parents=True, exist_ok=True)

    font_size = int(args.font_size)
    box_width = max(1, int(args.box_width))
    pred_box_width = max(1, int(args.pred_box_width))
    pred_color = _parse_rgb_color(str(args.pred_color))
    pred_show_score = bool(args.pred_show_score)
    gap = max(0, int(args.gap))
    margin_top = max(0, int(args.margin_top))

    logging.info("run_dir=%s images=%d ann=%s", run_dir, len(items), ann_path)
    logging.info("out_dir=%s show=%s oil_names=%s", out_dir, str(args.show), list(args.oil_names))

    preds_by_stem: dict[str, list[dict[str, Any]]] = {}
    if bool(args.draw_preds):
        preds_by_stem = _run_rtmsfdetr_predictions(
            items,
            run_meta=run_meta,
            run_dir=run_dir,
            oil_names=list(args.oil_names),
            pred_show=str(args.pred_show),
            pred_device=str(args.pred_device),
            pred_amp=bool(args.pred_amp),
            pred_score_thr=float(args.pred_score_thr),
            pred_max_dets=int(args.pred_max_dets),
            pred_batch_size=int(args.pred_batch_size),
            pred_use_ema=bool(args.pred_use_ema),
            pred_weights_only=bool(args.pred_weights_only),
        )

    for item in items:
        image_path = Path(item.image).expanduser()
        if not image_path.is_file():
            logging.warning("跳过缺失原图: %s", image_path)
            continue
        stem = image_path.stem
        overlay_path = overlay_dir / f"{stem}.png"
        if not overlay_path.is_file():
            logging.warning("跳过缺失 overlay: %s", overlay_path)
            continue

        img = Image.open(image_path).convert("RGB")
        overlay = Image.open(overlay_path).convert("RGB")
        if overlay.size != img.size:
            overlay = overlay.resize(img.size, Image.BILINEAR)

        if font_size <= 0:
            font_size_eff = max(12, int(min(img.size) * 0.03))
        else:
            font_size_eff = font_size
        font = _load_font(font_size_eff)

        img_id = stem_to_img_id.get(stem)
        anns = anns_by_img.get(int(img_id), []) if img_id is not None else []

        left = _draw_boxes(
            img,
            anns=anns,
            cat_id_to_name=cat_id_to_name,
            oil_cat_ids=oil_cat_ids,
            show=str(args.show),
            box_width=box_width,
            font=font,
        )
        if bool(args.draw_preds):
            preds = preds_by_stem.get(stem, [])
            left = _draw_pred_boxes(
                left,
                preds=preds,
                box_width=pred_box_width,
                color=pred_color,
                font=font,
                show_score=pred_show_score,
            )
        right = overlay
        if bool(args.draw_boxes_on_right):
            right = _draw_boxes(
                right,
                anns=anns,
                cat_id_to_name=cat_id_to_name,
                oil_cat_ids=oil_cat_ids,
                show=str(args.show),
                box_width=box_width,
                font=font,
            )
        if bool(args.draw_preds):
            preds = preds_by_stem.get(stem, [])
            right = _draw_pred_boxes(
                right,
                preds=preds,
                box_width=pred_box_width,
                color=pred_color,
                font=font,
                show_score=pred_show_score,
            )

        w, h = img.size
        canvas_w = w * 2 + gap
        canvas_h = h + margin_top
        canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        canvas.paste(left, (0, margin_top))
        canvas.paste(right, (w + gap, margin_top))

        # titles
        tl = str(args.title_left)
        tr = str(args.title_right)
        draw.text((8, 8), tl, fill=(0, 0, 0), font=font)
        draw.text((w + gap + 8, 8), tr, fill=(0, 0, 0), font=font)

        out_path = out_dir / f"{stem}.png"
        canvas.save(out_path)

    logging.info("完成：%s", out_dir)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
