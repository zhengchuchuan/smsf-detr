from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# python tools/cluster_analysis/repr_apply_oil_clusters.py --run-dir outputs/repr/rtmsfdetr/oil_rgb_val_k32x16_20260104

@dataclass
class RunIndexItem:
    image: str
    npz: str | None = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="把已聚类的 label map 折叠为二值 oil vs non-oil mask。")
    parser.add_argument("--run-dir", type=str, required=True, help="repr run 目录（包含 meta.json / labels / clusters）。")
    parser.add_argument(
        "--selected",
        type=str,
        default="",
        help="selected_clusters.json 路径（默认 <run-dir>/selection/selected_clusters.json）。",
    )
    parser.add_argument(
        "--oil-clusters",
        nargs="*",
        default=None,
        help="直接指定 oil cluster id 列表（优先级高于 --selected）。",
    )
    parser.add_argument("--alpha", type=float, default=0.45, help="mask overlay 透明度（默认 0.45）。")
    return parser.parse_args()


def _load_run_index(meta_json: Path) -> list[RunIndexItem]:
    meta = json.loads(meta_json.read_text(encoding="utf-8"))
    files = meta.get("files", []) or []
    items: list[RunIndexItem] = []
    for it in files:
        if not isinstance(it, dict):
            continue
        image = str(it.get("image", "")).strip()
        npz = str(it.get("npz", "")).strip() or None
        if not image:
            continue
        items.append(RunIndexItem(image=image, npz=npz))
    if not items:
        raise ValueError(f"meta.json 未包含有效 files: {meta_json}")
    return items


def _load_label_map(path: Path) -> np.ndarray:
    im = Image.open(path)
    arr = np.array(im)
    if arr.ndim != 2:
        raise ValueError(f"label map 期望二维数组，实际 shape={arr.shape} path={path}")
    return arr


def _resize_nearest(label_hw: np.ndarray, *, size_wh: tuple[int, int]) -> np.ndarray:
    w, h = size_wh
    pil = Image.fromarray(label_hw)
    up = pil.resize((int(w), int(h)), resample=Image.NEAREST)
    return np.array(up)


def _blend(a: np.ndarray, b: np.ndarray, *, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    out = (1.0 - alpha) * a.astype(np.float32) + alpha * b.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_mask_rgb(rgb: np.ndarray, mask: np.ndarray, *, alpha: float) -> np.ndarray:
    if mask.dtype != np.bool_:
        mask = mask.astype(bool)
    color = np.zeros_like(rgb)
    color[..., 0] = 255
    blended = _blend(rgb, color, alpha=alpha)
    out = rgb.copy()
    out[mask] = blended[mask]
    return out


def _load_selected(path: Path) -> set[int]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    clusters = obj.get("oil_clusters", obj.get("clusters", obj.get("selected_clusters", []))) or []
    out: set[int] = set()
    for x in clusters:
        try:
            out.add(int(x))
        except Exception:
            continue
    return out


def main() -> int:
    args = _parse_args()
    run_dir = Path(args.run_dir).expanduser()
    meta_json = run_dir / "meta.json"
    labels_dir = run_dir / "labels"
    if not meta_json.is_file():
        raise FileNotFoundError(f"缺少 meta.json: {meta_json}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"缺少 labels/: {labels_dir}")

    if args.oil_clusters is not None and len(args.oil_clusters) > 0:
        oil_clusters = {int(x) for x in args.oil_clusters}
    else:
        selected_path = Path(args.selected).expanduser() if str(args.selected).strip() else (run_dir / "selection" / "selected_clusters.json")
        if not selected_path.is_file():
            raise FileNotFoundError(
                f"未找到 selected_clusters.json: {selected_path}；请先运行 tools/repr_select_oil_clusters.py 或传 --oil-clusters。"
            )
        oil_clusters = _load_selected(selected_path)

    if not oil_clusters:
        raise ValueError("oil_clusters 为空：请先选择 oil 对应的 cluster id。")

    items = _load_run_index(meta_json)
    out_mask = run_dir / "selection" / "pseudo_mask"
    out_overlay = run_dir / "selection" / "overlay"
    out_mask.mkdir(parents=True, exist_ok=True)
    out_overlay.mkdir(parents=True, exist_ok=True)

    logging.info("run_dir=%s", run_dir)
    logging.info("images=%d oil_clusters=%d", len(items), len(oil_clusters))

    for item in items:
        image_path = Path(item.image).expanduser()
        if not image_path.is_file():
            logging.warning("跳过缺失原图: %s", image_path)
            continue
        stem = image_path.stem
        label_path = labels_dir / f"{stem}.png"
        if not label_path.is_file():
            logging.warning("跳过缺失 label map: %s", label_path)
            continue

        img = Image.open(image_path).convert("RGB")
        rgb = np.array(img, dtype=np.uint8)
        label_map = _load_label_map(label_path)
        up = _resize_nearest(label_map, size_wh=img.size)

        mask = np.isin(up, np.array(sorted(oil_clusters), dtype=up.dtype))
        mask_u8 = (mask.astype(np.uint8) * 255)
        Image.fromarray(mask_u8, mode="L").save(out_mask / f"{stem}.png")

        overlay = _apply_mask_rgb(rgb, mask, alpha=float(args.alpha))
        Image.fromarray(overlay).save(out_overlay / f"{stem}.png")

    logging.info("完成：mask=%s overlay=%s", out_mask, out_overlay)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())

