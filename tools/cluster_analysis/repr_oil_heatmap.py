from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


#   python tools/cluster_analysis/repr_oil_heatmap.py --run-dir outputs/repr/rtmsfdetr/oil_rgb_val_k32x16_20260104 --score-key precision --normalize minmax --blur 5

@dataclass
class RunIndexItem:
    image: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将聚类结果转换为“oil 相关热力图”（cluster->oil score），并保存 heatmap/overlay。"
    )
    parser.add_argument("--run-dir", type=str, required=True, help="repr run 目录（包含 meta.json / labels / selection）。")
    parser.add_argument(
        "--scores",
        type=str,
        default="",
        help="cluster_scores.json 路径（默认 <run-dir>/selection/cluster_scores.json）。",
    )
    parser.add_argument(
        "--score-key",
        type=str,
        default="precision",
        choices=["precision", "coverage", "lift", "selected"],
        help="每个 cluster 的油相关分数来源（默认 precision）。",
    )
    parser.add_argument(
        "--normalize",
        type=str,
        default="minmax",
        choices=["none", "minmax", "clip01"],
        help="对 score map 的归一化方式（默认 minmax）。",
    )
    parser.add_argument("--alpha", type=float, default=0.45, help="overlay 透明度（默认 0.45）。")
    parser.add_argument(
        "--blur",
        type=int,
        default=0,
        help="可选：对热力图做高斯模糊的 kernel size（0=不模糊；建议 0 或 5/7/9）。",
    )
    parser.add_argument("--limit", type=int, default=0, help="最多处理多少张图（0=不限制）。")
    return parser.parse_args()


def _load_run_index(meta_json: Path) -> list[RunIndexItem]:
    meta = json.loads(meta_json.read_text(encoding="utf-8"))
    files = meta.get("files", []) or []
    items: list[RunIndexItem] = []
    for it in files:
        if not isinstance(it, dict):
            continue
        image = str(it.get("image", "")).strip()
        if not image:
            continue
        items.append(RunIndexItem(image=image))
    if not items:
        raise ValueError(f"meta.json 未包含有效 files: {meta_json}")
    return items


def _load_label_map(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path))
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


def _load_cluster_scores(scores_path: Path, *, score_key: str) -> dict[int, float]:
    rows = json.loads(scores_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise TypeError(f"cluster_scores.json 期望 list，实际为: {type(rows)}")
    out: dict[int, float] = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        cid = r.get("cluster", None)
        if cid is None:
            continue
        try:
            cid_i = int(cid)
        except Exception:
            continue
        if score_key == "selected":
            out[cid_i] = 1.0 if bool(r.get("selected", False)) else 0.0
        else:
            v = r.get(score_key, 0.0)
            try:
                out[cid_i] = float(v)
            except Exception:
                out[cid_i] = 0.0
    return out


def _normalize_map(x: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return x
    if mode == "clip01":
        return np.clip(x, 0.0, 1.0)
    if mode == "minmax":
        lo = float(np.nanmin(x))
        hi = float(np.nanmax(x))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-12:
            return np.zeros_like(x, dtype=np.float32)
        return ((x - lo) / (hi - lo)).astype(np.float32)
    raise ValueError(f"unsupported normalize={mode}")


def main() -> int:
    args = _parse_args()
    run_dir = Path(args.run_dir).expanduser()
    meta_json = run_dir / "meta.json"
    labels_dir = run_dir / "labels"
    if not meta_json.is_file():
        raise FileNotFoundError(f"缺少 meta.json: {meta_json}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"缺少 labels/: {labels_dir}")

    scores_path = Path(args.scores).expanduser() if str(args.scores).strip() else (run_dir / "selection" / "cluster_scores.json")
    if not scores_path.is_file():
        raise FileNotFoundError(
            f"缺少 cluster_scores.json: {scores_path}；请先运行 tools/repr_auto_select_oil_clusters.py。"
        )

    items = _load_run_index(meta_json)
    if args.limit and int(args.limit) > 0:
        items = items[: int(args.limit)]

    score_key = str(args.score_key)
    score_map = _load_cluster_scores(scores_path, score_key=score_key)

    out_heat = run_dir / "selection" / "heatmap"
    out_overlay = run_dir / "selection" / "overlay_heatmap"
    out_heat.mkdir(parents=True, exist_ok=True)
    out_overlay.mkdir(parents=True, exist_ok=True)

    blur_k = int(args.blur)
    if blur_k < 0 or blur_k % 2 == 0:
        blur_k = 0

    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("需要 cv2 才能生成 heatmap（applyColorMap/blur）。") from exc

    logging.info("run_dir=%s images=%d score_key=%s normalize=%s", run_dir, len(items), score_key, str(args.normalize))

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

        # cluster id -> score
        scores = np.zeros_like(up, dtype=np.float32)
        # vectorized mapping via unique ids
        uniq = np.unique(up)
        for cid in uniq.tolist():
            try:
                cid_i = int(cid)
            except Exception:
                continue
            scores[up == cid] = float(score_map.get(cid_i, 0.0))

        scores = _normalize_map(scores, mode=str(args.normalize)).astype(np.float32)
        if blur_k > 0:
            scores = cv2.GaussianBlur(scores, (blur_k, blur_k), sigmaX=0)

        heat_u8 = (np.clip(scores, 0.0, 1.0) * 255.0).astype(np.uint8)
        heat_bgr = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
        heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)

        overlay = _blend(rgb, heat_rgb, alpha=float(args.alpha))
        Image.fromarray(heat_rgb).save(out_heat / f"{stem}.png")
        Image.fromarray(overlay).save(out_overlay / f"{stem}.png")

    logging.info("完成：heatmap=%s overlay=%s", out_heat, out_overlay)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())

