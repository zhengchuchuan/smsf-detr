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
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

"""
python tools/cluster_analysis/repr_auto_select_oil_clusters.py \
    --run-dir outputs/repr/rtmsfdetr/oil_rgb_val_k32x16_20260104 --ann data/oil_20260101/annotations/val.json \
    --metric iou --max-clusters 32
"""

@dataclass(frozen=True)
class ClusterStats:
    tp: int
    fp: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="基于 COCO oil 标注（bbox）自动选择 oil 对应的 cluster id，并写入 selected_clusters.json。"
    )
    parser.add_argument("--run-dir", type=str, required=True, help="repr run 目录（包含 meta.json/labels/clusters）。")
    parser.add_argument(
        "--ann",
        type=str,
        default="",
        help="COCO annotations json（默认自动：从 meta.json 记录的 config 推导 data.dataset_dir/annotations/<split>.json）。",
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
        "--metric",
        type=str,
        default="iou",
        choices=["iou", "f1"],
        help="选择 cluster 子集时的优化指标（默认 iou）。",
    )
    parser.add_argument("--max-clusters", type=int, default=16, help="最多选择多少个 oil cluster（默认 16）。")
    parser.add_argument(
        "--min-support",
        type=int,
        default=50,
        help="过滤掉总出现次数（tp+fp）小于该值的 cluster（单位：feat 网格像素，默认 50）。",
    )
    parser.add_argument(
        "--min-tp",
        type=int,
        default=10,
        help="过滤掉 tp 小于该值的 cluster（默认 10）。",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="输出 selected_clusters.json（默认 <run-dir>/selection/selected_clusters.json）。",
    )
    return parser.parse_args()


def _load_run_meta(meta_json: Path) -> dict[str, Any]:
    obj = json.loads(meta_json.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise TypeError(f"meta.json 不是 dict: {meta_json}")
    return obj


def _load_palette_size(palette_json: Path) -> int:
    obj = json.loads(palette_json.read_text(encoding="utf-8"))
    colors = obj.get("colors", {}) or {}
    if not isinstance(colors, dict):
        raise TypeError(f"palette.json 的 colors 必须是 dict，实际为: {type(colors)}")
    if not colors:
        raise ValueError(f"palette.json 的 colors 为空: {palette_json}")
    return max(int(k) for k in colors.keys()) + 1


def _resolve_ann_path(run_meta: dict[str, Any], *, run_dir: Path, ann_arg: str, split_arg: str) -> Path:
    if str(ann_arg).strip():
        raw = Path(ann_arg).expanduser()
        tried: list[Path] = []
        if raw.is_absolute():
            tried.append(raw)
            if raw.is_file():
                return raw
            raise FileNotFoundError(f"--ann 不存在: {raw}")

        # 优先按当前工作目录解析（用户通常在 repo root 执行）
        cand_cwd = (Path.cwd() / raw).resolve()
        tried.append(cand_cwd)
        if cand_cwd.is_file():
            return cand_cwd

        # 其次按 repo root 解析
        cand_repo = (REPO_ROOT / raw).resolve()
        tried.append(cand_repo)
        if cand_repo.is_file():
            return cand_repo

        # 最后按 run_dir 解析（兼容把 ann 放到 run_dir 下的用法）
        cand_run = (run_dir / raw).resolve()
        tried.append(cand_run)
        if cand_run.is_file():
            return cand_run

        tried_str = " | ".join(str(p) for p in tried)
        raise FileNotFoundError(f"--ann 不存在，尝试过：{tried_str}")

    config_path = str(run_meta.get("config", "")).strip()
    if not config_path:
        raise ValueError("meta.json 未记录 config 路径，无法自动推导 ann；请显式传 --ann。")
    cfg_file = Path(config_path).expanduser()
    if not cfg_file.is_absolute():
        cfg_file = (run_dir / cfg_file).resolve()
    if not cfg_file.is_file():
        # 允许 config 是相对 repo root 的路径
        cfg_file2 = (REPO_ROOT / config_path).resolve()
        if cfg_file2.is_file():
            cfg_file = cfg_file2
        else:
            raise FileNotFoundError(f"无法读取 config 文件以推导 ann：{cfg_file} / {cfg_file2}")

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


def _load_coco_ann(ann_path: Path) -> dict[str, Any]:
    obj = json.loads(ann_path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise TypeError(f"COCO ann 不是 dict: {ann_path}")
    return obj


def _index_coco_by_stem(coco: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    images = coco.get("images", []) or []
    anns = coco.get("annotations", []) or []

    by_stem: dict[str, dict[str, Any]] = {}
    for im in images:
        if not isinstance(im, dict):
            continue
        file_name = str(im.get("file_name", "")).strip()
        if not file_name:
            continue
        stem = Path(file_name).stem
        # 若重复，保留第一条（数据集一般不会重复）
        by_stem.setdefault(stem, im)

    by_img: dict[int, list[dict[str, Any]]] = {}
    for ann in anns:
        if not isinstance(ann, dict):
            continue
        img_id = ann.get("image_id", None)
        if img_id is None:
            continue
        try:
            img_id_int = int(img_id)
        except Exception:
            continue
        by_img.setdefault(img_id_int, []).append(ann)

    return by_stem, by_img


def _oil_category_ids(coco: dict[str, Any], oil_names: Iterable[str]) -> set[int]:
    names = {str(n).strip().lower() for n in oil_names if str(n).strip()}
    cats = coco.get("categories", []) or []
    out: set[int] = set()
    for cat in cats:
        if not isinstance(cat, dict):
            continue
        name = str(cat.get("name", "")).strip().lower()
        if name in names:
            try:
                out.add(int(cat.get("id")))
            except Exception:
                continue
    if not out:
        raise ValueError(f"未在 COCO categories 中找到 oil_names={sorted(names)} 对应的类别。")
    return out


def _bbox_to_feat_slices(
    bbox_xywh: list[float] | tuple[float, float, float, float],
    *,
    orig_wh: tuple[int, int],
    input_wh: tuple[int, int],
    feat_hw: tuple[int, int],
) -> tuple[slice, slice] | None:
    x, y, w, h = [float(v) for v in bbox_xywh]
    if w <= 0 or h <= 0:
        return None
    orig_w, orig_h = int(orig_wh[0]), int(orig_wh[1])
    if orig_w <= 0 or orig_h <= 0:
        return None
    in_w, in_h = int(input_wh[0]), int(input_wh[1])
    fh, fw = int(feat_hw[0]), int(feat_hw[1])
    if in_w <= 0 or in_h <= 0 or fw <= 0 or fh <= 0:
        return None

    sx = float(in_w) / float(orig_w)
    sy = float(in_h) / float(orig_h)
    x0 = x * sx
    y0 = y * sy
    x1 = (x + w) * sx
    y1 = (y + h) * sy

    # Map to feature grid (exclusive end)
    stride_w = float(in_w) / float(fw)
    stride_h = float(in_h) / float(fh)
    fx0 = int(np.floor(x0 / stride_w))
    fy0 = int(np.floor(y0 / stride_h))
    fx1 = int(np.ceil(x1 / stride_w))
    fy1 = int(np.ceil(y1 / stride_h))

    fx0 = int(np.clip(fx0, 0, fw))
    fx1 = int(np.clip(fx1, 0, fw))
    fy0 = int(np.clip(fy0, 0, fh))
    fy1 = int(np.clip(fy1, 0, fh))
    if fx1 <= fx0 or fy1 <= fy0:
        return None
    return slice(fy0, fy1), slice(fx0, fx1)


def _load_label_map(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim != 2:
        raise ValueError(f"label map 期望二维数组，实际 shape={arr.shape} path={path}")
    return arr


def _score(metric: str, *, tp: int, fp: int, fn: int) -> float:
    tp_f = float(tp)
    fp_f = float(fp)
    fn_f = float(fn)
    if metric == "iou":
        denom = tp_f + fp_f + fn_f
        return (tp_f / denom) if denom > 0 else 0.0
    if metric == "f1":
        denom = 2.0 * tp_f + fp_f + fn_f
        return (2.0 * tp_f / denom) if denom > 0 else 0.0
    raise ValueError(f"unsupported metric={metric}")


def _greedy_select(
    stats: list[ClusterStats],
    *,
    total_oil: int,
    metric: str,
    max_clusters: int,
    candidates: list[int],
) -> tuple[list[int], float, dict[str, int]]:
    selected: list[int] = []
    selected_set: set[int] = set()
    tp_sel = 0
    fp_sel = 0
    best = 0.0

    for _ in range(int(max_clusters)):
        best_i = None
        best_score = best
        best_tp = tp_sel
        best_fp = fp_sel

        for i in candidates:
            if i in selected_set:
                continue
            tp_new = tp_sel + int(stats[i].tp)
            fp_new = fp_sel + int(stats[i].fp)
            fn_new = int(total_oil - tp_new)
            if fn_new < 0:
                fn_new = 0
            s = _score(metric, tp=tp_new, fp=fp_new, fn=fn_new)
            if s > best_score + 1e-12:
                best_score = s
                best_i = i
                best_tp = tp_new
                best_fp = fp_new

        if best_i is None:
            break
        selected.append(int(best_i))
        selected_set.add(int(best_i))
        tp_sel = best_tp
        fp_sel = best_fp
        best = best_score

    fn_sel = int(total_oil - tp_sel)
    if fn_sel < 0:
        fn_sel = 0
    return selected, float(best), {"tp": int(tp_sel), "fp": int(fp_sel), "fn": int(fn_sel)}


def main() -> int:
    args = _parse_args()
    run_dir = Path(args.run_dir).expanduser()
    meta_json = run_dir / "meta.json"
    labels_dir = run_dir / "labels"
    palette_json = run_dir / "clusters" / "palette.json"
    if not meta_json.is_file():
        raise FileNotFoundError(f"缺少 meta.json: {meta_json}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"缺少 labels/: {labels_dir}")
    if not palette_json.is_file():
        raise FileNotFoundError(f"缺少 clusters/palette.json: {palette_json}")

    run_meta = _load_run_meta(meta_json)
    ann_path = _resolve_ann_path(run_meta, run_dir=run_dir, ann_arg=str(args.ann), split_arg=str(args.split))
    coco = _load_coco_ann(ann_path)

    k_total = _load_palette_size(palette_json)
    by_stem, by_img = _index_coco_by_stem(coco)
    oil_cat_ids = _oil_category_ids(coco, args.oil_names)

    # input size used in export (square)
    img_size = int(run_meta.get("img_size", 640) or 640)
    input_wh = (img_size, img_size)

    # Iterate label maps based on meta.json index (ensures same image set as features export)
    files = run_meta.get("files", []) or []
    if not isinstance(files, list):
        raise TypeError("meta.json 的 files 不是 list")

    stats = [ClusterStats(tp=0, fp=0) for _ in range(k_total)]
    total_oil = 0
    total_cells = 0

    matched = 0
    skipped = 0

    for it in files:
        if not isinstance(it, dict):
            continue
        image_path = Path(str(it.get("image", ""))).expanduser()
        if not image_path.is_file():
            skipped += 1
            continue
        stem = image_path.stem
        coco_im = by_stem.get(stem)
        if coco_im is None:
            skipped += 1
            continue

        img_id = int(coco_im.get("id"))
        orig_w = int(coco_im.get("width", 0) or 0)
        orig_h = int(coco_im.get("height", 0) or 0)
        if orig_w <= 0 or orig_h <= 0:
            # fallback: read image size
            try:
                from PIL import Image as _PILImage

                img = _PILImage.open(image_path)
                orig_w, orig_h = img.size
            except Exception:
                skipped += 1
                continue

        label_path = labels_dir / f"{stem}.png"
        if not label_path.is_file():
            skipped += 1
            continue
        label_map = _load_label_map(label_path)
        fh, fw = int(label_map.shape[0]), int(label_map.shape[1])

        oil_mask = np.zeros((fh, fw), dtype=bool)
        anns = by_img.get(img_id, []) or []
        for ann in anns:
            if not isinstance(ann, dict):
                continue
            if int(ann.get("iscrowd", 0) or 0) != 0:
                continue
            try:
                cat_id = int(ann.get("category_id"))
            except Exception:
                continue
            if cat_id not in oil_cat_ids:
                continue
            bbox = ann.get("bbox", None)
            if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
                continue
            slices = _bbox_to_feat_slices(
                bbox, orig_wh=(orig_w, orig_h), input_wh=input_wh, feat_hw=(fh, fw)
            )
            if slices is None:
                continue
            ys, xs = slices
            oil_mask[ys, xs] = True

        matched += 1
        total_oil += int(oil_mask.sum())
        total_cells += int(label_map.size)

        # bincount on feature grid pixels
        flat = label_map.reshape(-1).astype(np.int64)
        oil_flat = flat[oil_mask.reshape(-1)]
        non_flat = flat[~oil_mask.reshape(-1)]

        oil_bc = np.bincount(oil_flat, minlength=k_total)
        non_bc = np.bincount(non_flat, minlength=k_total)

        for cid in range(k_total):
            tp = int(oil_bc[cid])
            fp = int(non_bc[cid])
            if tp == 0 and fp == 0:
                continue
            prev = stats[cid]
            stats[cid] = ClusterStats(tp=prev.tp + tp, fp=prev.fp + fp)

    if matched == 0:
        raise RuntimeError("没有任何图片能匹配到 COCO annotations（按 stem 匹配）。请检查 --ann / 输入目录。")
    if total_oil <= 0:
        raise RuntimeError("在匹配到的图片中，没有任何 oil bbox（total_oil=0）。请检查 oil 类别名或 split。")

    # Candidate clusters filtering
    candidates: list[int] = []
    for cid, st in enumerate(stats):
        support = int(st.tp + st.fp)
        if support < int(args.min_support):
            continue
        if int(st.tp) < int(args.min_tp):
            continue
        candidates.append(int(cid))

    if not candidates:
        raise RuntimeError("候选 cluster 为空：请降低 --min-support/--min-tp，或检查聚类输出。")

    selected, best_score, counts = _greedy_select(
        stats,
        total_oil=int(total_oil),
        metric=str(args.metric),
        max_clusters=int(args.max_clusters),
        candidates=candidates,
    )

    out_path = Path(args.out).expanduser() if str(args.out).strip() else (run_dir / "selection" / "selected_clusters.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "oil_clusters": selected,
                "method": "auto_bbox_greedy",
                "metric": str(args.metric),
                "best_score": float(best_score),
                "counts": counts,
                "total_oil": int(total_oil),
                "k_total": int(k_total),
                "ann": str(ann_path),
                "oil_names": [str(x) for x in args.oil_names],
                "min_support": int(args.min_support),
                "min_tp": int(args.min_tp),
                "max_clusters": int(args.max_clusters),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Save ranking report
    report_path = out_path.parent / "cluster_scores.json"
    base_rate = float(total_oil) / float(total_cells) if total_cells > 0 else 0.0
    rows = []
    selected_set = set(selected)
    for cid, st in enumerate(stats):
        tp = int(st.tp)
        fp = int(st.fp)
        support = tp + fp
        prec = float(tp) / float(support) if support > 0 else 0.0
        cov = float(tp) / float(total_oil) if total_oil > 0 else 0.0
        lift = (prec / base_rate) if base_rate > 1e-12 else 0.0
        rows.append(
            {
                "cluster": int(cid),
                "tp": tp,
                "fp": fp,
                "support": int(support),
                "precision": prec,
                "coverage": cov,
                "lift": lift,
                "selected": bool(cid in selected_set),
            }
        )
    rows.sort(key=lambda r: (r["selected"], r["precision"], r["tp"]), reverse=True)
    report_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    logging.info(
        "done: matched=%d skipped=%d total_oil=%d selected=%d best_%s=%.4f -> %s",
        matched,
        skipped,
        int(total_oil),
        len(selected),
        str(args.metric),
        float(best_score),
        out_path,
    )
    logging.info("report: %s", report_path)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
