from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

# Headless-friendly backend
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    meta_json: Path
    clusters_dir: Path
    selection_dir: Path
    labels_dir: Path
    features_dir: Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="对 repr 聚类结果做分析可视化：oil 分布/混淆、类间距离、训练 loss 曲线等。"
    )
    parser.add_argument("--run-dir", type=str, required=True, help="repr run 目录（包含 meta.json/labels/clusters/selection）。")
    parser.add_argument("--ann", type=str, default="", help="COCO annotations json（默认优先用 selected_clusters.json 里的 ann）。")
    parser.add_argument(
        "--metrics-csv",
        type=str,
        default="",
        help="训练 metrics.csv 路径（默认：从 meta.json 的 config 推导同目录 metrics.csv）。",
    )
    parser.add_argument("--out-dir", type=str, default="", help="输出目录（默认 <run-dir>/analysis）。")
    parser.add_argument("--top-n", type=int, default=30, help="Top-N 图表中显示多少个 cluster（默认 30）。")
    parser.add_argument("--limit", type=int, default=0, help="可选：只处理前 N 张图（0=不限制；用于加速）。")
    parser.add_argument("--recompute-centers", action="store_true", help="忽略缓存，强制重算 cluster centers 与距离。")
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_run_paths(run_dir: Path) -> RunPaths:
    run_dir = run_dir.expanduser().resolve()
    meta_json = run_dir / "meta.json"
    clusters_dir = run_dir / "clusters"
    selection_dir = run_dir / "selection"
    labels_dir = run_dir / "labels"
    features_dir = run_dir / "features"
    for p in [meta_json, clusters_dir, selection_dir, labels_dir, features_dir]:
        if not p.exists():
            raise FileNotFoundError(f"缺少必需路径: {p}")
    return RunPaths(
        run_dir=run_dir,
        meta_json=meta_json,
        clusters_dir=clusters_dir,
        selection_dir=selection_dir,
        labels_dir=labels_dir,
        features_dir=features_dir,
    )


def _palette_k_total(palette_json: Path) -> int:
    obj = _load_json(palette_json)
    colors = obj.get("colors", {}) or {}
    if not isinstance(colors, dict) or not colors:
        raise ValueError(f"palette.json 非法或 colors 为空: {palette_json}")
    return max(int(k) for k in colors.keys()) + 1


def _resolve_ann_path(*, args_ann: str, selected_json: dict[str, Any], run_meta: dict[str, Any]) -> Path:
    if str(args_ann).strip():
        p = Path(args_ann).expanduser()
        if p.is_absolute():
            if not p.is_file():
                raise FileNotFoundError(f"--ann 不存在: {p}")
            return p
        # prefer cwd, then repo root
        cand1 = (Path.cwd() / p).resolve()
        if cand1.is_file():
            return cand1
        cand2 = (REPO_ROOT / p).resolve()
        if cand2.is_file():
            return cand2
        raise FileNotFoundError(f"--ann 不存在（尝试过 cwd/repo_root）: {p}")

    ann = str(selected_json.get("ann", "")).strip()
    if ann:
        p = Path(ann).expanduser()
        if p.is_file():
            return p
        raise FileNotFoundError(f"selected_clusters.json 记录的 ann 不存在: {p}")

    raise ValueError("未提供 --ann，且 selected_clusters.json 未记录 ann；无法做 COCO 类别混淆统计。")


def _resolve_metrics_csv(*, args_metrics: str, run_meta: dict[str, Any]) -> Path | None:
    if str(args_metrics).strip():
        p = Path(args_metrics).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"--metrics-csv 不存在: {p}")
        return p
    cfg = str(run_meta.get("config", "")).strip()
    if not cfg:
        return None
    cfg_path = Path(cfg).expanduser()
    if not cfg_path.is_absolute():
        cfg_path = (REPO_ROOT / cfg).resolve()
    if not cfg_path.is_file():
        return None
    m = cfg_path.parent / "metrics.csv"
    return m if m.is_file() else None


def _index_coco(coco: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[int, list[dict[str, Any]]], dict[int, str]]:
    images = coco.get("images", []) or []
    anns = coco.get("annotations", []) or []
    cats = coco.get("categories", []) or []

    by_stem: dict[str, dict[str, Any]] = {}
    for im in images:
        if not isinstance(im, dict):
            continue
        fn = str(im.get("file_name", "")).strip()
        if not fn:
            continue
        by_stem.setdefault(Path(fn).stem, im)

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

    id_to_name: dict[int, str] = {}
    for c in cats:
        if not isinstance(c, dict):
            continue
        try:
            cid = int(c.get("id"))
        except Exception:
            continue
        id_to_name[cid] = str(c.get("name", "")).strip() or str(cid)

    if not id_to_name:
        raise ValueError("COCO categories 为空，无法做类别混淆统计。")
    return by_stem, by_img, id_to_name


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
    return arr.astype(np.int64)


def _format_pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"


def _maybe_relpath(path: Path, *, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except Exception:
        return str(path.resolve())


def _set_mpl_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 150,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _plot_top_coverage(scores: list[dict[str, Any]], *, selected: set[int], out_png: Path, top_n: int) -> None:
    rows = sorted(scores, key=lambda r: float(r["coverage"]), reverse=True)[: int(top_n)]
    clusters = [int(r["cluster"]) for r in rows]
    cov = [float(r["coverage"]) for r in rows]
    colors = ["#ff7f0e" if c in selected else "#1f77b4" for c in clusters]

    plt.figure(figsize=(10, max(4.0, 0.25 * len(rows))))
    y = np.arange(len(rows))
    plt.barh(y, cov, color=colors)
    plt.yticks(y, [str(c) for c in clusters])
    plt.gca().invert_yaxis()
    plt.xlabel("coverage (tp / total_oil)")
    plt.title(f"Top-{len(rows)} clusters by oil coverage (selected=orange)")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close()


def _plot_scatter_precision_coverage(
    scores: list[dict[str, Any]],
    *,
    selected: set[int],
    base_rate: float,
    out_png: Path,
    label_top_n: int = 12,
) -> None:
    cov = np.array([float(r["coverage"]) for r in scores], dtype=np.float64)
    prec = np.array([float(r["precision"]) for r in scores], dtype=np.float64)
    sup = np.array([float(r["support"]) for r in scores], dtype=np.float64)
    cid = np.array([int(r["cluster"]) for r in scores], dtype=np.int64)
    sel_mask = np.array([int(r["cluster"]) in selected for r in scores], dtype=bool)

    size = np.sqrt(np.maximum(sup, 1.0))
    size = 8.0 + 0.12 * size

    plt.figure(figsize=(8.5, 6.5))
    plt.scatter(cov[~sel_mask], prec[~sel_mask], s=size[~sel_mask], alpha=0.35, c="#1f77b4", label="non-selected")
    plt.scatter(cov[sel_mask], prec[sel_mask], s=size[sel_mask], alpha=0.8, c="#ff7f0e", label="selected")
    plt.axhline(base_rate, color="#2ca02c", linestyle="--", linewidth=1.2, label=f"base_rate={_format_pct(base_rate)}")
    plt.xlabel("coverage (tp / total_oil)")
    plt.ylabel("precision (tp / (tp+fp))")
    plt.title("Cluster oil enrichment: precision vs coverage (size ~ sqrt(support))")
    plt.legend(loc="best")

    # annotate a few top-coverage points for readability
    order = np.argsort(-cov)[: int(label_top_n)]
    for i in order:
        plt.text(float(cov[i]), float(prec[i]), str(int(cid[i])), fontsize=8)

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close()


def _plot_cum_coverage(scores: list[dict[str, Any]], *, out_png: Path) -> None:
    rows = sorted(scores, key=lambda r: float(r["coverage"]), reverse=True)
    cov = np.array([float(r["coverage"]) for r in rows], dtype=np.float64)
    cum = np.cumsum(cov)
    x = np.arange(1, len(cum) + 1)

    plt.figure(figsize=(8.5, 4.8))
    plt.plot(x, cum, linewidth=2.0)
    plt.xlabel("#clusters (sorted by coverage desc)")
    plt.ylabel("cumulative coverage")
    plt.ylim(0.0, min(1.0, float(cum[-1]) * 1.02))
    plt.title("Oil coverage is long-tailed: cumulative coverage curve")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close()


def _plot_selected_tp_fp(scores: list[dict[str, Any]], *, selected: set[int], out_png: Path) -> None:
    sel_rows = [r for r in scores if int(r["cluster"]) in selected]
    sel_rows.sort(key=lambda r: int(r["fp"]), reverse=True)

    clusters = [int(r["cluster"]) for r in sel_rows]
    tp = np.array([int(r["tp"]) for r in sel_rows], dtype=np.int64)
    fp = np.array([int(r["fp"]) for r in sel_rows], dtype=np.int64)

    plt.figure(figsize=(12, 4.8))
    x = np.arange(len(clusters))
    plt.bar(x, fp, color="#d62728", alpha=0.75, label="fp (outside oil bbox)")
    plt.bar(x, tp, bottom=fp, color="#2ca02c", alpha=0.85, label="tp (inside oil bbox)")
    plt.xticks(x, [str(c) for c in clusters], rotation=60, ha="right")
    plt.ylabel("feat-grid pixels")
    plt.title("Selected oil clusters: tp/fp (sorted by fp desc, larger fp => more background confusion)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close()


def _compute_category_overlap(
    *,
    paths: RunPaths,
    run_meta: dict[str, Any],
    coco: dict[str, Any],
    k_total: int,
    limit: int,
) -> tuple[dict[int, str], dict[int, int], np.ndarray]:
    by_stem, by_img, id_to_name = _index_coco(coco)
    cat_ids = sorted(id_to_name)

    img_size = int(run_meta.get("img_size", 640) or 640)
    input_wh = (img_size, img_size)

    files = run_meta.get("files", []) or []
    if not isinstance(files, list):
        raise TypeError("meta.json 的 files 不是 list")
    if limit and int(limit) > 0:
        files = files[: int(limit)]

    cat_tp = np.zeros((len(cat_ids), k_total), dtype=np.int64)
    cat_total = {cid: 0 for cid in cat_ids}

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
            skipped += 1
            continue

        label_path = paths.labels_dir / f"{stem}.png"
        if not label_path.is_file():
            skipped += 1
            continue
        label_map = _load_label_map(label_path)
        fh, fw = int(label_map.shape[0]), int(label_map.shape[1])

        masks = {cid: np.zeros((fh, fw), dtype=bool) for cid in cat_ids}
        anns = by_img.get(img_id, []) or []
        for ann in anns:
            if not isinstance(ann, dict):
                continue
            if int(ann.get("iscrowd", 0) or 0) != 0:
                continue
            try:
                cid = int(ann.get("category_id"))
            except Exception:
                continue
            if cid not in masks:
                continue
            bbox = ann.get("bbox", None)
            if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
                continue
            slices = _bbox_to_feat_slices(bbox, orig_wh=(orig_w, orig_h), input_wh=input_wh, feat_hw=(fh, fw))
            if slices is None:
                continue
            ys, xs = slices
            masks[cid][ys, xs] = True

        flat = label_map.reshape(-1).astype(np.int64)
        for idx, cid in enumerate(cat_ids):
            mask = masks[cid].reshape(-1)
            if not mask.any():
                continue
            cat_total[cid] += int(mask.sum())
            bc = np.bincount(flat[mask], minlength=k_total)
            cat_tp[idx] += bc.astype(np.int64)

        matched += 1

    logging.info("category overlap: matched=%d skipped=%d", matched, skipped)
    return id_to_name, cat_total, cat_tp


def _plot_category_overlap_summary(
    *,
    id_to_name: dict[int, str],
    cat_total: dict[int, int],
    cat_tp: np.ndarray,
    support: np.ndarray,
    selected: set[int],
    out_png: Path,
) -> dict[str, Any]:
    cat_ids = [cid for cid in sorted(id_to_name)]
    sel = np.array(sorted(selected), dtype=np.int64)
    sel_support = float(np.sum(support[sel])) if sel.size > 0 else 0.0

    # aggregate for selected clusters
    cov = []
    share = []
    names = []
    for i, cid in enumerate(cat_ids):
        total = float(cat_total.get(cid, 0))
        tp = float(np.sum(cat_tp[i, sel])) if sel.size > 0 else 0.0
        names.append(id_to_name[cid])
        cov.append((tp / total) if total > 0 else 0.0)
        share.append((tp / sel_support) if sel_support > 0 else 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].bar(names, cov, color="#1f77b4")
    axes[0].set_title("Coverage of each category by selected clusters")
    axes[0].set_ylabel("coverage (tp_in_cat_bbox / total_cat_bbox_pixels)")
    axes[0].set_ylim(0.0, max(cov) * 1.2 + 1e-6)

    axes[1].bar(names, share, color="#ff7f0e")
    axes[1].set_title("Share of selected-cluster support inside each category bbox")
    axes[1].set_ylabel("share (tp_in_cat_bbox / sum_support_selected)")
    axes[1].set_ylim(0.0, max(share) * 1.2 + 1e-6)

    for ax in axes:
        ax.grid(True, alpha=0.25)
        for tick in ax.get_xticklabels():
            tick.set_rotation(25)
            tick.set_ha("right")

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close(fig)

    out = {}
    for i, cid in enumerate(cat_ids):
        out[id_to_name[cid]] = {
            "coverage": float(cov[i]),
            "share_within_selected_support": float(share[i]),
            "total_pixels": int(cat_total.get(cid, 0)),
        }
    return out


def _plot_selected_best_other_precision(
    *,
    id_to_name: dict[int, str],
    cat_tp: np.ndarray,
    support: np.ndarray,
    scores_by_cluster: dict[int, dict[str, Any]],
    selected: set[int],
    out_png: Path,
) -> list[dict[str, Any]]:
    cat_ids = [cid for cid in sorted(id_to_name)]
    if "oil" not in {n.lower() for n in id_to_name.values()}:
        raise ValueError("categories 中未找到 'oil'，无法生成混淆图（需要 oil / building / machine / photovoltaic）。")
    oil_cid = next(cid for cid, name in id_to_name.items() if name.lower() == "oil")
    oil_idx = cat_ids.index(oil_cid)

    rows: list[dict[str, Any]] = []
    for cl in sorted(selected):
        sup = float(support[int(cl)])
        if sup <= 0:
            continue
        oil_tp = float(cat_tp[oil_idx, int(cl)])
        oil_prec = oil_tp / sup
        best_other = None
        for i, cid in enumerate(cat_ids):
            if cid == oil_cid:
                continue
            tp = float(cat_tp[i, int(cl)])
            prec = tp / sup
            if best_other is None or prec > best_other["prec"]:
                best_other = {"category": id_to_name[cid], "prec": prec, "tp": tp}
        if best_other is None:
            continue
        srow = scores_by_cluster.get(int(cl), {})
        rows.append(
            {
                "cluster": int(cl),
                "support": int(sup),
                "oil_precision_bbox": float(srow.get("precision", oil_prec)),
                "oil_coverage": float(srow.get("coverage", 0.0)),
                "best_other_category": str(best_other["category"]),
                "best_other_precision": float(best_other["prec"]),
                "best_other_tp": int(best_other["tp"]),
            }
        )

    rows.sort(key=lambda r: r["best_other_precision"], reverse=True)

    # plot
    labels = [f"{r['cluster']}({r['best_other_category']})" for r in rows]
    vals = [float(r["best_other_precision"]) for r in rows]
    colors = []
    color_map = {
        "building": "#9467bd",
        "machine": "#8c564b",
        "photovoltaic": "#17becf",
    }
    for r in rows:
        colors.append(color_map.get(str(r["best_other_category"]).lower(), "#7f7f7f"))

    plt.figure(figsize=(12, max(4.0, 0.25 * len(rows))))
    y = np.arange(len(rows))
    plt.barh(y, vals, color=colors)
    plt.yticks(y, labels)
    plt.gca().invert_yaxis()
    plt.xlabel("best_other_precision = tp_in_other_bbox / support")
    plt.title("Selected clusters: strongest overlap with non-oil categories (higher => more confusion)")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close()

    return rows


def _compute_cluster_centers_from_assignments(
    *,
    paths: RunPaths,
    run_meta: dict[str, Any],
    cluster_meta: dict[str, Any],
    k_total: int,
    limit: int,
    cache_path: Path,
    recompute: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if cache_path.is_file() and not recompute:
        obj = np.load(cache_path, allow_pickle=False)
        if (
            int(obj["k_total"]) == int(k_total)
            and int(obj["feat_channels"]) > 0
            and bool(obj["l2norm"]) == bool(cluster_meta.get("l2norm", False))
        ):
            centers = obj["centers"].astype(np.float64)
            counts = obj["counts"].astype(np.int64)
            return centers, counts

    files = run_meta.get("files", []) or []
    if not isinstance(files, list):
        raise TypeError("meta.json 的 files 不是 list")
    if limit and int(limit) > 0:
        files = files[: int(limit)]

    # infer feat_channels from first npz
    npz0 = None
    for it in files:
        if not isinstance(it, dict):
            continue
        p = Path(str(it.get("npz", ""))).expanduser()
        if p.is_file():
            npz0 = p
            break
        stem = Path(str(it.get("image", ""))).stem
        p2 = paths.features_dir / f"{stem}.npz"
        if p2.is_file():
            npz0 = p2
            break
    if npz0 is None:
        raise FileNotFoundError("无法在 meta.json.files 中找到任何可用的 npz 路径。")

    feat0 = np.load(npz0, allow_pickle=False)["feat"]
    if feat0.ndim != 3:
        raise ValueError(f"feat 期望 [C,H,W]，实际 shape={feat0.shape} file={npz0}")
    c = int(feat0.shape[0])

    sum_vec = np.zeros((k_total, c), dtype=np.float64)
    counts = np.zeros((k_total,), dtype=np.int64)

    l2norm = bool(cluster_meta.get("l2norm", False))
    processed = 0

    for it in files:
        if not isinstance(it, dict):
            continue
        image_path = Path(str(it.get("image", ""))).expanduser()
        stem = image_path.stem
        npz_path = Path(str(it.get("npz", ""))).expanduser()
        if not npz_path.is_file():
            npz_path = paths.features_dir / f"{stem}.npz"
        label_path = paths.labels_dir / f"{stem}.png"
        if not npz_path.is_file() or not label_path.is_file():
            continue

        feat = np.load(npz_path, allow_pickle=False)["feat"]  # [C,H,W]
        if feat.shape[0] != c:
            raise ValueError(f"feat_channels 不一致: expect {c} got {feat.shape[0]} file={npz_path}")
        x = np.transpose(feat, (1, 2, 0)).reshape(-1, c).astype(np.float32)  # [N,C]
        if l2norm:
            denom = np.linalg.norm(x, axis=1, keepdims=True)
            x = x / np.maximum(denom, 1e-12)
        y = _load_label_map(label_path).reshape(-1)  # [N]

        counts += np.bincount(y, minlength=k_total).astype(np.int64)
        np.add.at(sum_vec, y, x)
        processed += 1

    if processed == 0:
        raise RuntimeError("未能处理任何 feature/label 文件对，无法计算 centers。")

    centers = sum_vec / np.maximum(counts[:, None], 1)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        centers=centers.astype(np.float32),
        counts=counts,
        k_total=np.array(int(k_total), dtype=np.int64),
        feat_channels=np.array(int(c), dtype=np.int64),
        l2norm=np.array(bool(l2norm)),
        files=np.array(int(processed), dtype=np.int64),
    )
    return centers, counts


def _distance_report_from_centers(centers: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    c = centers.astype(np.float64)
    c = c / np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-12)
    sim = c @ c.T
    sim = np.clip(sim, -1.0, 1.0)
    dist = 1.0 - sim
    tri = dist[np.triu_indices(dist.shape[0], k=1)]
    stats = {
        "min": float(tri.min()),
        "p1": float(np.percentile(tri, 1)),
        "p5": float(np.percentile(tri, 5)),
        "median": float(np.median(tri)),
        "p95": float(np.percentile(tri, 95)),
        "max": float(tri.max()),
        "mean": float(tri.mean()),
    }
    return dist, stats


def _plot_distance_hist(dist: np.ndarray, *, out_png: Path) -> None:
    tri = dist[np.triu_indices(dist.shape[0], k=1)]
    plt.figure(figsize=(8.5, 4.8))
    plt.hist(tri, bins=60, color="#1f77b4", alpha=0.85)
    plt.xlabel("cosine distance (1 - cos)")
    plt.ylabel("#pairs")
    plt.title("All cluster pairs distance histogram")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close()


def _plot_nearest_oil_to_non(
    dist: np.ndarray, *, oil_clusters: list[int], out_png: Path
) -> dict[str, Any]:
    oil = np.array(sorted(set(int(x) for x in oil_clusters)), dtype=np.int64)
    mask = np.ones((dist.shape[0],), dtype=bool)
    mask[oil] = False
    non = np.arange(dist.shape[0], dtype=np.int64)[mask]
    d = dist[np.ix_(oil, non)]
    nearest = d.min(axis=1)

    plt.figure(figsize=(8.5, 4.8))
    plt.hist(nearest, bins=24, color="#ff7f0e", alpha=0.85)
    plt.xlabel("nearest cosine distance to a non-oil cluster")
    plt.ylabel("#oil clusters")
    plt.title("Oil clusters: nearest non-oil distance (smaller => more confusable)")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close()

    def pct(p: float) -> float:
        return float(np.percentile(nearest, p))

    # top confusable pairs
    argmin = d.argmin(axis=1)
    closest_non = non[argmin]
    order = np.argsort(nearest)
    pairs = []
    for j in order[:10]:
        pairs.append(
            {
                "oil_cluster": int(oil[j]),
                "nearest_non_oil_cluster": int(closest_non[j]),
                "distance": float(nearest[j]),
            }
        )

    return {
        "nearest_stats": {
            "min": float(nearest.min()),
            "p5": pct(5),
            "median": float(np.median(nearest)),
            "mean": float(nearest.mean()),
            "p95": pct(95),
            "max": float(nearest.max()),
        },
        "most_confusable_pairs_top10": pairs,
    }


def _plot_oil_oil_heatmap(dist: np.ndarray, *, oil_clusters: list[int], out_png: Path) -> dict[str, float]:
    oil = np.array(sorted(set(int(x) for x in oil_clusters)), dtype=np.int64)
    d = dist[np.ix_(oil, oil)]
    tri = d[np.triu_indices(d.shape[0], k=1)]

    plt.figure(figsize=(7.8, 6.6))
    im = plt.imshow(d, cmap="viridis", vmin=0.0, vmax=min(1.6, float(np.max(d))))
    plt.colorbar(im, fraction=0.046, pad=0.04, label="cosine distance")
    plt.xticks(np.arange(len(oil)), [str(int(x)) for x in oil], rotation=90, fontsize=7)
    plt.yticks(np.arange(len(oil)), [str(int(x)) for x in oil], fontsize=7)
    plt.title("Oil clusters: pairwise cosine distance heatmap")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close()

    return {
        "min": float(tri.min()) if tri.size else 0.0,
        "p5": float(np.percentile(tri, 5)) if tri.size else 0.0,
        "median": float(np.median(tri)) if tri.size else 0.0,
        "mean": float(tri.mean()) if tri.size else 0.0,
        "p95": float(np.percentile(tri, 95)) if tri.size else 0.0,
        "max": float(tri.max()) if tri.size else 0.0,
    }


def _plot_training_curves(metrics_csv: Path, *, out_png: Path) -> dict[str, Any]:
    df = pd.read_csv(metrics_csv)
    if "epoch" not in df.columns or "train_loss" not in df.columns or "val_loss" not in df.columns:
        raise ValueError(f"metrics.csv 缺少必需列: epoch/train_loss/val_loss, file={metrics_csv}")

    epoch = df["epoch"].to_numpy()
    train_loss = df["train_loss"].to_numpy(dtype=float)
    val_loss = df["val_loss"].to_numpy(dtype=float)

    best_epoch = None
    best_metric = None
    metric_col = None
    for cand in ["val_map50_95", "val_coco_eval_bbox_0"]:
        if cand in df.columns:
            metric_col = cand
            idx = int(df[cand].astype(float).idxmax())
            best_epoch = int(df.loc[idx, "epoch"])
            best_metric = float(df.loc[idx, cand])
            break

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.0), sharex=True)
    axes[0].plot(epoch, train_loss, label="train_loss", linewidth=2.0)
    axes[0].plot(epoch, val_loss, label="val_loss", linewidth=2.0)
    axes[0].set_ylabel("loss")
    axes[0].set_title("Training curves")
    axes[0].legend(loc="best")
    if best_epoch is not None:
        axes[0].axvline(best_epoch, color="#d62728", linestyle="--", linewidth=1.2, label="best")

    if metric_col is not None:
        metric = df[metric_col].to_numpy(dtype=float)
        axes[1].plot(epoch, metric, label=metric_col, color="#2ca02c", linewidth=2.0)
        axes[1].set_ylabel(metric_col)
        axes[1].legend(loc="best")
        if best_epoch is not None:
            axes[1].axvline(best_epoch, color="#d62728", linestyle="--", linewidth=1.2)
            axes[1].text(best_epoch, float(np.nanmax(metric)), f"best@{best_epoch}", fontsize=9, va="top")
    else:
        axes[1].text(0.02, 0.7, "No val metric column found", transform=axes[1].transAxes)

    axes[1].set_xlabel("epoch")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close(fig)

    last = df.iloc[-1]
    out = {
        "last_epoch": int(last["epoch"]),
        "last_train_loss": float(last["train_loss"]),
        "last_val_loss": float(last["val_loss"]),
    }
    if best_epoch is not None and metric_col is not None:
        # re-locate best row (idxmax computed above)
        idx = int(df[metric_col].astype(float).idxmax())
        out.update(
            {
                "best_epoch": int(df.loc[idx, "epoch"]),
                "best_metric_col": str(metric_col),
                "best_metric": float(df.loc[idx, metric_col]),
                "best_train_loss": float(df.loc[idx, "train_loss"]),
                "best_val_loss": float(df.loc[idx, "val_loss"]),
            }
        )
    return out


def main() -> int:
    _set_mpl_style()
    args = _parse_args()
    paths = _ensure_run_paths(Path(args.run_dir))

    out_dir = Path(args.out_dir).expanduser() if str(args.out_dir).strip() else (paths.run_dir / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    run_meta = _load_json(paths.meta_json)
    cluster_meta = _load_json(paths.clusters_dir / "cluster_meta.json")
    selected_json = _load_json(paths.selection_dir / "selected_clusters.json")
    cluster_scores_path = paths.selection_dir / "cluster_scores.json"
    if not cluster_scores_path.is_file():
        raise FileNotFoundError(f"缺少 cluster_scores.json: {cluster_scores_path}；请先运行 repr_auto_select_oil_clusters.py。")
    scores = _load_json(cluster_scores_path)
    if not isinstance(scores, list) or not scores:
        raise TypeError(f"cluster_scores.json 期望 list，实际为 {type(scores)}")

    palette_json = paths.clusters_dir / "palette.json"
    k_total = _palette_k_total(palette_json)
    selected = {int(x) for x in (selected_json.get("oil_clusters", []) or [])}

    # Convert scores to lookup
    scores_by_cluster = {int(r["cluster"]): r for r in scores if isinstance(r, dict) and "cluster" in r}

    # base rate (consistent with repr_auto_select_oil_clusters.py)
    total_oil = float(sum(int(r.get("tp", 0)) for r in scores))
    total_cells = float(sum(int(r.get("support", 0)) for r in scores))
    base_rate = (total_oil / total_cells) if total_cells > 0 else 0.0

    # --- cluster distribution plots ---
    _plot_top_coverage(scores, selected=selected, out_png=out_dir / f"cluster_top{int(args.top_n)}_coverage.png", top_n=int(args.top_n))
    _plot_scatter_precision_coverage(
        scores,
        selected=selected,
        base_rate=base_rate,
        out_png=out_dir / "cluster_precision_vs_coverage_scatter.png",
    )
    _plot_cum_coverage(scores, out_png=out_dir / "cluster_cumulative_coverage.png")
    _plot_selected_tp_fp(scores, selected=selected, out_png=out_dir / "selected_clusters_tp_fp.png")

    # Export a compact table
    pd.DataFrame(scores).to_csv(out_dir / "cluster_scores.csv", index=False)

    # --- category confusion plots ---
    ann_path = _resolve_ann_path(args_ann=str(args.ann), selected_json=selected_json, run_meta=run_meta)
    coco = _load_json(ann_path)
    id_to_name, cat_total, cat_tp = _compute_category_overlap(
        paths=paths,
        run_meta=run_meta,
        coco=coco,
        k_total=k_total,
        limit=int(args.limit),
    )
    support = np.zeros((k_total,), dtype=np.int64)
    for r in scores:
        support[int(r["cluster"])] = int(r["support"])

    category_summary = _plot_category_overlap_summary(
        id_to_name=id_to_name,
        cat_total=cat_total,
        cat_tp=cat_tp,
        support=support,
        selected=selected,
        out_png=out_dir / "category_overlap_summary.png",
    )
    best_other_rows = _plot_selected_best_other_precision(
        id_to_name=id_to_name,
        cat_tp=cat_tp,
        support=support,
        scores_by_cluster=scores_by_cluster,
        selected=selected,
        out_png=out_dir / "selected_clusters_best_other_precision.png",
    )
    Path(out_dir / "selected_clusters_best_other_precision.json").write_text(
        json.dumps(best_other_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    Path(out_dir / "category_overlap_summary.json").write_text(
        json.dumps(category_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # --- distance plots ---
    cache_path = out_dir / "cache_cluster_centers.npz"
    centers, counts = _compute_cluster_centers_from_assignments(
        paths=paths,
        run_meta=run_meta,
        cluster_meta=cluster_meta,
        k_total=k_total,
        limit=int(args.limit),
        cache_path=cache_path,
        recompute=bool(args.recompute_centers),
    )
    dist, dist_stats = _distance_report_from_centers(centers)
    _plot_distance_hist(dist, out_png=out_dir / "distance_hist_all_pairs.png")
    nearest_report = _plot_nearest_oil_to_non(dist, oil_clusters=sorted(selected), out_png=out_dir / "distance_nearest_oil_to_non.png")
    oil_oil_stats = _plot_oil_oil_heatmap(dist, oil_clusters=sorted(selected), out_png=out_dir / "distance_heatmap_oil_oil.png")

    distance_report = {
        "method": "center_cosine_distance",
        "center_definition": "mean of per-pixel features assigned to each cluster (features L2-normalized if cluster_meta.l2norm=true)",
        "all_pairs": dist_stats,
        "oil_to_non": nearest_report,
        "oil_oil": oil_oil_stats,
    }
    Path(out_dir / "distance_report.json").write_text(json.dumps(distance_report, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- training curves ---
    metrics_csv = _resolve_metrics_csv(args_metrics=str(args.metrics_csv), run_meta=run_meta)
    training_summary = None
    if metrics_csv is not None and metrics_csv.is_file():
        training_summary = _plot_training_curves(metrics_csv, out_png=out_dir / "training_curves.png")
        Path(out_dir / "training_summary.json").write_text(
            json.dumps({"metrics_csv": str(metrics_csv), **training_summary}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    else:
        logging.warning("未找到 metrics.csv（可用 --metrics-csv 显式指定），跳过训练曲线可视化。")

    # --- summary ---
    summary = {
        "run_dir": str(paths.run_dir),
        "k_total": int(k_total),
        "k1": int(cluster_meta.get("k1", 0) or 0),
        "k2": int(cluster_meta.get("k2", 0) or 0),
        "l2norm": bool(cluster_meta.get("l2norm", False)),
        "base_rate": float(base_rate),
        "selected_clusters": sorted(selected),
        "selected_count": int(len(selected)),
        "ann": str(ann_path),
        "charts": {
            "cluster_top_coverage": _maybe_relpath(out_dir / f"cluster_top{int(args.top_n)}_coverage.png", base=paths.run_dir),
            "cluster_scatter": _maybe_relpath(out_dir / "cluster_precision_vs_coverage_scatter.png", base=paths.run_dir),
            "cluster_cum_coverage": _maybe_relpath(out_dir / "cluster_cumulative_coverage.png", base=paths.run_dir),
            "selected_tp_fp": _maybe_relpath(out_dir / "selected_clusters_tp_fp.png", base=paths.run_dir),
            "category_overlap": _maybe_relpath(out_dir / "category_overlap_summary.png", base=paths.run_dir),
            "selected_best_other": _maybe_relpath(out_dir / "selected_clusters_best_other_precision.png", base=paths.run_dir),
            "distance_hist": _maybe_relpath(out_dir / "distance_hist_all_pairs.png", base=paths.run_dir),
            "distance_nearest": _maybe_relpath(out_dir / "distance_nearest_oil_to_non.png", base=paths.run_dir),
            "distance_oil_heatmap": _maybe_relpath(out_dir / "distance_heatmap_oil_oil.png", base=paths.run_dir),
            "training_curves": _maybe_relpath(out_dir / "training_curves.png", base=paths.run_dir)
            if (out_dir / "training_curves.png").is_file()
            else "",
        },
    }
    (out_dir / "analysis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("done: out_dir=%s", out_dir)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
