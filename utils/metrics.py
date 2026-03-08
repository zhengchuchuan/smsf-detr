from pathlib import Path
from typing import Union, Optional, Sequence, Any, Tuple

import matplotlib

# 强制使用无 GUI 后端，避免 Tkinter 主线程错误
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import math

plt.ioff()

_MIN_LONG_EDGE_PX = 2048
_BASE_DPI = 150


def _save_figure(fig, path: Path, *, bbox_inches: str | None = "tight") -> None:
    """
    统一保存图表：
    - 自动调高 dpi，保证图像长边至少 _MIN_LONG_EDGE_PX；
    - 默认使用 bbox_inches='tight'，减少空白边缘。
    """
    try:
        w_in, h_in = fig.get_size_inches()
        long_in = float(max(w_in, h_in)) if w_in and h_in else 0.0
    except Exception:
        long_in = 0.0

    if long_in > 0:
        dpi_req = int(math.ceil(_MIN_LONG_EDGE_PX / long_in))
        dpi = max(_BASE_DPI, dpi_req)
    else:
        dpi = _BASE_DPI

    path.parent.mkdir(parents=True, exist_ok=True)

    # bbox_inches='tight' 会裁剪空白，导致实际输出像素低于按 figsize 推算的像素；
    # 这里采用“保存 -> 读取尺寸 -> 必要时提高 dpi 再保存一次”的方式确保长边达标。
    for _ in range(2):
        fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
        try:
            from PIL import Image  # Pillow 通常已作为 torchvision/PIL 依赖存在

            with Image.open(path) as im:
                w_px, h_px = im.size
            long_px = int(max(w_px, h_px))
            if long_px >= _MIN_LONG_EDGE_PX:
                break
            if long_px <= 0:
                break
            scale = float(_MIN_LONG_EDGE_PX) / float(long_px)
            dpi = int(math.ceil(float(dpi) * scale))
        except Exception:
            break


def safe_index(arr, idx):
    return arr[idx] if isinstance(arr, (list, tuple)) and 0 <= idx < len(arr) else None


class MetricsPlotSink:
    """
    轻量指标记录与可视化（参考 msif-detr），生成 metrics_plot.png。
    """
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.history: list[dict] = []

    def update(self, values: dict):
        self.history.append(values)

    def save(self):
        if not self.history:
            return

        def get_xy(key: str) -> tuple[np.ndarray, np.ndarray]:
            xs: list[float] = []
            ys: list[float] = []
            for h in self.history:
                if "epoch" not in h or key not in h:
                    continue
                x_val = _sanitize_numeric(h.get("epoch"))
                y_val = _sanitize_numeric(h.get(key))
                if x_val is None or y_val is None:
                    continue
                xs.append(float(x_val))
                ys.append(float(y_val))
            return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)

        # 兼容 val/test 命名
        split_prefix = "val" if any("val_loss" in h for h in self.history) else "test"
        val_loss_key = f"{split_prefix}_loss"

        train_loss_x, train_loss_y = get_xy("train_loss")
        val_loss_x, val_loss_y = get_xy(val_loss_key)

        coco_x: list[float] = []
        coco_eval: list = []
        for h in self.history:
            x_val = _sanitize_numeric(h.get("epoch"))
            eval_stats = h.get("val_coco_eval_bbox") or h.get("test_coco_eval_bbox")
            if x_val is None or eval_stats is None:
                continue
            coco_x.append(float(x_val))
            coco_eval.append(eval_stats)
        coco_epochs = np.array(coco_x, dtype=np.float32)
        ap50_90 = np.array([safe_index(x, 0) for x in coco_eval], dtype=np.float32)  # AP@[.5:.95]
        ap50 = np.array([safe_index(x, 1) for x in coco_eval], dtype=np.float32)  # AP@0.5
        ap_s = np.array([safe_index(x, 3) for x in coco_eval], dtype=np.float32)  # AP small
        ap_m = np.array([safe_index(x, 4) for x in coco_eval], dtype=np.float32)  # AP medium
        ap_l = np.array([safe_index(x, 5) for x in coco_eval], dtype=np.float32)  # AP large
        ar_100 = np.array([safe_index(x, 8) for x in coco_eval], dtype=np.float32)  # AR@100

        fig, axes = plt.subplots(2, 2, figsize=(18, 12))

        if train_loss_x.size > 0:
            axes[0][0].plot(train_loss_x, train_loss_y, label="Training Loss", marker="o", linestyle="-")
        if val_loss_x.size > 0:
            axes[0][0].plot(val_loss_x, val_loss_y, label="Validation Loss", marker="o", linestyle="--")
        axes[0][0].set_title("Training and Validation Loss")
        axes[0][0].set_xlabel("Epoch")
        axes[0][0].set_ylabel("Loss")
        axes[0][0].set_ylim(bottom=0.0)
        axes[0][0].legend()
        axes[0][0].grid(True)

        if ap50.size > 0 and coco_epochs.size > 0:
            axes[0][1].plot(coco_epochs[: len(ap50)], ap50, marker="o", linestyle="-", label="AP50")
        if ap50_90.size > 0 and coco_epochs.size > 0:
            axes[0][1].plot(coco_epochs[: len(ap50_90)], ap50_90, marker="o", linestyle="--", label="AP50_95")
        if ap_s.size > 0 and coco_epochs.size > 0:
            axes[0][1].plot(coco_epochs[: len(ap_s)], ap_s, marker="o", linestyle=":", label="AP_s")
        if ap_m.size > 0 and coco_epochs.size > 0:
            axes[0][1].plot(coco_epochs[: len(ap_m)], ap_m, marker="o", linestyle="-.", label="AP_m")
        if ap_l.size > 0 and coco_epochs.size > 0:
            axes[0][1].plot(coco_epochs[: len(ap_l)], ap_l, marker="o", linestyle=(0, (3, 1, 1, 1)), label="AP_l")
        axes[0][1].set_title('Average Precision')
        axes[0][1].set_xlabel('Epoch')
        axes[0][1].set_ylabel('AP')
        axes[0][1].set_ylim(0.0, 1.0)
        axes[0][1].legend()
        axes[0][1].grid(True)

        if ar_100.size > 0 and coco_epochs.size > 0:
            axes[1][0].plot(coco_epochs[: len(ar_100)], ar_100, marker="o", linestyle="-", label="AR@100")
            axes[1][0].set_title('Average Recall')
            axes[1][0].set_xlabel('Epoch')
            axes[1][0].set_ylabel('AR')
            axes[1][0].set_ylim(0.0, 1.0)
            axes[1][0].legend()
            axes[1][0].grid(True)

        # Loss 明细（分类/回归等），支持 DETR 与 YOLO 命名。
        loss_label_map = {
            "loss_ce": "cls_ce",
            "loss_bbox": "bbox_l1",
            "loss_giou": "bbox_giou",
            "loss_mask_ce": "mask_ce",
            "loss_mask_dice": "mask_dice",
            "loss_obj": "obj",
            "loss_cls": "cls",
        }

        train_detail_keys = {
            k[len("train_") :]
            for h in self.history
            for k in h.keys()
            if isinstance(k, str) and k.startswith("train_loss_") and k not in {"train_loss"}
        }
        val_detail_keys = {
            k[len("val_") :]
            for h in self.history
            for k in h.keys()
            if isinstance(k, str) and k.startswith("val_loss_") and k not in {"val_loss"}
        }
        test_detail_keys = {
            k[len("test_") :]
            for h in self.history
            for k in h.keys()
            if isinstance(k, str) and k.startswith("test_loss_") and k not in {"test_loss"}
        }
        detail_bases = train_detail_keys | (val_detail_keys if split_prefix == "val" else test_detail_keys)
        detail_bases.discard("loss_total")
        detail_bases.discard("loss")

        preferred = ["loss_ce", "loss_bbox", "loss_giou", "loss_obj", "loss_cls"]
        ordered_bases = [b for b in preferred if b in detail_bases] + sorted([b for b in detail_bases if b not in preferred])

        ax_loss = axes[1][1]
        plotted_any = False

        if train_loss_x.size > 0:
            ax_loss.plot(train_loss_x, train_loss_y, label="loss_total (train)", linewidth=2.2, color="black")
            plotted_any = True
        if val_loss_x.size > 0:
            ax_loss.plot(val_loss_x, val_loss_y, label=f"loss_total ({split_prefix})", linewidth=2.2, linestyle="--", color="black")
            plotted_any = True

        base_to_color: dict[str, str] = {}
        color_cycle = plt.rcParams.get("axes.prop_cycle", None)
        colors = (
            [d.get("color") for d in color_cycle] if color_cycle is not None else None
        ) or ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]

        for idx, base in enumerate(ordered_bases):
            base_to_color[base] = colors[idx % len(colors)]

        for base in ordered_bases:
            label = loss_label_map.get(base, base.replace("loss_", ""))
            color = base_to_color[base]

            train_key = f"train_{base}"
            val_key = f"{split_prefix}_{base}"

            tx, ty = get_xy(train_key)
            vx, vy = get_xy(val_key)
            if tx.size > 0:
                ax_loss.plot(tx, ty, label=f"{label} (train)", color=color, linestyle="-", alpha=0.9)
                plotted_any = True
            if vx.size > 0:
                ax_loss.plot(vx, vy, label=f"{label} ({split_prefix})", color=color, linestyle="--", alpha=0.9)
                plotted_any = True

        if plotted_any:
            ax_loss.set_title("Loss Breakdown")
            ax_loss.set_xlabel("Epoch")
            ax_loss.set_ylabel("Loss")
            ax_loss.set_ylim(bottom=0.0)
            ax_loss.grid(True)
            ax_loss.legend(ncol=2, fontsize=9)
        else:
            ax_loss.axis("off")

        plt.tight_layout()
        out_file = self.output_dir / "metrics_plot.png"
        _save_figure(fig, out_file, bbox_inches=None)
        plt.close(fig)


def _sanitize_numeric(value):
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(numeric) else float(numeric)


def _prepare_curve(xs, ys):
    paired_x, paired_y = [], []
    if xs is None or ys is None:
        return np.array([]), np.array([])
    for x, y in zip(xs, ys):
        x_val = _sanitize_numeric(x)
        y_val = _sanitize_numeric(y)
        if x_val is None or y_val is None:
            continue
        paired_x.append(x_val)
        paired_y.append(y_val)
    return np.array(paired_x, dtype=np.float32), np.array(paired_y, dtype=np.float32)


def _annotate_barh(ax, bars, *, fmt: str = "{:.3f}", dx: float = 0.01, fontsize: int = 8):
    for bar in bars:
        try:
            width = float(bar.get_width())
            y = float(bar.get_y() + bar.get_height() / 2.0)
        except Exception:
            continue
        ax.text(
            width + dx,
            y,
            fmt.format(width),
            va="center",
            ha="left",
            fontsize=fontsize,
            color="black",
        )


def save_coco_dataset_distribution_charts(
    coco_api,
    output_dir: Union[str, Path],
    *,
    prefix: str = "dataset",
    small_thr: int = 32,
    medium_thr: int = 96,
    max_annotate_classes: int = 60,
    annotate: bool = True,
    save_class_counts: bool = True,
    save_size_ratio: bool = True,
    title_suffix: str = "",
):
    """
    基于 COCO 标注统计数据分布图：
    - {prefix}_class_counts.png：每类实例数量（含 all）
    - {prefix}_size_ratio.png：每类 small/medium/large 占比（含 all）

    默认使用 COCO 常用阈值（以 bbox 边长像素计）：small < 32^2, medium < 96^2, else large。
    """
    if coco_api is None or not hasattr(coco_api, "dataset"):
        return
    dataset = coco_api.dataset or {}
    cats = dataset.get("categories") or []
    anns = dataset.get("annotations") or []
    if not cats or not anns:
        return

    cat_id_to_name = {int(c["id"]): str(c.get("name", f"id_{c['id']}")) for c in cats if "id" in c}
    cat_ids = sorted(cat_id_to_name.keys())
    cat_id_to_idx = {cid: idx for idx, cid in enumerate(cat_ids)}

    small_area = float(small_thr * small_thr)
    medium_area = float(medium_thr * medium_thr)

    counts = np.zeros((len(cat_ids),), dtype=np.int64)
    size_counts = np.zeros((len(cat_ids), 3), dtype=np.int64)  # small/medium/large

    for ann in anns:
        try:
            cid = int(ann.get("category_id"))
        except Exception:
            continue
        if cid not in cat_id_to_idx:
            continue
        bbox = ann.get("bbox") or None
        if not (isinstance(bbox, (list, tuple)) and len(bbox) >= 4):
            continue
        w = float(bbox[2])
        h = float(bbox[3])
        if w <= 0 or h <= 0:
            continue
        area = w * h
        idx = cat_id_to_idx[cid]
        counts[idx] += 1
        if area < small_area:
            size_counts[idx, 0] += 1
        elif area < medium_area:
            size_counts[idx, 1] += 1
        else:
            size_counts[idx, 2] += 1

    total_count = int(counts.sum())
    total_sizes = size_counts.sum(axis=0)

    class_names = [cat_id_to_name[cid] for cid in cat_ids]
    class_names = ["all"] + class_names
    counts_all = np.concatenate([[total_count], counts.astype(np.int64)])
    size_all = np.vstack([total_sizes.astype(np.int64), size_counts.astype(np.int64)])

    # Sort by count (keep all at top)
    if counts_all.size > 1:
        order = np.argsort(-counts_all[1:]) + 1
        order = np.concatenate([[0], order])
        class_names = [class_names[i] for i in order.tolist()]
        counts_all = counts_all[order]
        size_all = size_all[order]

    charts_dir = Path(output_dir) / "metric_charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    prefix = prefix.replace(" ", "_") or "dataset"
    title_suffix = str(title_suffix).strip()

    def _with_suffix(title: str) -> str:
        return f"{title} - {title_suffix}" if title_suffix else title

    # Class counts
    y = np.arange(len(class_names))
    if save_class_counts:
        fig, ax = plt.subplots(figsize=(12, max(6, len(class_names) * 0.32)))
        bars = ax.barh(y, counts_all.astype(float), color="#1f77b4")
        ax.set_yticks(y)
        ax.set_yticklabels(class_names)
        ax.invert_yaxis()
        ax.set_xlabel("Instances")
        ax.set_title(_with_suffix("Dataset Class Counts"))
        ax.grid(True, axis="x", linestyle="--", alpha=0.3)
        if annotate and len(class_names) <= max_annotate_classes:
            _annotate_barh(ax, bars, fmt="{:.0f}", dx=max(1.0, float(counts_all.max()) * 0.01), fontsize=8)
        fig.tight_layout()
        _save_figure(fig, charts_dir / f"{prefix}_class_counts.png")
        plt.close(fig)

    # Size ratio (stacked)
    if save_size_ratio:
        totals = size_all.sum(axis=1, keepdims=True).astype(np.float32)
        totals[totals <= 0] = 1.0
        ratios = size_all.astype(np.float32) / totals
        labels = ["small", "medium", "large"]
        colors = ["#4c78a8", "#f58518", "#54a24b"]

        fig, ax = plt.subplots(figsize=(12, max(6, len(class_names) * 0.32)))
        left = np.zeros((len(class_names),), dtype=np.float32)
        for k in range(3):
            ax.barh(y, ratios[:, k], left=left, color=colors[k], label=labels[k])
            left += ratios[:, k]
        ax.set_yticks(y)
        ax.set_yticklabels(class_names)
        ax.invert_yaxis()
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("Ratio")
        ax.set_title(
            _with_suffix(f"Dataset Size Ratio (small<{small_thr}^2, medium<{medium_thr}^2)"),
            pad=14,
        )
        ax.grid(True, axis="x", linestyle="--", alpha=0.3)
        # legend 放到图外顶部，避免与标题/第一行重叠
        handles, legend_labels = ax.get_legend_handles_labels()
        fig.legend(
            handles,
            legend_labels,
            ncol=3,
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
        )
        # annotate totals on the right
        if annotate and len(class_names) <= max_annotate_classes:
            for yi, total in enumerate(size_all.sum(axis=1).tolist()):
                ax.text(1.01, yi, f"n={int(total)}", va="center", ha="left", fontsize=8)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
        _save_figure(fig, charts_dir / f"{prefix}_size_ratio.png")
        plt.close(fig)


def _coco_to_yolo_labels(
    coco_api: Any,
) -> Tuple[np.ndarray, list[str], np.ndarray]:
    """
    将 COCO 标注转换为 YOLO 风格的 labels 数组：
    shape = [N, 5]，每行: [cls_idx, x_center, y_center, w, h]，坐标归一化到 [0, 1]。
    """
    dataset = getattr(coco_api, "dataset", None)
    if not isinstance(dataset, dict):
        return np.zeros((0, 5), dtype=np.float32), [], np.zeros((0,), dtype=np.int64)

    categories = dataset.get("categories") or []
    images = dataset.get("images") or []
    annotations = dataset.get("annotations") or []
    if not isinstance(categories, list) or not isinstance(images, list) or not isinstance(annotations, list):
        return np.zeros((0, 5), dtype=np.float32), [], np.zeros((0,), dtype=np.int64)
    if not categories or not annotations:
        return np.zeros((0, 5), dtype=np.float32), [], np.zeros((0,), dtype=np.int64)

    cat_id_to_name: dict[int, str] = {}
    for cat in categories:
        if not isinstance(cat, dict) or "id" not in cat:
            continue
        try:
            cid = int(cat["id"])
        except Exception:
            continue
        name = cat.get("name", f"id_{cid}")
        cat_id_to_name[cid] = str(name)
    if not cat_id_to_name:
        return np.zeros((0, 5), dtype=np.float32), [], np.zeros((0,), dtype=np.int64)

    cat_ids = sorted(cat_id_to_name.keys())
    cat_id_to_idx = {cid: idx for idx, cid in enumerate(cat_ids)}
    names = [cat_id_to_name[cid] for cid in cat_ids]

    image_id_to_wh: dict[int, tuple[float, float]] = {}
    for img in images:
        if not isinstance(img, dict) or "id" not in img:
            continue
        try:
            img_id = int(img["id"])
            w = float(img.get("width", 0))
            h = float(img.get("height", 0))
        except Exception:
            continue
        if w > 0 and h > 0:
            image_id_to_wh[img_id] = (w, h)

    rows: list[list[float]] = []
    image_ids: list[int] = []
    for ann in annotations:
        if not isinstance(ann, dict):
            continue
        try:
            cid = int(ann.get("category_id"))
            img_id = int(ann.get("image_id"))
        except Exception:
            continue
        if cid not in cat_id_to_idx:
            continue
        bbox = ann.get("bbox")
        if not (isinstance(bbox, (list, tuple)) and len(bbox) >= 4):
            continue
        try:
            x0 = float(bbox[0])
            y0 = float(bbox[1])
            bw = float(bbox[2])
            bh = float(bbox[3])
        except Exception:
            continue
        if bw <= 0 or bh <= 0:
            continue
        img_wh = image_id_to_wh.get(img_id)
        if not img_wh:
            continue
        img_w, img_h = img_wh
        if img_w <= 0 or img_h <= 0:
            continue

        x_c = (x0 + bw / 2.0) / img_w
        y_c = (y0 + bh / 2.0) / img_h
        w_n = bw / img_w
        h_n = bh / img_h

        # Clamp to valid normalized range (robust against slightly out-of-bound boxes)
        x_c = float(np.clip(x_c, 0.0, 1.0))
        y_c = float(np.clip(y_c, 0.0, 1.0))
        w_n = float(np.clip(w_n, 0.0, 1.0))
        h_n = float(np.clip(h_n, 0.0, 1.0))

        rows.append([float(cat_id_to_idx[cid]), x_c, y_c, w_n, h_n])
        image_ids.append(img_id)

    labels = np.asarray(rows, dtype=np.float32) if rows else np.zeros((0, 5), dtype=np.float32)
    image_ids_arr = np.asarray(image_ids, dtype=np.int64) if image_ids else np.zeros((0,), dtype=np.int64)
    return labels, names, image_ids_arr


def save_yolo_labels_correlogram(
    coco_api: Any,
    output_dir: Union[str, Path],
    *,
    save_subdir: str = "metric_charts",
    max_points: int = 20000,
    seed: int = 0,
) -> None:
    """
    生成 Ultralytics/YOLO 风格的标签分布可视化：
    - labels.jpg
    - labels_correlogram.jpg

    说明：
    - 输入为 COCO API（需要 coco_api.dataset 至少包含 categories/images/annotations）；
    - 会写入 output_dir/save_subdir/ 下；
    - seaborn 缺失时会跳过 correlogram（与 upstream plot_labels 行为一致）。
    """
    labels, names, _image_ids = _coco_to_yolo_labels(coco_api)
    if labels.size == 0:
        return

    save_dir = Path(output_dir) / str(save_subdir)
    save_dir.mkdir(parents=True, exist_ok=True)

    import warnings

    with warnings.catch_warnings():
        # seaborn/joblib 在部分受限环境下会提示并退回串行，这里静默以免污染训练日志。
        warnings.filterwarnings("ignore", message=r".*joblib will operate in serial mode.*", category=UserWarning)
        from engines.models.mscft.utils.plots import plot_labels  # noqa: WPS433 (lazy import)

    plot_labels(labels.copy(), names=names, save_dir=save_dir, max_points=max_points, seed=seed)


def save_yolo_labels_extra_charts(
    coco_api: Any,
    output_dir: Union[str, Path],
    *,
    save_subdir: str = "metric_charts",
    filename: str = "labels_extra.jpg",
    bins: int = 60,
    max_boxes: int = 200000,
    seed: int = 0,
) -> None:
    """
    生成额外的标签统计图（面向目标检测数据集）：
    - objects/image 分布
    - bbox 面积分布（归一化，log10）
    - bbox aspect ratio 分布（log10(w/h)）
    - log10(aspect) vs log10(area) 2D 直方图
    """
    labels, _names, image_ids = _coco_to_yolo_labels(coco_api)
    if labels.size == 0:
        return

    bins = int(bins)
    bins = 30 if bins <= 0 else bins

    n_boxes = int(labels.shape[0])
    labels_plot = labels
    max_boxes = int(max_boxes) if max_boxes is not None else 0
    if max_boxes > 0 and n_boxes > max_boxes:
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(n_boxes, size=max_boxes, replace=False)
        labels_plot = labels[idx]

    w = labels_plot[:, 3].astype(np.float32)
    h = labels_plot[:, 4].astype(np.float32)
    area = np.clip(w * h, 0.0, 1.0)
    eps = np.float32(1e-9)
    aspect = w / (h + eps)

    log_area = np.log10(area + eps)
    log_aspect = np.log10(aspect + eps)

    # objects per image (use full image_ids, not sampled)
    counts_per_image = np.array([], dtype=np.int32)
    if image_ids.size > 0:
        _uniq, counts = np.unique(image_ids, return_counts=True)
        counts_per_image = counts.astype(np.int32)

    save_dir = Path(output_dir) / str(save_subdir)
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), tight_layout=True)
    ax0, ax1, ax2, ax3 = axes.ravel().tolist()

    # Objects per image
    if counts_per_image.size > 0:
        max_count = int(counts_per_image.max())
        obj_bins = min(max_count, max(10, bins))
        ax0.hist(counts_per_image, bins=obj_bins, color="#4c78a8", alpha=0.9)
        ax0.set_xlabel("Objects / Image")
        ax0.set_ylabel("Images")
        ax0.set_title("Objects per Image")
        ax0.grid(True, axis="y", linestyle="--", alpha=0.3)
    else:
        ax0.text(0.5, 0.5, "N/A", ha="center", va="center")
        ax0.set_axis_off()

    # Area distribution
    ax1.hist(log_area, bins=bins, color="#f58518", alpha=0.9)
    ax1.set_xlabel("log10(area_norm)")
    ax1.set_ylabel("Boxes")
    ax1.set_title("Box Area (Normalized)")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.3)

    # Aspect distribution
    ax2.hist(log_aspect, bins=bins, color="#54a24b", alpha=0.9)
    ax2.set_xlabel("log10(w/h)")
    ax2.set_ylabel("Boxes")
    ax2.set_title("Aspect Ratio")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.3)

    # 2D histogram: aspect vs area
    hist = ax3.hist2d(log_aspect, log_area, bins=bins, cmap="Blues")
    ax3.set_xlabel("log10(w/h)")
    ax3.set_ylabel("log10(area_norm)")
    ax3.set_title("Aspect vs Area")
    try:
        fig.colorbar(hist[3], ax=ax3, fraction=0.046, pad=0.04)
    except Exception:
        pass

    n_images = int(np.unique(image_ids).size) if image_ids.size > 0 else 0
    fig.suptitle(f"Label Stats (boxes={n_boxes:,}, images={n_images:,})", fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    _save_figure(fig, save_dir / filename)
    plt.close(fig)


def save_yolo_labels_per_class_heatmaps(
    coco_api: Any,
    output_dir: Union[str, Path],
    *,
    save_subdir: str = "metric_charts",
    top_k: int = 16,
    bins: int = 50,
    max_boxes_per_class: int = 20000,
    seed: int = 0,
    centers_filename: str = "labels_centers_topk.jpg",
    wh_filename: str = "labels_wh_topk.jpg",
) -> None:
    """
    生成 per-class 热力图（Top-K 类别）：
    - centers: x_center vs y_center 2D 直方图（归一化坐标）
    - wh: width vs height 2D 直方图（归一化尺寸）

    为避免类别很多时输出过大，默认仅绘制 top_k 个实例最多的类别，并对每类采样 max_boxes_per_class。
    """
    labels, names, _image_ids = _coco_to_yolo_labels(coco_api)
    if labels.size == 0:
        return

    top_k = int(top_k)
    if top_k <= 0:
        return
    bins = int(bins)
    bins = 30 if bins <= 0 else bins
    max_boxes_per_class = int(max_boxes_per_class)
    max_boxes_per_class = 0 if max_boxes_per_class < 0 else max_boxes_per_class

    class_idx = labels[:, 0].astype(np.int64)
    n_classes = int(max(int(class_idx.max()) + 1, len(names))) if class_idx.size else len(names)
    counts = np.bincount(class_idx, minlength=max(1, n_classes)).astype(np.int64)
    top = np.argsort(-counts)[:top_k]
    top = [int(i) for i in top.tolist() if counts[int(i)] > 0]
    if not top:
        return

    def _grid(n: int) -> tuple[int, int]:
        cols = int(math.ceil(math.sqrt(n)))
        rows = int(math.ceil(float(n) / float(max(1, cols))))
        return rows, cols

    def _plot_heatmaps(
        x_idx: int,
        y_idx: int,
        *,
        xlabel: str,
        ylabel: str,
        title: str,
        filename: str,
    ) -> None:
        import matplotlib.colors as mcolors

        rows, cols = _grid(len(top))
        fig_w = max(10.0, float(cols) * 3.2)
        fig_h = max(7.0, float(rows) * 3.0)
        fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), squeeze=False)

        rng = np.random.default_rng(int(seed))
        for p, cls in enumerate(top):
            r = p // cols
            c = p % cols
            ax = axes[r][c]
            cls_mask = class_idx == cls
            data = labels[cls_mask]
            if data.size == 0:
                ax.set_axis_off()
                continue
            if max_boxes_per_class > 0 and data.shape[0] > max_boxes_per_class:
                sel = rng.choice(data.shape[0], size=max_boxes_per_class, replace=False)
                data = data[sel]

            xs = data[:, x_idx].astype(np.float32)
            ys = data[:, y_idx].astype(np.float32)
            # 2D hist in [0, 1] x [0, 1]
            h2d, xedges, yedges = np.histogram2d(
                xs,
                ys,
                bins=bins,
                range=[[0.0, 1.0], [0.0, 1.0]],
            )
            h2d = h2d.T  # imshow expects [y, x]
            norm = mcolors.LogNorm(vmin=1.0, vmax=max(1.0, float(h2d.max())))
            im = ax.imshow(
                h2d,
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                origin="lower",
                cmap="Blues",
                norm=norm,
                aspect="equal",
            )
            name = names[cls] if 0 <= cls < len(names) else f"cls_{cls}"
            ax.set_title(f"{name} (n={int(counts[cls])})", fontsize=10)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(False)

        # hide unused axes
        for p in range(len(top), rows * cols):
            r = p // cols
            c = p % cols
            axes[r][c].set_axis_off()

        fig.suptitle(title, fontsize=12)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
        try:
            cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
            cbar.set_label("count (log)")
        except Exception:
            pass

        save_dir = Path(output_dir) / str(save_subdir)
        save_dir.mkdir(parents=True, exist_ok=True)
        _save_figure(fig, save_dir / filename)
        plt.close(fig)

    _plot_heatmaps(
        1,
        2,
        xlabel="x_center",
        ylabel="y_center",
        title=f"Per-class Centers (Top-{len(top)})",
        filename=centers_filename,
    )
    _plot_heatmaps(
        3,
        4,
        xlabel="width",
        ylabel="height",
        title=f"Per-class Width/Height (Top-{len(top)})",
        filename=wh_filename,
    )


def save_yolo_labels_cooccurrence_heatmap(
    coco_api: Any,
    output_dir: Union[str, Path],
    *,
    save_subdir: str = "metric_charts",
    filename: str = "labels_cooccurrence.jpg",
    metric: str = "jaccard",
    top_k: int = 30,
    annotate: bool = False,
    max_annotate_classes: int = 20,
) -> None:
    """
    生成类别共现热力图（基于“同一张图像中出现过该类别”）：
    - metric='count'：共现的图像数
    - metric='jaccard'：Jaccard(i,j)=co(i,j)/(|i|+|j|-co(i,j))
    - metric='p(j|i)'：条件概率 co(i,j)/|i|（行归一化）
    """
    labels, names, image_ids = _coco_to_yolo_labels(coco_api)
    if labels.size == 0 or image_ids.size == 0:
        return

    top_k = int(top_k)
    top_k = 1 if top_k <= 0 else top_k

    class_idx = labels[:, 0].astype(np.int64)
    n_classes = int(max(int(class_idx.max()) + 1, len(names))) if class_idx.size else len(names)
    counts = np.bincount(class_idx, minlength=max(1, n_classes)).astype(np.int64)
    top = np.argsort(-counts)[:top_k]
    top = [int(i) for i in top.tolist() if counts[int(i)] > 0]
    if not top:
        return

    selected = set(top)
    idx_map = {cls: i for i, cls in enumerate(top)}
    m = len(top)
    co = np.zeros((m, m), dtype=np.int64)

    # group by image_id, build class set per image
    order = np.argsort(image_ids)
    image_ids_sorted = image_ids[order]
    cls_sorted = class_idx[order]
    uniq_ids, starts, lens = np.unique(image_ids_sorted, return_index=True, return_counts=True)
    for start, length in zip(starts.tolist(), lens.tolist()):
        cls_set = set(np.unique(cls_sorted[start : start + length]).tolist())
        cls_set = [c for c in cls_set if c in selected]
        if not cls_set:
            continue
        for ci in cls_set:
            i = idx_map[ci]
            for cj in cls_set:
                j = idx_map[cj]
                co[i, j] += 1

    metric_key = str(metric or "jaccard").strip().lower()
    values: np.ndarray
    cmap = "Blues"
    if metric_key == "count":
        values = co.astype(np.float32)
        cmap = "Blues"
    elif metric_key in {"p(j|i)", "p", "cond", "conditional"}:
        diag = np.diag(co).astype(np.float32)
        denom = np.clip(diag[:, None], 1.0, None)
        values = co.astype(np.float32) / denom
        cmap = "viridis"
    else:  # default jaccard
        diag = np.diag(co).astype(np.float32)
        denom = (diag[:, None] + diag[None, :] - co.astype(np.float32))
        denom = np.clip(denom, 1.0, None)
        values = co.astype(np.float32) / denom
        cmap = "viridis"
        metric_key = "jaccard"

    class_names = [names[c] if 0 <= c < len(names) else f"cls_{c}" for c in top]

    fig_w = max(8.0, float(m) * 0.35)
    fig_h = max(7.0, float(m) * 0.35)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(values, cmap=cmap, vmin=float(values.min()), vmax=float(values.max()))
    ax.set_xticks(np.arange(m))
    ax.set_yticks(np.arange(m))
    ax.set_xticklabels(class_names, rotation=90, fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Class")
    ax.set_ylabel("Class")
    ax.set_title(f"Class Co-occurrence ({metric_key}, top={m})")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.grid(False)

    if annotate and m <= int(max_annotate_classes):
        for i in range(m):
            for j in range(m):
                val = float(values[i, j])
                text = f"{val:.2f}" if metric_key != "count" else f"{int(co[i, j])}"
                ax.text(j, i, text, ha="center", va="center", fontsize=6, color="black")

    fig.tight_layout()
    save_dir = Path(output_dir) / str(save_subdir)
    save_dir.mkdir(parents=True, exist_ok=True)
    _save_figure(fig, save_dir / filename)
    plt.close(fig)


def save_inference_profile_chart(
    profile: dict,
    output_dir: Union[str, Path],
    *,
    prefix: str = "val",
    save_subdir: str = "metric_charts",
):
    """
    保存推理 profile 条形图（FPS/latency/params/flops），并在条形上标注数值。
    """
    if not isinstance(profile, dict):
        return
    fps = _sanitize_numeric(profile.get("fps"))
    latency_ms = _sanitize_numeric(profile.get("latency_ms"))
    params_m = _sanitize_numeric(profile.get("params_m"))
    gflops = _sanitize_numeric(profile.get("gflops"))
    device = str(profile.get("device", ""))

    labels = []
    values = []
    ann = []
    if fps is not None:
        labels.append("FPS")
        values.append(float(fps))
        ann.append(f"{fps:.2f}")
    if latency_ms is not None:
        labels.append("Latency(ms/img)")
        values.append(float(latency_ms))
        ann.append(f"{latency_ms:.2f}")
    if params_m is not None:
        labels.append("Params(M)")
        values.append(float(params_m))
        ann.append(f"{params_m:.2f}")
    if gflops is not None:
        labels.append("GFLOPs/img")
        values.append(float(gflops))
        ann.append(f"{gflops:.2f}")

    if not values:
        return

    charts_dir = Path(output_dir) / str(save_subdir)
    charts_dir.mkdir(parents=True, exist_ok=True)
    prefix = prefix.replace(" ", "_") or "metrics"

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
    bars = ax.bar(labels, values, color=colors[: len(values)])
    ax.set_title(f"Inference Profile{(' - ' + device) if device else ''}")
    for bar, text in zip(bars, ann):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            float(bar.get_height()) * 1.01,
            text,
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    _save_figure(fig, charts_dir / f"{prefix}_inference_profile.png")
    plt.close(fig)


def save_detection_visual_samples(
    *,
    image_path: Union[str, Path],
    out_path: Union[str, Path],
    msi_channel: int | None = None,
    gt_boxes_xyxy: np.ndarray | None = None,
    gt_labels: Sequence[int] | None = None,
    pred_boxes_xyxy: np.ndarray | None = None,
    pred_labels: Sequence[int] | None = None,
    pred_scores: Sequence[float] | None = None,
    gt_masks: np.ndarray | None = None,
    pred_masks: np.ndarray | None = None,
    class_id_to_name: dict[int, str] | None = None,
    score_threshold: float = 0.3,
    max_dets: int = 50,
    mask_threshold: float = 0.5,
    mask_alpha: int = 90,
) -> None:
    """
    保存单张样例可视化（GT + Pred 叠加）。
    - GT: 绿色框
    - Pred: 红色框（带 score）
    - 若提供 masks：GT 为绿色半透明、Pred 为红色半透明

    多光谱（*.tif/*.tiff）可通过 `msi_channel` 指定“第 N 个通道”（1-based）作为灰度底图。
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return

    image_path = Path(image_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not image_path.is_file():
        return

    def _scale_to_u8(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr)
        if arr.size == 0:
            return np.zeros((1, 1), dtype=np.uint8)
        arr_f = arr.astype(np.float32, copy=False)
        finite = np.isfinite(arr_f)
        if not finite.any():
            return np.zeros(arr_f.shape, dtype=np.uint8)
        vmin = float(np.min(arr_f[finite]))
        vmax = float(np.max(arr_f[finite]))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            return np.zeros(arr_f.shape, dtype=np.uint8)
        scaled = (arr_f - vmin) / (vmax - vmin)
        return (scaled.clip(0.0, 1.0) * 255.0).astype(np.uint8)

    def _load_for_vis(path: Path, *, msi_channel_1based: int | None) -> "Image.Image | None":
        # 默认走 PIL；若指定 msi_channel 或 PIL 失败，则尝试 tifffile。
        if msi_channel_1based is None:
            try:
                return Image.open(path).convert("RGB")
            except Exception:
                pass

        try:
            import tifffile  # type: ignore
        except Exception:
            return None

        try:
            arr = tifffile.imread(str(path))
        except Exception:
            return None

        if arr.ndim == 2:
            arr = arr[..., None]
        if arr.ndim != 3:
            return None

        # 判断通道维：优先识别“通道数较小”的那个维度
        c_first = int(arr.shape[0])
        c_last = int(arr.shape[-1])
        if c_first <= 32 and c_first < c_last:
            channel_first = True
            c = c_first
        elif c_last <= 32 and c_last < c_first:
            channel_first = False
            c = c_last
        else:
            channel_first = True
            c = c_first

        if msi_channel_1based is not None:
            idx0 = max(0, min(c - 1, int(msi_channel_1based) - 1))
            gray = arr[idx0] if channel_first else arr[..., idx0]
            gray_u8 = _scale_to_u8(gray)
            try:
                return Image.fromarray(gray_u8, mode="L").convert("RGB")
            except Exception:
                return None

        # 兜底：按前三个通道合成 RGB
        if c <= 0:
            return None
        take = min(3, c)
        if channel_first:
            rgb = np.transpose(arr[:take], (1, 2, 0))
        else:
            rgb = arr[..., :take]
        if rgb.ndim != 3:
            return None
        if rgb.shape[-1] == 1:
            rgb = np.repeat(rgb, 3, axis=-1)
        elif rgb.shape[-1] == 2:
            rgb = np.concatenate([rgb, rgb[..., :1]], axis=-1)
        rgb_u8 = _scale_to_u8(rgb)
        try:
            return Image.fromarray(rgb_u8, mode="RGB")
        except Exception:
            return None

    im = _load_for_vis(image_path, msi_channel_1based=msi_channel)
    if im is None:
        return

    # masks（如有）先叠加到底图上，再画框与文字
    def _as_mask_stack(mask_like: np.ndarray | None) -> np.ndarray | None:
        if mask_like is None:
            return None
        try:
            arr = np.asarray(mask_like)
        except Exception:
            return None
        if arr.ndim == 4 and arr.shape[1] == 1:
            arr = arr[:, 0]
        if arr.ndim == 2:
            arr = arr[None, ...]
        if arr.ndim != 3:
            return None
        return arr

    def _overlay_masks(
        base: "Image.Image",
        masks: np.ndarray,
        *,
        color_rgb: tuple[int, int, int],
        alpha: int,
        threshold: float,
    ) -> "Image.Image":
        if masks.size == 0:
            return base
        base_rgba = base.convert("RGBA")
        w, h = base_rgba.size
        masks_u8 = masks
        for i in range(int(masks_u8.shape[0])):
            m = masks_u8[i]
            if m.ndim != 2:
                continue
            if m.dtype == np.bool_:
                mb = m
            elif np.issubdtype(m.dtype, np.floating):
                mb = m.astype(np.float32, copy=False) >= float(threshold)
            else:
                mb = m.astype(np.int64, copy=False) > 0
            if mb.shape != (h, w):
                try:
                    m_img = Image.fromarray((mb.astype(np.uint8) * 255), mode="L").resize((w, h), resample=Image.NEAREST)
                    mb = np.asarray(m_img) > 0
                except Exception:
                    continue
            if not bool(mb.any()):
                continue
            overlay = np.zeros((h, w, 4), dtype=np.uint8)
            overlay[mb, 0] = int(color_rgb[0])
            overlay[mb, 1] = int(color_rgb[1])
            overlay[mb, 2] = int(color_rgb[2])
            overlay[mb, 3] = int(max(0, min(255, alpha)))
            try:
                overlay_im = Image.fromarray(overlay, mode="RGBA")
                base_rgba = Image.alpha_composite(base_rgba, overlay_im)
            except Exception:
                continue
        return base_rgba.convert("RGB")

    gt_mask_stack = _as_mask_stack(gt_masks)
    if gt_mask_stack is not None:
        im = _overlay_masks(im, gt_mask_stack, color_rgb=(0, 200, 0), alpha=mask_alpha, threshold=mask_threshold)

    pred_mask_stack = _as_mask_stack(pred_masks)
    if pred_mask_stack is not None:
        # 与 pred 过滤逻辑对齐：仅叠加 score>=threshold 的 top-k
        keep_idx: list[int] = []
        if pred_scores is not None:
            try:
                scores = [float(s) for s in pred_scores]
                keep_idx = [i for i, s in enumerate(scores) if s >= float(score_threshold)]
                keep_idx.sort(key=lambda i: -scores[i])
                keep_idx = keep_idx[: max(0, int(max_dets))]
            except Exception:
                keep_idx = []
        if keep_idx:
            try:
                im = _overlay_masks(
                    im,
                    pred_mask_stack[np.asarray(keep_idx, dtype=int)],
                    color_rgb=(220, 0, 0),
                    alpha=mask_alpha,
                    threshold=mask_threshold,
                )
            except Exception:
                pass

    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    def _label_text(label_id: int) -> str:
        if class_id_to_name and int(label_id) in class_id_to_name:
            return str(class_id_to_name[int(label_id)])
        return str(int(label_id))

    # GT
    if gt_boxes_xyxy is not None and gt_labels is not None:
        try:
            gt_boxes = np.asarray(gt_boxes_xyxy, dtype=float).reshape(-1, 4)
        except Exception:
            gt_boxes = None
        if gt_boxes is not None:
            for box, lab in zip(gt_boxes, list(gt_labels)):
                x1, y1, x2, y2 = [float(x) for x in box.tolist()]
                draw.rectangle([x1, y1, x2, y2], outline=(0, 200, 0), width=2)
                text = f"GT:{_label_text(int(lab))}"
                if font is not None:
                    draw.text((x1 + 2, y1 + 2), text, fill=(0, 200, 0), font=font)
                else:
                    draw.text((x1 + 2, y1 + 2), text, fill=(0, 200, 0))

    # Pred
    if pred_boxes_xyxy is not None and pred_labels is not None and pred_scores is not None:
        try:
            pred_boxes = np.asarray(pred_boxes_xyxy, dtype=float).reshape(-1, 4)
            scores = [float(s) for s in pred_scores]
            labels = [int(x) for x in pred_labels]
        except Exception:
            pred_boxes = None
            scores, labels = [], []
        if pred_boxes is not None and scores and labels:
            items = [(i, scores[i]) for i in range(min(len(scores), pred_boxes.shape[0], len(labels)))]
            items = [(i, s) for i, s in items if s >= float(score_threshold)]
            items.sort(key=lambda x: -x[1])
            items = items[: max(0, int(max_dets))]
            for i, s in items:
                x1, y1, x2, y2 = [float(x) for x in pred_boxes[i].tolist()]
                draw.rectangle([x1, y1, x2, y2], outline=(220, 0, 0), width=2)
                text = f"{_label_text(labels[i])}:{s:.2f}"
                if font is not None:
                    draw.text((x1 + 2, max(0.0, y1 - 12)), text, fill=(220, 0, 0), font=font)
                else:
                    draw.text((x1 + 2, max(0.0, y1 - 12)), text, fill=(220, 0, 0))

    try:
        im.save(out_path)
    except Exception:
        return


def _extract_class_map(results, *, include_all: bool = False):
    class_map = results.get("class_map", [])
    if isinstance(class_map, dict):
        for value in class_map.values():
            if isinstance(value, list):
                class_map = value
                break
        else:
            class_map = []
    excluded = {None} if include_all else {"all", None}
    return [entry for entry in class_map if entry.get("class") not in excluded]


def save_detection_per_class_curves_from_coco_eval(
    coco_eval,
    output_dir: Union[str, Path],
    prefix: str = "val",
    *,
    save_subdir: str = "metric_charts",
):
    """
    基于 COCOeval 的 eval['precision']/eval['scores'] 保存“每个类别一条线”的曲线图：
    - PR 曲线（Recall 为横轴）
    - Precision-Confidence 曲线（Confidence 为横轴）
    - Recall-Confidence 曲线（Confidence 为横轴）
    """
    try:
        P = coco_eval.eval.get("precision")
        S = coco_eval.eval.get("scores")
        rec_thrs = coco_eval.params.recThrs
        iou_thrs = coco_eval.params.iouThrs
        cat_ids = coco_eval.params.catIds
    except Exception:
        return

    if P is None or S is None or rec_thrs is None or iou_thrs is None or not cat_ids:
        return

    try:
        iou50_idx = int(np.argwhere(np.isclose(iou_thrs, 0.50)))
    except Exception:
        return

    area_idx, maxdet_idx = 0, 2
    try:
        prec_raw = P[iou50_idx, :, :, area_idx, maxdet_idx].astype(float)  # [R, K]
        score_raw = S[iou50_idx, :, :, area_idx, maxdet_idx].astype(float)  # [R, K]
    except Exception:
        return

    prec = prec_raw.copy()
    prec[prec < 0] = np.nan
    score = score_raw.copy()
    score[prec_raw < 0] = np.nan

    try:
        cat_id_to_name = {c["id"]: c["name"] for c in coco_eval.cocoGt.loadCats(cat_ids)}
    except Exception:
        cat_id_to_name = {int(cid): f"class_{idx}" for idx, cid in enumerate(cat_ids)}

    charts_dir = Path(output_dir) / str(save_subdir)
    charts_dir.mkdir(parents=True, exist_ok=True)
    prefix = prefix.replace(" ", "_") or "metrics"

    def _prepare_sorted_curve(xs, ys):
        x_arr, y_arr = _prepare_curve(xs, ys)
        if x_arr.size < 2 or y_arr.size < 2:
            return np.array([]), np.array([])
        order = np.argsort(x_arr)
        return x_arr[order], y_arr[order]

    def _save_multi_line(lines, xlabel, ylabel, title, filename, *, xlim=(0.0, 1.0), ylim=(0.0, 1.0)):
        if not lines:
            return
        fig, ax = plt.subplots(figsize=(14, 6))
        for name, x_arr, y_arr in lines:
            if x_arr.size < 2 or y_arr.size < 2:
                continue
            ax.plot(x_arr, y_arr, linewidth=1.5, label=name)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xlim(float(xlim[0]), float(xlim[1]))
        ax.set_ylim(float(ylim[0]), float(ylim[1]))
        ax.grid(True, linestyle="--", alpha=0.4)
        # 类别很多时 legend 会很大：尽量放到图外，避免遮挡曲线。
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ncol = 1 if len(labels) <= 12 else (2 if len(labels) <= 30 else 3)
            ax.legend(
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                fontsize=7 if len(labels) > 30 else 8,
                ncol=ncol,
                frameon=False,
            )
        fig.tight_layout()
        _save_figure(fig, charts_dir / f"{prefix}_{filename}")
        plt.close(fig)

    pr_lines = []
    prec_conf_lines = []
    rec_conf_lines = []

    for k, cid in enumerate(cat_ids):
        class_name = cat_id_to_name.get(int(cid), f"class_{k}")
        prec_k = prec[:, k]
        conf_k = score[:, k]

        x_pr, y_pr = _prepare_sorted_curve(rec_thrs, prec_k)
        if x_pr.size >= 2:
            pr_lines.append((class_name, x_pr, y_pr))

        x_pc, y_pc = _prepare_sorted_curve(conf_k, prec_k)
        if x_pc.size >= 2:
            prec_conf_lines.append((class_name, x_pc, y_pc))

        x_rc, y_rc = _prepare_sorted_curve(conf_k, rec_thrs)
        if x_rc.size >= 2:
            rec_conf_lines.append((class_name, x_rc, y_rc))

    _save_multi_line(pr_lines, "Recall", "Precision", "Per-Class PR Curves", "pr_curve_per_class.png")
    _save_multi_line(
        prec_conf_lines,
        "Confidence",
        "Precision",
        "Per-Class Precision-Confidence Curves",
        "precision_curve_per_class.png",
    )
    _save_multi_line(
        rec_conf_lines,
        "Confidence",
        "Recall",
        "Per-Class Recall-Confidence Curves",
        "recall_curve_per_class.png",
    )


def save_detection_combined_curves(
    results: dict,
    coco_eval,
    output_dir: Union[str, Path],
    *,
    prefix: str = "val",
    all_class_name: str = "all_class",
    save_subdir: str = "metric_charts",
) -> None:
    """
    将 overall 曲线（results['curves']）与 per-class 曲线（COCOeval）合并到同一张图。

    - 各类别：细实线
    - all_class：黑色虚线加粗（用于与各类别区分）

    生成文件（写到 output_dir/<save_subdir>）：
    - {prefix}_pr_curve.png
    - {prefix}_precision_curve.png
    - {prefix}_recall_curve.png
    """

    if not isinstance(results, dict):
        return

    curves = results.get("curves", {}) or {}
    recall_curve = curves.get("recall", [])
    precision_curve = curves.get("precision", [])
    confidence_curve = curves.get("confidence", [])

    def _prepare_sorted_curve(xs, ys):
        x_arr, y_arr = _prepare_curve(xs, ys)
        if x_arr.size < 2 or y_arr.size < 2:
            return np.array([]), np.array([])
        order = np.argsort(x_arr)
        return x_arr[order], y_arr[order]

    all_pr_x, all_pr_y = _prepare_sorted_curve(recall_curve, precision_curve)
    all_pc_x, all_pc_y = _prepare_sorted_curve(confidence_curve, precision_curve)
    all_rc_x, all_rc_y = _prepare_sorted_curve(confidence_curve, recall_curve)

    try:
        P = coco_eval.eval.get("precision")
        S = coco_eval.eval.get("scores")
        rec_thrs = coco_eval.params.recThrs
        iou_thrs = coco_eval.params.iouThrs
        cat_ids = coco_eval.params.catIds
    except Exception:
        return

    if P is None or S is None or rec_thrs is None or iou_thrs is None or not cat_ids:
        return

    try:
        iou50_idx = int(np.argwhere(np.isclose(iou_thrs, 0.50)))
    except Exception:
        return

    area_idx, maxdet_idx = 0, 2
    try:
        prec_raw = P[iou50_idx, :, :, area_idx, maxdet_idx].astype(float)  # [R, K]
        score_raw = S[iou50_idx, :, :, area_idx, maxdet_idx].astype(float)  # [R, K]
    except Exception:
        return

    prec = prec_raw.copy()
    prec[prec < 0] = np.nan
    score = score_raw.copy()
    score[prec_raw < 0] = np.nan

    try:
        cat_id_to_name = {c["id"]: c["name"] for c in coco_eval.cocoGt.loadCats(cat_ids)}
    except Exception:
        cat_id_to_name = {int(cid): f"class_{idx}" for idx, cid in enumerate(cat_ids)}

    charts_dir = Path(output_dir) / str(save_subdir)
    charts_dir.mkdir(parents=True, exist_ok=True)
    prefix = prefix.replace(" ", "_") or "metrics"

    def _save_combined(
        lines,
        *,
        xlabel: str,
        ylabel: str,
        title: str,
        filename: str,
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
        all_x=None,
        all_y=None,
    ) -> None:
        if not lines and (all_x is None or all_y is None):
            return
        fig, ax = plt.subplots(figsize=(14, 6))
        for name, x_arr, y_arr in lines:
            if x_arr.size < 2 or y_arr.size < 2:
                continue
            ax.plot(x_arr, y_arr, linewidth=1.4, alpha=0.85, label=name)
        if all_x is not None and all_y is not None and all_x.size >= 2 and all_y.size >= 2:
            ax.plot(
                all_x,
                all_y,
                linewidth=2.6,
                linestyle="--",
                color="black",
                label=all_class_name,
                zorder=10,
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xlim(float(xlim[0]), float(xlim[1]))
        ax.set_ylim(float(ylim[0]), float(ylim[1]))
        ax.grid(True, linestyle="--", alpha=0.4)
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ncol = 1 if len(labels) <= 12 else (2 if len(labels) <= 30 else 3)
            ax.legend(
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                fontsize=7 if len(labels) > 30 else 8,
                ncol=ncol,
                frameon=False,
            )
        fig.tight_layout()
        _save_figure(fig, charts_dir / f"{prefix}_{filename}")
        plt.close(fig)

    pr_lines = []
    prec_conf_lines = []
    rec_conf_lines = []

    for k, cid in enumerate(cat_ids):
        class_name = cat_id_to_name.get(int(cid), f"class_{k}")
        prec_k = prec[:, k]
        conf_k = score[:, k]

        x_pr, y_pr = _prepare_sorted_curve(rec_thrs, prec_k)
        if x_pr.size >= 2:
            pr_lines.append((class_name, x_pr, y_pr))

        x_pc, y_pc = _prepare_sorted_curve(conf_k, prec_k)
        if x_pc.size >= 2:
            prec_conf_lines.append((class_name, x_pc, y_pc))

        x_rc, y_rc = _prepare_sorted_curve(conf_k, rec_thrs)
        if x_rc.size >= 2:
            rec_conf_lines.append((class_name, x_rc, y_rc))

    _save_combined(
        pr_lines,
        xlabel="Recall",
        ylabel="Precision",
        title="PR Curves",
        filename="pr_curve.png",
        all_x=all_pr_x,
        all_y=all_pr_y,
    )
    _save_combined(
        prec_conf_lines,
        xlabel="Confidence",
        ylabel="Precision",
        title="Precision Curve",
        filename="precision_curve.png",
        all_x=all_pc_x,
        all_y=all_pc_y,
    )
    _save_combined(
        rec_conf_lines,
        xlabel="Confidence",
        ylabel="Recall",
        title="Recall Curve",
        filename="recall_curve.png",
        all_x=all_rc_x,
        all_y=all_rc_y,
    )


def save_detection_metric_charts(
    results: dict,
    output_dir: Union[str, Path],
    prefix: str = "val",
    *,
    chart_cfg: dict | None = None,
    save_subdir: str = "metric_charts",
):
    """
    保存检测指标可视化图像（PR/F1 曲线、整体指标条形图、各类精确率/召回率/AP）。
    """
    if not isinstance(results, dict):
        return

    chart_cfg = chart_cfg or {}

    def _flag(name: str, default: bool = True) -> bool:
        value = chart_cfg.get(name, default)
        return bool(value)

    annotate_bars = _flag("annotate_bars", True)
    max_annotate_classes = int(chart_cfg.get("max_annotate_classes", 20))

    curves = results.get("curves", {}) or {}
    recall_curve = curves.get("recall")
    precision_curve = curves.get("precision")
    confidence_curve = curves.get("confidence")
    f1_curve = curves.get("f1")

    precision = _sanitize_numeric(results.get("precision"))
    recall = _sanitize_numeric(results.get("recall"))
    f1 = _sanitize_numeric(results.get("f1"))
    map_50 = _sanitize_numeric(results.get("map"))
    map_5095 = _sanitize_numeric(results.get("map@50:95") or results.get("map50_95"))
    score_thr = _sanitize_numeric(results.get("score_threshold"))

    charts_dir = Path(output_dir) / str(save_subdir)
    charts_dir.mkdir(parents=True, exist_ok=True)
    prefix = prefix.replace(" ", "_") or "metrics"

    prune_disabled = _flag("prune_disabled", False)
    if prune_disabled:
        enabled_map = {
            "overall_metrics.png": _flag("overall_metrics", True),
            "per_class_precision_recall.png": _flag("per_class_precision_recall", True),
            "per_class_ap.png": _flag("per_class_ap", True),
            "per_class_metrics.png": _flag("per_class_metrics", True),
        }
        if not _flag("curves", True):
            enabled_map.update(
                {
                    "pr_curve.png": False,
                    "precision_curve.png": False,
                    "recall_curve.png": False,
                    "f1_curve.png": False,
                }
            )
        for filename, enabled in enabled_map.items():
            if enabled:
                continue
            path = charts_dir / f"{prefix}_{filename}"
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            except Exception:
                # 仅清理失败不应影响训练流程
                pass

    def _save_line_plot(x, y, xlabel, ylabel, title, filename, *, xlim=(0.0, 1.0), ylim=(0.0, 1.0)):
        if x.size < 2 or y.size < 2:
            return
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(x, y, linewidth=2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if xlim is not None:
            ax.set_xlim(float(xlim[0]), float(xlim[1]))
        if ylim is not None:
            ax.set_ylim(float(ylim[0]), float(ylim[1]))
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        _save_figure(fig, charts_dir / f"{prefix}_{filename}")
        plt.close(fig)

    if _flag("curves", True):
        pr_x, pr_y = _prepare_curve(recall_curve, precision_curve)
        _save_line_plot(pr_x, pr_y, "Recall", "Precision", "Precision-Recall Curve", "pr_curve.png")

        prec_x, prec_y = _prepare_curve(confidence_curve, precision_curve)
        _save_line_plot(prec_x, prec_y, "Confidence", "Precision", "Precision Curve", "precision_curve.png")

        rec_x, rec_y = _prepare_curve(confidence_curve, recall_curve)
        _save_line_plot(rec_x, rec_y, "Confidence", "Recall", "Recall Curve", "recall_curve.png")

        f1_x, f1_y = _prepare_curve(confidence_curve, f1_curve)
        _save_line_plot(f1_x, f1_y, "Confidence", "F1 Score", "F1 Curve", "f1_curve.png")

    def _normalize_overall_key(key: str) -> str:
        key = str(key or "").strip().lower()
        aliases = {
            "p": "precision",
            "prec": "precision",
            "precision": "precision",
            "r": "recall",
            "rec": "recall",
            "recall": "recall",
            "f1": "f1",
            "f1_score": "f1",
            "f1-score": "f1",
            "map": "map50",
            "map50": "map50",
            "map@0.5": "map50",
            "map50_95": "map50_95",
            "map@50:95": "map50_95",
            "map5095": "map50_95",
            "ap_s": "ap_s",
            "ap_m": "ap_m",
            "ap_l": "ap_l",
            "score_thr": "score_threshold",
            "score_threshold": "score_threshold",
            "scorethr": "score_threshold",
        }
        return aliases.get(key, key)

    include_keys = chart_cfg.get("overall_metrics_include") if isinstance(chart_cfg, dict) else None
    if include_keys is None and isinstance(chart_cfg, dict):
        # Backward-compatible alias.
        include_keys = chart_cfg.get("overall_metrics_keys")
    include_set = None
    if include_keys:
        try:
            include_list = list(include_keys) if isinstance(include_keys, (list, tuple)) else [include_keys]
        except Exception:
            include_list = [include_keys]
        normalized = [_normalize_overall_key(k) for k in include_list]
        include_set = {k for k in normalized if k}

    metrics_labels = []
    metrics_values = []
    # key, label, value
    candidates = [
        ("precision", "Precision", precision),
        ("recall", "Recall", recall),
        ("f1", "F1 Score", f1),
        ("map50", "mAP@0.5", map_50),
        ("map50_95", "mAP@0.5:0.95", map_5095),
        ("ap_s", "AP_s", _sanitize_numeric(results.get("map_s"))),
        ("ap_m", "AP_m", _sanitize_numeric(results.get("map_m"))),
        ("ap_l", "AP_l", _sanitize_numeric(results.get("map_l"))),
        ("score_threshold", "Score Thr", score_thr),
    ]
    for key, label, value in candidates:
        if include_set is not None and key not in include_set:
            continue
        if value is not None:
            metrics_labels.append(label)
            metrics_values.append(value)

    if metrics_values and _flag("overall_metrics", True):
        # 横向条形图：避免指标项过多时横坐标拥挤
        fig_h = max(4.8, 0.45 * len(metrics_labels))
        fig, ax = plt.subplots(figsize=(8.5, fig_h))
        y = np.arange(len(metrics_labels))
        bars = ax.barh(y, metrics_values, color="#1f77b4")
        ax.set_yticks(y)
        ax.set_yticklabels(metrics_labels)
        ax.invert_yaxis()
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("Value")
        ax.set_title("Overall Metrics")
        ax.grid(True, axis="x", linestyle="--", alpha=0.3)
        if annotate_bars and len(metrics_labels) <= 30:
            _annotate_barh(ax, bars, fmt="{:.3f}", dx=0.01, fontsize=9)
        fig.tight_layout()
        _save_figure(fig, charts_dir / f"{prefix}_overall_metrics.png")
        plt.close(fig)

    include_all_in_per_class = _flag("include_all_in_per_class", True)
    per_class = _extract_class_map(results, include_all=include_all_in_per_class)
    if per_class:
        # 将 all 放到最上方，便于对照整体与各类别。
        ordered = sorted(per_class, key=lambda x: 0 if x.get("class") == "all" else 1)
        class_names = [entry.get("class", f"class_{idx}") for idx, entry in enumerate(ordered)]
        class_precision = [_sanitize_numeric(entry.get("precision")) or 0.0 for entry in ordered]
        class_recall = [_sanitize_numeric(entry.get("recall")) or 0.0 for entry in ordered]
        class_f1 = []
        for idx, entry in enumerate(ordered):
            f1_value = _sanitize_numeric(entry.get("f1"))
            if f1_value is not None:
                class_f1.append(f1_value)
                continue
            denom = class_precision[idx] + class_recall[idx]
            class_f1.append((2.0 * class_precision[idx] * class_recall[idx] / denom) if denom > 0 else 0.0)
        class_map_5095 = [_sanitize_numeric(entry.get("map@50:95")) or 0.0 for entry in ordered]
        class_map_50 = [_sanitize_numeric(entry.get("map@50")) or 0.0 for entry in ordered]

        indices = np.arange(len(class_names))
        height = 0.4

        if _flag("per_class_precision_recall", True):
            fig, ax = plt.subplots(figsize=(10, max(6, len(class_names) * 0.35)))
            bars_p = ax.barh(indices - height / 2, class_precision, height, label="Precision")
            bars_r = ax.barh(indices + height / 2, class_recall, height, label="Recall")
            ax.set_xlabel("Value")
            ax.set_title("Per-Class Precision/Recall")
            ax.set_yticks(indices)
            ax.set_yticklabels(class_names)
            ax.set_xlim(0, 1)
            ax.legend()
            if annotate_bars and len(class_names) <= max_annotate_classes:
                _annotate_barh(ax, bars_p, fmt="{:.3f}", dx=0.01, fontsize=7)
                _annotate_barh(ax, bars_r, fmt="{:.3f}", dx=0.01, fontsize=7)
            fig.tight_layout()
            _save_figure(fig, charts_dir / f"{prefix}_per_class_precision_recall.png")
            plt.close(fig)

        if _flag("per_class_ap", True):
            fig, ax = plt.subplots(figsize=(10, max(6, len(class_names) * 0.35)))
            bars_5095 = ax.barh(indices - height / 2, class_map_5095, height, label="mAP@0.5:0.95")
            bars_50 = ax.barh(indices + height / 2, class_map_50, height, label="mAP@0.5")
            ax.set_xlabel("AP")
            ax.set_title("Per-Class AP")
            ax.set_yticks(indices)
            ax.set_yticklabels(class_names)
            ax.set_xlim(0, 1)
            ax.legend()
            if annotate_bars and len(class_names) <= max_annotate_classes:
                _annotate_barh(ax, bars_5095, fmt="{:.3f}", dx=0.01, fontsize=7)
                _annotate_barh(ax, bars_50, fmt="{:.3f}", dx=0.01, fontsize=7)
            fig.tight_layout()
            _save_figure(fig, charts_dir / f"{prefix}_per_class_ap.png")
            plt.close(fig)

        # 一张图汇总各类别多指标：Precision/Recall/F1/mAP@0.5/mAP@0.5:0.95
        metric_series = [
            ("Precision", class_precision),
            ("Recall", class_recall),
            ("F1", class_f1),
            ("mAP@0.5", class_map_50),
            ("mAP@0.5:0.95", class_map_5095),
        ]
        n_metrics = len(metric_series)
        group_height = 0.8
        bar_h = group_height / float(n_metrics)
        offsets = (np.arange(n_metrics) - (n_metrics - 1) / 2.0) * bar_h

        if _flag("per_class_metrics", True):
            fig, ax = plt.subplots(figsize=(12, max(6, len(class_names) * 0.35)))
            bar_containers = []
            for m_idx, (name, values) in enumerate(metric_series):
                bar_containers.append(ax.barh(indices + offsets[m_idx], values, bar_h, label=name))
            ax.set_xlabel("Value")
            ax.set_title("Per-Class Metrics")
            ax.set_yticks(indices)
            ax.set_yticklabels(class_names)
            ax.set_xlim(0.0, 1.0)
            ax.legend(ncol=2)
            if annotate_bars and len(class_names) <= max_annotate_classes:
                for bars in bar_containers:
                    _annotate_barh(ax, bars, fmt="{:.3f}", dx=0.01, fontsize=6)
            fig.tight_layout()
            _save_figure(fig, charts_dir / f"{prefix}_per_class_metrics.png")
            plt.close(fig)



def save_detection_confusion_matrix(
    confusion: np.ndarray,
    class_names: Sequence[str],
    output_dir: Union[str, Path],
    *,
    prefix: str = "val",
    normalize: bool = True,
    save_subdir: str = "metric_charts",
):
    """
    保存目标检测混淆矩阵图（支持含 background 的 (K+1)x(K+1)）。

    约定：
    - 行为 GT 类别，列为预测类别；
    - 若包含 background，建议放在最后一个类别。
    """
    if confusion is None:
        return
    confusion = np.asarray(confusion)
    if confusion.ndim != 2 or confusion.shape[0] != confusion.shape[1]:
        return
    n = int(confusion.shape[0])
    names = list(class_names or [])
    if len(names) != n:
        names = [f"class_{i}" for i in range(n)]

    matrix = confusion.astype(np.float32)
    if normalize:
        row_sum = matrix.sum(axis=1, keepdims=True)
        row_sum[row_sum <= 0] = 1.0
        matrix = matrix / row_sum

    charts_dir = Path(output_dir) / str(save_subdir)
    charts_dir.mkdir(parents=True, exist_ok=True)
    prefix = prefix.replace(" ", "_") or "metrics"

    fig_w = max(8.0, min(24.0, 0.55 * n))
    fig_h = max(7.0, min(24.0, 0.55 * n))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(matrix, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0 if normalize else None)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(n),
        yticks=np.arange(n),
        xticklabels=names,
        yticklabels=names,
        ylabel="GT",
        xlabel="Pred",
        title="Confusion Matrix (row-normalized)" if normalize else "Confusion Matrix (counts)",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # 类别过多时不写数值，避免图像不可读
    if n <= 30:
        threshold = float(matrix.max()) * 0.5 if matrix.size else 0.0
        for i in range(n):
            for j in range(n):
                count = int(confusion[i, j])
                if normalize:
                    value = float(matrix[i, j])
                    text = f"{count}\n{value:.2f}"
                else:
                    text = f"{count}"
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if float(matrix[i, j]) > threshold else "black",
                )

    fig.tight_layout()
    filename = "confusion_matrix.png" if normalize else "confusion_matrix_counts.png"
    _save_figure(fig, charts_dir / f"{prefix}_{filename}")
    plt.close(fig)
