from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
from PIL import Image
from sklearn.cluster import MiniBatchKMeans



REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

"""
python tools/cluster_analysis/repr_cluster_rtmsfdetr.py \
    --run-dir outputs/repr/rtmsfdetr/oil_rgb_val_k32x16_20260104 --k1 32 --k2 16 --l2norm --save-vis --alpha 0.45
"""

@dataclass(frozen=True)
class ClusterConfig:
    k1: int
    k2: int
    sample_per_image: int
    seed: int
    l2norm: bool
    batch_size: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对 RTMSFDETR 导出的 dense 特征做两层聚类，并生成伪彩色/overlay。")
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="`tools/repr_export_rtmsfdetr.py` 生成的 run 目录（包含 features/ 与 meta.json）。",
    )
    parser.add_argument("--k1", type=int, default=8, help="第一层聚类 K（默认 8）。")
    parser.add_argument("--k2", type=int, default=8, help="第二层聚类 K（默认 8）。")
    parser.add_argument("--sample-per-image", type=int, default=2000, help="每张图用于训练聚类的采样点数（默认 2000）。")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认 42）。")
    parser.add_argument("--l2norm", action="store_true", help="对特征向量做 L2 normalize（建议开启）。")
    parser.add_argument("--mbk-batch-size", type=int, default=4096, help="MiniBatchKMeans batch_size（默认 4096）。")
    parser.add_argument("--save-vis", action="store_true", help="生成伪彩色与 overlay 可视化。")
    parser.add_argument("--alpha", type=float, default=0.45, help="overlay 透明度（默认 0.45）。")
    parser.add_argument("--limit", type=int, default=0, help="最多处理多少个 feature 文件（0 表示不限制）。")
    return parser.parse_args()


def _load_npz(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    data = np.load(path, allow_pickle=False)
    feat = data["feat"]
    meta_raw = data["meta"].item() if isinstance(data["meta"], np.ndarray) else data["meta"]
    meta = json.loads(str(meta_raw))
    if feat.ndim != 3:
        raise ValueError(f"feat 期望 [C,H,W]，实际 shape={feat.shape} file={path}")
    return feat, meta


def _l2_normalize(x: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    # Ensure float32 for stable norm (float16 may overflow on some feature distributions).
    x = x.astype(np.float32, copy=False)
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(denom, eps)


def _sample_vectors(
    feat_chw: np.ndarray,
    *,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    c, h, w = feat_chw.shape
    x = np.transpose(feat_chw, (1, 2, 0)).reshape(h * w, c)
    if n <= 0 or n >= x.shape[0]:
        return x
    idx = rng.choice(x.shape[0], size=int(n), replace=False)
    return x[idx]


def _ensure_min_rows(x: np.ndarray, *, min_rows: int, rng: np.random.Generator) -> np.ndarray:
    if x.shape[0] >= min_rows:
        return x
    if x.shape[0] == 0:
        return x
    need = int(min_rows - x.shape[0])
    idx = rng.integers(0, x.shape[0], size=need)
    return np.concatenate([x, x[idx]], axis=0)


def _make_palette(n: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    colors = rng.integers(0, 256, size=(n, 3), dtype=np.uint8)
    colors[0] = np.array([0, 0, 0], dtype=np.uint8)
    return colors


def _blend_overlay(rgb: np.ndarray, color: np.ndarray, *, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    out = (1.0 - alpha) * rgb.astype(np.float32) + alpha * color.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def _iter_feature_files(features_dir: Path) -> list[Path]:
    return sorted([p for p in features_dir.iterdir() if p.is_file() and p.suffix.lower() == ".npz"])


def _fit_kmeans_l1(
    files: Iterable[Path],
    *,
    cfg: ClusterConfig,
) -> MiniBatchKMeans:
    rng = np.random.default_rng(cfg.seed)
    km = MiniBatchKMeans(
        n_clusters=int(cfg.k1),
        random_state=int(cfg.seed),
        batch_size=int(cfg.batch_size),
        n_init="auto",
    )
    fitted = False
    pending: list[np.ndarray] = []
    pending_n = 0

    for npz in files:
        feat, _ = _load_npz(npz)
        x = _sample_vectors(feat, n=int(cfg.sample_per_image), rng=rng)
        if cfg.l2norm:
            x = _l2_normalize(x)

        if not fitted:
            pending.append(x)
            pending_n += x.shape[0]
            if pending_n >= int(cfg.k1):
                init = np.concatenate(pending, axis=0)
                init = _ensure_min_rows(init, min_rows=int(cfg.k1), rng=rng)
                km.partial_fit(init)
                fitted = True
                pending.clear()
                pending_n = 0
            continue

        km.partial_fit(x)

    if not fitted:
        raise RuntimeError("第一层聚类训练失败：样本不足。请增加图片数量或降低 --k1。")
    return km


def _fit_kmeans_l2(
    files: Iterable[Path],
    *,
    kmeans_l1: MiniBatchKMeans,
    cfg: ClusterConfig,
) -> list[MiniBatchKMeans]:
    rng = np.random.default_rng(cfg.seed + 17)
    models = [
        MiniBatchKMeans(
            n_clusters=int(cfg.k2),
            random_state=int(cfg.seed + i + 1),
            batch_size=int(cfg.batch_size),
            n_init="auto",
        )
        for i in range(int(cfg.k1))
    ]

    fitted = [False for _ in range(int(cfg.k1))]
    pending: list[list[np.ndarray]] = [[] for _ in range(int(cfg.k1))]
    pending_n = [0 for _ in range(int(cfg.k1))]

    for npz in files:
        feat, _ = _load_npz(npz)
        x = _sample_vectors(feat, n=int(cfg.sample_per_image), rng=rng)
        if cfg.l2norm:
            x = _l2_normalize(x)
        c1 = kmeans_l1.predict(x)

        for i in range(int(cfg.k1)):
            sel = x[c1 == i]
            if sel.size == 0:
                continue
            if not fitted[i]:
                pending[i].append(sel)
                pending_n[i] += sel.shape[0]
                if pending_n[i] >= int(cfg.k2):
                    init = np.concatenate(pending[i], axis=0)
                    init = _ensure_min_rows(init, min_rows=int(cfg.k2), rng=rng)
                    models[i].partial_fit(init)
                    fitted[i] = True
                    pending[i].clear()
                    pending_n[i] = 0
            else:
                models[i].partial_fit(sel)

    # 收尾：极小簇（样本仍不足 k2）用重复采样强行初始化
    for i in range(int(cfg.k1)):
        if fitted[i]:
            continue
        if not pending[i]:
            raise RuntimeError(
                f"第二层聚类训练失败：一级簇 {i} 没有任何样本。请降低 --k1 或增加数据。"
            )
        init = np.concatenate(pending[i], axis=0)
        init = _ensure_min_rows(init, min_rows=int(cfg.k2), rng=rng)
        models[i].partial_fit(init)
        fitted[i] = True

    return models


def _assign_labels(
    feat_chw: np.ndarray,
    *,
    kmeans_l1: MiniBatchKMeans,
    kmeans_l2: list[MiniBatchKMeans],
    cfg: ClusterConfig,
) -> np.ndarray:
    c, h, w = feat_chw.shape
    x = np.transpose(feat_chw, (1, 2, 0)).reshape(h * w, c)
    if cfg.l2norm:
        x = _l2_normalize(x)
    c1 = kmeans_l1.predict(x)
    final = np.empty((x.shape[0],), dtype=np.int32)
    for i in range(int(cfg.k1)):
        mask = c1 == i
        if not mask.any():
            continue
        c2 = kmeans_l2[i].predict(x[mask])
        final[mask] = int(i) * int(cfg.k2) + c2.astype(np.int32)
    return final.reshape(h, w)


def main() -> int:
    args = _parse_args()
    run_dir = Path(args.run_dir).expanduser()
    meta_path = run_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"run-dir 缺少 meta.json: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    features_dir = run_dir / "features"
    if not features_dir.is_dir():
        raise FileNotFoundError(f"run-dir 缺少 features/: {features_dir}")
    files = _iter_feature_files(features_dir)
    if args.limit and int(args.limit) > 0:
        files = files[: int(args.limit)]
    if not files:
        raise FileNotFoundError(f"features/ 下未找到 npz: {features_dir}")

    cfg = ClusterConfig(
        k1=int(args.k1),
        k2=int(args.k2),
        sample_per_image=int(args.sample_per_image),
        seed=int(args.seed),
        l2norm=bool(args.l2norm),
        batch_size=int(args.mbk_batch_size),
    )

    out_clusters = run_dir / "clusters"
    out_labels = run_dir / "labels"
    out_clusters.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)
    out_viz = run_dir / "viz"
    out_pseudo = out_viz / "pseudo_color"
    out_overlay = out_viz / "overlay"
    if args.save_vis:
        out_pseudo.mkdir(parents=True, exist_ok=True)
        out_overlay.mkdir(parents=True, exist_ok=True)

    logging.info("run_dir=%s", run_dir)
    logging.info("features=%s (N=%d)", features_dir, len(files))
    logging.info("k1=%d k2=%d sample_per_image=%d l2norm=%s", cfg.k1, cfg.k2, cfg.sample_per_image, cfg.l2norm)

    kmeans_l1 = _fit_kmeans_l1(files, cfg=cfg)
    kmeans_l2 = _fit_kmeans_l2(files, kmeans_l1=kmeans_l1, cfg=cfg)

    joblib.dump(kmeans_l1, out_clusters / "kmeans_l1.joblib")
    for i, km in enumerate(kmeans_l2):
        joblib.dump(km, out_clusters / f"kmeans_l2_{i:02d}.joblib")

    k_total = int(cfg.k1) * int(cfg.k2)
    palette = _make_palette(k_total, seed=int(cfg.seed))
    (out_clusters / "palette.json").write_text(
        json.dumps(
            {"k1": int(cfg.k1), "k2": int(cfg.k2), "colors": {str(i): palette[i].tolist() for i in range(k_total)}},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    label_dtype = np.uint8 if k_total <= 255 else np.uint16
    for npz in files:
        feat, m = _load_npz(npz)
        label_map = _assign_labels(feat, kmeans_l1=kmeans_l1, kmeans_l2=kmeans_l2, cfg=cfg).astype(label_dtype)

        stem = npz.stem
        out_label = out_labels / f"{stem}.png"
        Image.fromarray(label_map).save(out_label)

        if not args.save_vis:
            continue

        image_path = Path(str(m.get("image_path", ""))).expanduser()
        if not image_path.is_file():
            logging.warning("原图不存在，跳过可视化: %s", image_path)
            continue

        orig_hw = m.get("orig_hw", None)
        if not (isinstance(orig_hw, (list, tuple)) and len(orig_hw) == 2):
            img = Image.open(image_path).convert("RGB")
            orig_w, orig_h = img.size
        else:
            orig_h, orig_w = int(orig_hw[0]), int(orig_hw[1])
            img = Image.open(image_path).convert("RGB")

        # label_map is (Hfeat, Wfeat) -> upsample to orig (H, W)
        import cv2  # local import to avoid unnecessary dependency at --help time

        up = cv2.resize(label_map, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        color = palette[up.astype(np.int32)]
        rgb = np.array(img.resize((orig_w, orig_h), Image.BILINEAR), dtype=np.uint8)
        overlay = _blend_overlay(rgb, color, alpha=float(args.alpha))

        Image.fromarray(color).save(out_pseudo / f"{stem}.png")
        Image.fromarray(overlay).save(out_overlay / f"{stem}.png")

    (out_clusters / "cluster_meta.json").write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "source_meta": meta_path.name,
                "k1": int(cfg.k1),
                "k2": int(cfg.k2),
                "k_total": int(k_total),
                "sample_per_image": int(cfg.sample_per_image),
                "seed": int(cfg.seed),
                "l2norm": bool(cfg.l2norm),
                "files": len(files),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    logging.info("完成：labels=%s clusters=%s", out_labels, out_clusters)
    if args.save_vis:
        logging.info("可视化：%s %s", out_pseudo, out_overlay)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
