#!/usr/bin/env python
"""Convert oil_20260202 COCO splits to MRT-DETR fusion dataset layout.

Input (default):
  data/oil_20260202/
    annotations_3cls/{train,val,test}.json
    rgb/{train,val,test}/*.jpg
    msi/{train,val,test}/*.tif

Output:
  <out_root>/
    annotations/{train,val,test}.json
    train_RGB/      train_thermal/
    val_RGB/        val_thermal/
    test_RGB/       test_thermal/  (optional split)

The output COCO image entries will include both `file_name_RGB` and `file_name_IR`
(required by MRT-DETR's `CocoFusionDetection`).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
import tifffile
from PIL import Image


def _repo_root() -> Path:
    # scripts/mrt-detr/oil_dataset_to_mrtdetr.py -> repo root is parents[2]
    return Path(__file__).resolve().parents[2]


def _resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else (_repo_root() / path)


def _as_chw(ms: np.ndarray) -> np.ndarray:
    if ms.ndim == 2:
        return ms[None, ...]
    if ms.ndim != 3:
        raise ValueError(f"MSI array must be 2D/3D, got shape={ms.shape}")

    # CHW: small first dim, large spatial dims
    if ms.shape[0] <= 32 and ms.shape[1] > 32 and ms.shape[2] > 32:
        return ms
    # HWC: small last dim, large spatial dims
    if ms.shape[2] <= 32 and ms.shape[0] > 32 and ms.shape[1] > 32:
        return np.transpose(ms, (2, 0, 1))

    # Fallback to CHW to avoid accidental channel permutation.
    return ms


def _to_u8_rgb(chw: np.ndarray, scale: float) -> np.ndarray:
    if chw.ndim != 3:
        raise ValueError(f"Expected CHW array, got {chw.shape}")
    chw = chw.astype(np.float32)
    if scale > 0:
        chw = chw / scale
    else:
        # Per-channel min-max if scale <= 0.
        cmin = chw.min(axis=(1, 2), keepdims=True)
        cmax = chw.max(axis=(1, 2), keepdims=True)
        denom = np.maximum(cmax - cmin, 1e-6)
        chw = (chw - cmin) / denom

    chw = np.clip(chw, 0.0, 1.0)
    hwc = np.transpose((chw * 255.0).round().astype(np.uint8), (1, 2, 0))
    return hwc


def _materialize_file(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return

    if mode == "copy":
        shutil.copy2(src, dst)
        return

    if mode == "symlink":
        rel_src = os.path.relpath(src, dst.parent)
        dst.symlink_to(rel_src)
        return

    if mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
        return

    raise ValueError(f"Unsupported mode: {mode}")


def _find_with_same_stem(folder: Path, stem: str) -> Path | None:
    candidates = sorted(folder.glob(f"{stem}.*"))
    return candidates[0] if candidates else None


def _parse_bands(text: str) -> list[int]:
    bands = [int(x.strip()) for x in text.split(",") if x.strip()]
    if len(bands) != 3:
        raise ValueError("--ir-bands must contain exactly 3 indices, e.g. 0,1,2")
    if any(b < 0 for b in bands):
        raise ValueError("--ir-bands cannot contain negative indices")
    return bands


def _convert_msi_to_ir_png(msi_path: Path, out_path: Path, bands: list[int], ms_scale: float) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return

    ms = tifffile.imread(str(msi_path))
    ms = _as_chw(ms)
    ch = ms.shape[0]
    if max(bands) >= ch:
        raise ValueError(f"Bands {bands} out of range for {msi_path.name} with {ch} channels")

    rgb = ms[bands, :, :]
    rgb_u8 = _to_u8_rgb(rgb, scale=ms_scale)
    Image.fromarray(rgb_u8, mode="RGB").save(out_path)


def _iter_splits(ann_dir: Path, requested: Iterable[str]) -> list[str]:
    splits: list[str] = []
    for split in requested:
        ann_path = ann_dir / f"{split}.json"
        if ann_path.is_file():
            splits.append(split)
        else:
            print(f"[WARN] skip split={split}: annotation not found: {ann_path}")
    return splits


def _convert_split(
    *,
    split: str,
    ann_path: Path,
    rgb_dir: Path,
    msi_dir: Path,
    out_root: Path,
    rgb_mode: str,
    ir_mode: str,
    ir_file_mode: str,
    ir_bands: list[int],
    ms_scale: float,
) -> tuple[int, int, int]:
    coco = json.loads(ann_path.read_text())
    images = list(coco.get("images", []))
    annotations = list(coco.get("annotations", []))

    out_rgb_dir = out_root / f"{split}_RGB"
    out_ir_dir = out_root / f"{split}_thermal"

    split_rgb_dir = rgb_dir / split
    split_msi_dir = msi_dir / split

    kept_images = []
    kept_image_ids: set[int] = set()
    missing_pairs = 0

    for img in images:
        file_name = str(img["file_name"])
        stem = Path(file_name).stem

        rgb_src = split_rgb_dir / file_name
        if not rgb_src.is_file():
            rgb_src = _find_with_same_stem(split_rgb_dir, stem)

        msi_src = _find_with_same_stem(split_msi_dir, stem)

        if rgb_src is None or msi_src is None or not rgb_src.is_file() or not msi_src.is_file():
            missing_pairs += 1
            continue

        rgb_name = rgb_src.name
        out_rgb_path = out_rgb_dir / rgb_name
        _materialize_file(rgb_src, out_rgb_path, mode=rgb_mode)

        if ir_mode == "convert_msi_to_png":
            ir_name = f"{stem}.png"
            out_ir_path = out_ir_dir / ir_name
            _convert_msi_to_ir_png(msi_src, out_ir_path, bands=ir_bands, ms_scale=ms_scale)
        else:
            ir_name = msi_src.name
            out_ir_path = out_ir_dir / ir_name
            _materialize_file(msi_src, out_ir_path, mode=ir_file_mode)

        img_new = dict(img)
        img_new["file_name"] = rgb_name
        img_new["file_name_RGB"] = rgb_name
        img_new["file_name_IR"] = ir_name
        kept_images.append(img_new)
        kept_image_ids.add(int(img_new["id"]))

    kept_annotations = [ann for ann in annotations if int(ann["image_id"]) in kept_image_ids]

    coco_new = dict(coco)
    coco_new["images"] = kept_images
    coco_new["annotations"] = kept_annotations

    out_ann_dir = out_root / "annotations"
    out_ann_dir.mkdir(parents=True, exist_ok=True)
    out_ann_path = out_ann_dir / f"{split}.json"
    out_ann_path.write_text(json.dumps(coco_new, ensure_ascii=False, indent=2))

    return len(images), len(kept_images), missing_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert oil_20260202 to MRT-DETR format")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/oil_20260202"),
        help="Source dataset root.",
    )
    parser.add_argument(
        "--ann-dir",
        type=Path,
        default=None,
        help="COCO annotation dir (default: <dataset-dir>/annotations_3cls).",
    )
    parser.add_argument(
        "--rgb-dir",
        type=Path,
        default=None,
        help="RGB split dir (default: <dataset-dir>/rgb).",
    )
    parser.add_argument(
        "--msi-dir",
        type=Path,
        default=None,
        help="MSI split dir (default: <dataset-dir>/msi).",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("data/oil_20260202_mrtdetr"),
        help="Output root in MRT-DETR layout.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Splits to process (default: train val test).",
    )
    parser.add_argument(
        "--rgb-mode",
        choices=["hardlink", "copy", "symlink"],
        default="hardlink",
        help="How to place RGB images in output.",
    )
    parser.add_argument(
        "--ir-mode",
        choices=["convert_msi_to_png", "keep_original"],
        default="keep_original",
        help="IR mode: keep original tif by default, or convert MSI to 3-channel PNG.",
    )
    parser.add_argument(
        "--ir-file-mode",
        choices=["copy", "hardlink", "symlink"],
        default="copy",
        help="How to place IR files when --ir-mode=keep_original.",
    )
    parser.add_argument(
        "--ir-bands",
        type=str,
        default="3,4,5",
        help="Three MSI channel indices used to build IR RGB PNG when --ir-mode=convert_msi_to_png.",
    )
    parser.add_argument(
        "--ms-scale",
        type=float,
        default=65535.0,
        help="Scale for MSI->uint8 conversion; set <=0 to use per-image min-max.",
    )
    args = parser.parse_args()

    dataset_dir = _resolve_repo_path(args.dataset_dir)
    ann_dir = _resolve_repo_path(args.ann_dir) if args.ann_dir else (dataset_dir / "annotations_3cls")
    rgb_dir = _resolve_repo_path(args.rgb_dir) if args.rgb_dir else (dataset_dir / "rgb")
    msi_dir = _resolve_repo_path(args.msi_dir) if args.msi_dir else (dataset_dir / "msi")
    out_root = _resolve_repo_path(args.out_root)
    ir_bands = _parse_bands(args.ir_bands)

    if not ann_dir.is_dir():
        raise FileNotFoundError(f"Annotation dir not found: {ann_dir}")
    if not rgb_dir.is_dir():
        raise FileNotFoundError(f"RGB dir not found: {rgb_dir}")
    if not msi_dir.is_dir():
        raise FileNotFoundError(f"MSI dir not found: {msi_dir}")

    splits = _iter_splits(ann_dir, args.splits)
    if not splits:
        raise RuntimeError("No valid split found. Check --ann-dir and --splits.")

    print(f"[INFO] src dataset: {dataset_dir}")
    print(f"[INFO] output root: {out_root}")
    print(f"[INFO] splits: {splits}")

    total_src = 0
    total_kept = 0
    total_missing = 0
    for split in splits:
        src_n, kept_n, miss_n = _convert_split(
            split=split,
            ann_path=ann_dir / f"{split}.json",
            rgb_dir=rgb_dir,
            msi_dir=msi_dir,
            out_root=out_root,
            rgb_mode=args.rgb_mode,
            ir_mode=args.ir_mode,
            ir_file_mode=args.ir_file_mode,
            ir_bands=ir_bands,
            ms_scale=float(args.ms_scale),
        )
        total_src += src_n
        total_kept += kept_n
        total_missing += miss_n
        print(
            f"[OK] split={split}: images {kept_n}/{src_n}, "
            f"missing_pairs={miss_n}, ann={out_root / 'annotations' / f'{split}.json'}"
        )

    print(
        f"[DONE] images kept {total_kept}/{total_src}, "
        f"missing_pairs={total_missing}, out={out_root}"
    )


if __name__ == "__main__":
    main()
