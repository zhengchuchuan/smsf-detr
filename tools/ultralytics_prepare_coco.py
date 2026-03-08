#!/usr/bin/env python
"""
Prepare Ultralytics YOLO-format labels from COCO annotations and create image symlinks.

This supports datasets where COCO file_name extension may not match RGB files
by resolving to an existing image with the same stem.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ann", type=str, required=True, help="Path to COCO annotations JSON")
    p.add_argument("--images-dir", type=str, required=True, help="Directory of RGB images for this split")
    p.add_argument("--out-images", type=str, required=True, help="Output images/ split dir (symlinks created here)")
    p.add_argument("--out-labels", type=str, required=True, help="Output labels/ split dir (txt files created here)")
    p.add_argument(
        "--class-names",
        type=str,
        default="",
        help="Comma-separated class names to enforce ordering, e.g. 'people,bike,car'",
    )
    return p.parse_args()


def _resolve_image(images_dir: Path, file_name: str) -> Optional[Path]:
    p = images_dir / file_name
    if p.exists():
        return p
    stem = Path(file_name).stem
    for ext in [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"]:
        alt = images_dir / f"{stem}{ext}"
        if alt.exists():
            return alt
    return None


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def main() -> int:
    args = _parse_args()
    ann_path = Path(args.ann)
    images_dir = Path(args.images_dir)
    out_images = Path(args.out_images)
    out_labels = Path(args.out_labels)
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    data = json.load(open(ann_path, "r"))
    categories = data.get("categories", [])
    images = data.get("images", [])
    annotations = data.get("annotations", [])

    class_names: List[str] = [n.strip() for n in args.class_names.split(",") if n.strip()]
    if class_names:
        name_to_idx = {n: i for i, n in enumerate(class_names)}
        cat_id_to_idx = {}
        for c in categories:
            name = c["name"]
            if name not in name_to_idx:
                raise ValueError(f"Category '{name}' not found in --class-names list.")
            cat_id_to_idx[c["id"]] = name_to_idx[name]
    else:
        categories_sorted = sorted(categories, key=lambda x: x["id"])
        class_names = [c["name"] for c in categories_sorted]
        cat_id_to_idx = {c["id"]: i for i, c in enumerate(categories_sorted)}

    # Build annotations per image_id.
    ann_by_image: Dict[int, List[dict]] = {}
    for ann in annotations:
        ann_by_image.setdefault(ann["image_id"], []).append(ann)

    missing = []
    for img in images:
        img_id = img["id"]
        file_name = img["file_name"]
        width = img.get("width", 0) or 0
        height = img.get("height", 0) or 0
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image size in {ann_path}: id={img_id} w={width} h={height}")

        src = _resolve_image(images_dir, file_name)
        if src is None:
            missing.append(file_name)
            continue

        # Keep COCO file_name for output to keep labels aligned with COCO metadata.
        out_img = out_images / file_name
        out_img.parent.mkdir(parents=True, exist_ok=True)
        if not out_img.exists() or out_img.is_symlink():
            rel = os.path.relpath(src, out_img.parent)
            out_img.unlink(missing_ok=True)
            out_img.symlink_to(rel)

        # Write labels
        label_path = out_labels / f"{Path(file_name).stem}.txt"
        lines: List[str] = []
        for ann in ann_by_image.get(img_id, []):
            if ann.get("iscrowd", 0):
                continue
            bbox = ann.get("bbox", None)
            if not bbox or len(bbox) != 4:
                continue
            x, y, w, h = bbox
            if w <= 0 or h <= 0:
                continue
            cx = _clamp01((x + w / 2.0) / width)
            cy = _clamp01((y + h / 2.0) / height)
            nw = _clamp01(w / width)
            nh = _clamp01(h / height)
            cls_id = cat_id_to_idx.get(ann["category_id"])
            if cls_id is None:
                continue
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    if missing:
        sample = ", ".join(missing[:5])
        raise FileNotFoundError(
            f"Missing {len(missing)} images under {images_dir}. Sample: {sample}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

