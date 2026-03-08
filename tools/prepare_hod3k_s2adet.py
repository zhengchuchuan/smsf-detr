#!/usr/bin/env python
"""Convert HOD3K_rgb_msi (COCO) to S2ADet-ready format.

- Reads configs/data/HOD3K_rgb_msi.yaml for dataset_dir/class_names/ms_suffix.
- Converts COCO bboxes to YOLO txt labels.
- Mirrors RGB/MSI images into S2ADet folder layout via symlink/hardlink/copy.
- Writes S2ADet data YAMLs (val + test).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import yaml


def _load_cfg(cfg_path: Path) -> dict:
    cfg = yaml.safe_load(cfg_path.read_text())
    data_cfg = cfg.get("data", {})
    if not data_cfg:
        raise ValueError(f"No data section in {cfg_path}")
    return data_cfg


def _resolve_image(img_dir: Path, stem: str, suffix_hint: str | None) -> Path | None:
    if suffix_hint:
        candidate = img_dir / f"{stem}{suffix_hint}"
        if candidate.is_file():
            return candidate
    matches = sorted(img_dir.glob(f"{stem}.*"))
    return matches[0] if matches else None


def _link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    try:
        os.symlink(src, dst)
        return
    except OSError:
        pass
    try:
        os.link(src, dst)
        return
    except OSError:
        shutil.copy2(src, dst)


def _write_labels(label_path: Path, lines: list[str]) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    if lines:
        label_path.write_text("\n".join(lines) + "\n")
    else:
        label_path.write_text("")


def _convert_split(
    *,
    split: str,
    ann_path: Path,
    rgb_dir: Path,
    msi_dir: Path,
    out_root: Path,
    ms_suffix: str,
    class_names: list[str],
    link_mode: str,
) -> None:
    coco = json.loads(ann_path.read_text())
    id_to_name = {c["id"]: c["name"] for c in coco.get("categories", [])}
    if class_names:
        name_to_idx = {name: i for i, name in enumerate(class_names)}
    else:
        ordered = [c["name"] for c in sorted(coco.get("categories", []), key=lambda x: x["id"])]
        name_to_idx = {name: i for i, name in enumerate(ordered)}

    images = {img["id"]: img for img in coco.get("images", [])}
    anns_by_img: dict[int, list[dict]] = {img_id: [] for img_id in images}
    for ann in coco.get("annotations", []):
        if ann.get("iscrowd"):
            continue
        img_id = ann["image_id"]
        if img_id in anns_by_img:
            anns_by_img[img_id].append(ann)

    rgb_img_dir = out_root / "rgb" / "images" / split
    ir_img_dir = out_root / "ir" / "images" / split
    rgb_lbl_dir = out_root / "rgb" / "labels" / split
    ir_lbl_dir = out_root / "ir" / "labels" / split

    for img_id, img_info in images.items():
        file_name = img_info["file_name"]
        stem = Path(file_name).stem
        width = float(img_info["width"])
        height = float(img_info["height"])

        rgb_src = _resolve_image(rgb_dir / split, stem, ".jpg")
        if rgb_src is None:
            print(f"[WARN] RGB not found for {stem} in {rgb_dir / split}")
            continue
        msi_src = _resolve_image(msi_dir / split, stem, ms_suffix)
        if msi_src is None:
            print(f"[WARN] MSI not found for {stem} in {msi_dir / split}")
            continue

        rgb_out = rgb_img_dir / rgb_src.name
        ir_out = ir_img_dir / msi_src.name
        _link_or_copy(rgb_src, rgb_out, link_mode)
        _link_or_copy(msi_src, ir_out, link_mode)

        lines = []
        for ann in anns_by_img.get(img_id, []):
            cat_name = id_to_name.get(ann["category_id"])
            if cat_name not in name_to_idx:
                continue
            cls = name_to_idx[cat_name]
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            cx = (x + w / 2.0) / width
            cy = (y + h / 2.0) / height
            bw = w / width
            bh = h / height
            cx = min(max(cx, 0.0), 1.0)
            cy = min(max(cy, 0.0), 1.0)
            bw = min(max(bw, 0.0), 1.0)
            bh = min(max(bh, 0.0), 1.0)
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        _write_labels(rgb_lbl_dir / f"{stem}.txt", lines)
        _write_labels(ir_lbl_dir / f"{stem}.txt", lines)


def _write_data_yaml(out_path: Path, *, out_root: Path, split: str, names: list[str]) -> None:
    data = {
        "train_rgb": str(out_root / "rgb" / "images" / "train"),
        "val_rgb": str(out_root / "rgb" / "images" / split),
        "train_ir": str(out_root / "ir" / "images" / "train"),
        "val_ir": str(out_root / "ir" / "images" / split),
        "nc": len(names),
        "names": names,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(data, sort_keys=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data/HOD3K_rgb_msi.yaml"),
        help="MSI dataset config (msifp-detr).",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("third_party/S2ADet/data/HOD3K/s2adet_hod3k_rgb_msi"),
        help="Output root for S2ADet-ready dataset.",
    )
    parser.add_argument(
        "--ann-dir",
        type=Path,
        default=None,
        help="Override annotations dir (defaults to <dataset_dir>/annotations).",
    )
    parser.add_argument(
        "--rgb-dir",
        type=Path,
        default=None,
        help="Override RGB dir (defaults to <dataset_dir>/rgb).",
    )
    parser.add_argument(
        "--msi-dir",
        type=Path,
        default=None,
        help="Override MSI dir (defaults to <dataset_dir>/msi).",
    )
    parser.add_argument(
        "--data-yaml",
        type=Path,
        default=Path("third_party/S2ADet/data/hod3k_rgb_msi.yaml"),
        help="Output data.yaml for val split.",
    )
    parser.add_argument(
        "--data-yaml-test",
        type=Path,
        default=Path("third_party/S2ADet/data/hod3k_rgb_msi_test.yaml"),
        help="Output data.yaml for test split (val paths point to test).",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy images instead of symlink/hardlink.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Splits to convert.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    cfg_path = args.config
    if not cfg_path.is_file():
        fallback = repo_root / cfg_path
        if fallback.is_file():
            cfg_path = fallback
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config not found: {args.config}")

    data_cfg = _load_cfg(cfg_path)
    dataset_dir = Path(data_cfg.get("dataset_dir", "data/HOD3K"))
    class_names = list(data_cfg.get("class_names", []))
    if not class_names:
        raise ValueError("class_names missing in config")

    ms_suffix = str(data_cfg.get("ms_suffix", ".png"))

    def _resolve_repo_path(path: Path) -> Path:
        return path if path.is_absolute() else repo_root / path

    dataset_dir = _resolve_repo_path(dataset_dir)
    ann_dir = _resolve_repo_path(args.ann_dir) if args.ann_dir else (dataset_dir / "annotations")
    rgb_dir = _resolve_repo_path(args.rgb_dir) if args.rgb_dir else (dataset_dir / "rgb")
    msi_dir = _resolve_repo_path(args.msi_dir) if args.msi_dir else (dataset_dir / "msi")
    out_root = _resolve_repo_path(args.out_root)
    data_yaml = _resolve_repo_path(args.data_yaml)
    data_yaml_test = _resolve_repo_path(args.data_yaml_test)
    link_mode = "copy" if args.copy else "link"

    for split in args.splits:
        ann_path = ann_dir / f"{split}.json"
        if not ann_path.is_file():
            print(f"[WARN] Missing annotations: {ann_path}")
            continue
        _convert_split(
            split=split,
            ann_path=ann_path,
            rgb_dir=rgb_dir,
            msi_dir=msi_dir,
            out_root=out_root,
            ms_suffix=ms_suffix,
            class_names=class_names,
            link_mode=link_mode,
        )

    _write_data_yaml(data_yaml, out_root=out_root, split="val", names=class_names)
    _write_data_yaml(data_yaml_test, out_root=out_root, split="test", names=class_names)
    print("Done.")


if __name__ == "__main__":
    main()
