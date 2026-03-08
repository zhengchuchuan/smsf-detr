#!/usr/bin/env python
"""Prepare oil_msi_20260202_3cls for S2ADet (HOD3K-like layout).

- Converts COCO annotations to YOLO labels.
- Converts 7-channel MSI .tif into two 3-channel 8-bit streams.
- Writes S2ADet data YAMLs (val + test).

Output layout (same style as third_party/S2ADet/dataset/hod3k):
  <out_root>/
    se_information/{images,labels}/{train,val,test}
    sa_information/{images,labels}/{train,val,test}

Stream mapping (uses all 7 bands):
  se_information = [band0, band1, band2]
  sa_information = [band3, band4, mean(band5, band6)]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
import yaml
from PIL import Image


def _load_cfg(cfg_path: Path) -> dict:
    cfg = yaml.safe_load(cfg_path.read_text())
    data_cfg = cfg.get("data", {})
    if not data_cfg:
        raise ValueError(f"No 'data' section in {cfg_path}")
    return data_cfg


def _repo_root() -> Path:
    # scripts/s2adet/oil_dataset_to_s2adet.py -> repo root is parents[2]
    return Path(__file__).resolve().parents[2]


def _resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else (_repo_root() / path)


def _as_chw(ms: np.ndarray) -> np.ndarray:
    if ms.ndim != 3:
        raise ValueError(f"MSI array must be 3D, got {ms.shape}")
    # Accept CHW or HWC
    if ms.shape[0] in (7, 8, 16) and ms.shape[2] not in (7, 8, 16):
        return ms
    if ms.shape[2] in (7, 8, 16) and ms.shape[0] not in (7, 8, 16):
        return np.transpose(ms, (2, 0, 1))
    # Fallback: assume CHW
    return ms


def _split_streams(ms_chw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if ms_chw.shape[0] < 7:
        raise ValueError(f"Expected >=7 channels, got {ms_chw.shape}")
    a = ms_chw[0:3]
    b = np.stack([ms_chw[3], ms_chw[4], 0.5 * (ms_chw[5] + ms_chw[6])], axis=0)
    return a, b


def _to_uint8(chw: np.ndarray) -> np.ndarray:
    chw = np.clip(chw, 0.0, 1.0)
    chw = (chw * 255.0).round().astype(np.uint8)
    return chw


def _save_chw_png(chw: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    hwc = np.transpose(chw, (1, 2, 0))
    Image.fromarray(hwc, mode="RGB").save(out_path)


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
    msi_dir: Path,
    out_root: Path,
    ms_suffix: str,
    class_names: list[str],
    fixed_scale: float,
) -> None:
    coco = json.loads(ann_path.read_text())
    id_to_name = {c["id"]: c["name"] for c in coco.get("categories", [])}
    name_to_idx = {name: i for i, name in enumerate(class_names)}

    images = {img["id"]: img for img in coco.get("images", [])}
    anns_by_img: dict[int, list[dict]] = {img_id: [] for img_id in images}
    for ann in coco.get("annotations", []):
        if ann.get("iscrowd"):
            continue
        img_id = ann["image_id"]
        if img_id in anns_by_img:
            anns_by_img[img_id].append(ann)

    se_img_dir = out_root / "se_information" / "images" / split
    sa_img_dir = out_root / "sa_information" / "images" / split
    se_lbl_dir = out_root / "se_information" / "labels" / split
    sa_lbl_dir = out_root / "sa_information" / "labels" / split

    for img_id, img_info in images.items():
        file_name = img_info["file_name"]
        stem = Path(file_name).stem
        width = float(img_info["width"])
        height = float(img_info["height"])

        msi_path = msi_dir / split / f"{stem}{ms_suffix}"
        if not msi_path.is_file():
            # Fallback: find any file with same stem
            globbed = next(iter((msi_dir / split).glob(f"{stem}.*")), None)
            if globbed is None:
                print(f"[WARN] MSI not found for {stem} in {msi_dir / split}")
                continue
            msi_path = globbed

        ms = tifffile.imread(str(msi_path))
        ms = _as_chw(ms).astype(np.float32)
        ms = ms / float(fixed_scale)
        stream_a, stream_b = _split_streams(ms)
        stream_a = _to_uint8(stream_a)
        stream_b = _to_uint8(stream_b)

        se_out = se_img_dir / f"{stem}.png"
        sa_out = sa_img_dir / f"{stem}.png"
        if not se_out.is_file():
            _save_chw_png(stream_a, se_out)
        if not sa_out.is_file():
            _save_chw_png(stream_b, sa_out)

        # Labels
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
            # clamp
            cx = min(max(cx, 0.0), 1.0)
            cy = min(max(cy, 0.0), 1.0)
            bw = min(max(bw, 0.0), 1.0)
            bh = min(max(bh, 0.0), 1.0)
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        _write_labels(se_lbl_dir / f"{stem}.txt", lines)
        _write_labels(sa_lbl_dir / f"{stem}.txt", lines)


def _write_data_yaml(out_path: Path, *, out_root: Path, split: str, names: list[str]) -> None:
    data = {
        "train_rgb": str(out_root / "se_information" / "images" / "train"),
        "val_rgb": str(out_root / "se_information" / "images" / split),
        "train_ir": str(out_root / "sa_information" / "images" / "train"),
        "val_ir": str(out_root / "sa_information" / "images" / split),
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
        default=Path("configs/data/oil_msi_20260202_3cls.yaml"),
        help="MSI dataset config (msifp-detr).",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("third_party/S2ADet/dataset/oil_20260202_3cls"),
        help="Output root for S2ADet-ready dataset.",
    )
    parser.add_argument(
        "--ann-dir",
        type=Path,
        default=None,
        help="Override annotations dir (defaults to <dataset_dir>/annotations_3cls).",
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
        default=Path("third_party/S2ADet/data/oil_msi_20260202_3cls.yaml"),
        help="Output data.yaml for val split.",
    )
    parser.add_argument(
        "--data-yaml-test",
        type=Path,
        default=Path("third_party/S2ADet/data/oil_msi_20260202_3cls_test.yaml"),
        help="Output data.yaml for test split (val paths point to test).",
    )
    args = parser.parse_args()

    config_path = _resolve_repo_path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    data_cfg = _load_cfg(config_path)
    dataset_dir = _resolve_repo_path(Path(data_cfg.get("dataset_dir", "data/oil_20260202")))
    class_names = list(data_cfg.get("class_names", []))
    if not class_names:
        raise ValueError("class_names missing in config")

    fixed_scale = float(data_cfg.get("ms_fixed_scale", 65535.0))
    ms_suffix = str(data_cfg.get("ms_suffix", ".tif"))

    ann_dir = _resolve_repo_path(args.ann_dir) if args.ann_dir else (dataset_dir / "annotations_3cls")
    msi_dir = _resolve_repo_path(args.msi_dir) if args.msi_dir else (dataset_dir / "msi")
    out_root = _resolve_repo_path(args.out_root)
    data_yaml = _resolve_repo_path(args.data_yaml)
    data_yaml_test = _resolve_repo_path(args.data_yaml_test)

    for split in ("train", "val", "test"):
        ann_path = ann_dir / f"{split}.json"
        if not ann_path.is_file():
            print(f"[WARN] Missing annotations: {ann_path}")
            continue
        _convert_split(
            split=split,
            ann_path=ann_path,
            msi_dir=msi_dir,
            out_root=out_root,
            ms_suffix=ms_suffix,
            class_names=class_names,
            fixed_scale=fixed_scale,
        )

    _write_data_yaml(data_yaml, out_root=out_root, split="val", names=class_names)
    _write_data_yaml(data_yaml_test, out_root=out_root, split="test", names=class_names)
    print("Done.")


if __name__ == "__main__":
    main()
