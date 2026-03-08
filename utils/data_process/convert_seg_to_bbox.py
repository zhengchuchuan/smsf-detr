#!/usr/bin/env python3
"""Convert COCO-style instance segmentation annotations to pure bbox detection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List


DEFAULT_DATASETS = ("train.json", "val.json", "test.json")


def _flatten_segment(segment: Iterable[float]) -> List[float]:
    if segment is None:
        return []
    if isinstance(segment, (list, tuple)):
        flattened: List[float] = []
        for value in segment:
            if isinstance(value, (list, tuple)):
                flattened.extend(_flatten_segment(value))
            elif isinstance(value, (int, float)):
                flattened.append(float(value))
        return flattened
    return []


def segmentation_to_bbox(annotation: dict) -> List[float]:
    segmentation = annotation.get("segmentation")
    if isinstance(segmentation, list) and segmentation:
        coords = []
        for seg in segmentation:
            coords.extend(_flatten_segment(seg))
        if len(coords) < 4:
            raise ValueError(f"Unable to build bbox from segmentation: {annotation.get('id')}")
        xs = coords[0::2]
        ys = coords[1::2]
        x_min = min(xs)
        y_min = min(ys)
        x_max = max(xs)
        y_max = max(ys)
        return [float(x_min), float(y_min), float(max(0.0, x_max - x_min)), float(max(0.0, y_max - y_min))]
    bbox = annotation.get("bbox")
    if bbox:
        return [float(v) for v in bbox]
    raise ValueError(f"Annotation {annotation.get('id')} has neither segmentation nor bbox")


def convert_file(src: Path, dst: Path, keep_segmentation: bool = False) -> int:
    data = json.loads(src.read_text())
    updated = 0
    for ann in data.get("annotations", []):
        bbox = segmentation_to_bbox(ann)
        x, y, w, h = bbox
        ann["bbox"] = [x, y, w, h]
        ann["area"] = float(max(0.0, w) * max(0.0, h))
        if not keep_segmentation:
            ann.pop("segmentation", None)
        updated += 1

    data["type"] = "instances_bbox"
    dst.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    return updated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory that stores the COCO annotations (defaults to this script's directory)",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="File names to convert; defaults to train/val/test when present",
    )
    parser.add_argument(
        "--suffix",
        default="_bbox",
        help="Suffix appended to the converted file name (ignored when overwriting)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the input file instead of writing a new one",
    )
    parser.add_argument(
        "--keep-segmentation",
        action="store_true",
        help="Keep the segmentation field (it is removed by default)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = args.files or [name for name in DEFAULT_DATASETS if (args.input_dir / name).exists()]
    if not targets:
        raise SystemExit("No annotation files found for conversion")

    for name in targets:
        src = args.input_dir / name
        if not src.exists():
            raise FileNotFoundError(f"Missing annotation file: {src}")
        if args.overwrite:
            dst = src
        else:
            dst = src.with_name(src.stem + args.suffix + src.suffix)
            if dst.exists():
                raise FileExistsError(f"{dst} already exists; use --overwrite to replace it")

        count = convert_file(src, dst, keep_segmentation=args.keep_segmentation)
        print(f"[OK] {src.name} -> {dst.name}: {count} annotations converted")


if __name__ == "__main__":
    main()
