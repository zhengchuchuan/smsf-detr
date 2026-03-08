#!/usr/bin/env python3
"""
将旧的 annotations 目录结构同步为“扁平化”结构，便于检测/分割共用同一份 COCO 标注。

推荐结构：
  dataset_root/annotations/{train,val,test}.json

兼容旧结构（作为来源）：
  dataset_root/annotations/segmentation/{split}.json
  dataset_root/annotations/detection/{split}.json

默认策略：
- 优先选择 segmentation/{split}.json 作为 annotations/{split}.json（若存在）；
- 否则回退到 detection/{split}.json；
- 通过 --mode 控制 copy/symlink/hardlink/move。
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Iterable, Optional


def _pick_source(ann_dir: Path, split: str) -> Optional[Path]:
    candidates = [
        ann_dir / "segmentation" / f"{split}.json",
        ann_dir / "detection" / f"{split}.json",
        ann_dir / f"{split}.json",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def _remove_existing(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    if path.exists():
        raise IsADirectoryError(f"目标路径存在且不是文件: {path}")


def _sync_one(src: Path, dst: Path, *, mode: str, overwrite: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        _remove_existing(dst)

    if mode == "copy":
        shutil.copy2(src, dst)
        return
    if mode == "move":
        shutil.move(str(src), str(dst))
        return
    if mode == "hardlink":
        os.link(src, dst)
        return
    if mode == "symlink":
        rel = os.path.relpath(src, start=dst.parent)
        os.symlink(rel, dst)
        return
    raise ValueError(f"未知 mode: {mode}")


def simplify_annotations_layout(
    dataset_root: Path,
    *,
    splits: Iterable[str],
    mode: str,
    overwrite: bool,
) -> None:
    ann_dir = dataset_root / "annotations"
    if not ann_dir.exists():
        raise FileNotFoundError(f"未找到 annotations 目录: {ann_dir}")

    for split in splits:
        src = _pick_source(ann_dir, split)
        if src is None:
            raise FileNotFoundError(
                f"未找到 {split}.json 可用来源（已尝试 segmentation/detection/扁平路径）: {ann_dir}"
            )
        dst = ann_dir / f"{split}.json"
        _sync_one(src, dst, mode=mode, overwrite=overwrite)
        print(f"[OK] {dst} <= {src} ({mode})")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="data/oil",
        help="数据集根目录（包含 annotations/ rgb/ msi/ 等）",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=["train", "val", "test"],
        help="要同步的 split 列表",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="copy",
        choices=["copy", "symlink", "hardlink", "move"],
        help="生成方式：copy(默认)/symlink/hardlink/move",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="若目标 annotations/{split}.json 已存在则覆盖",
    )
    args = parser.parse_args()

    simplify_annotations_layout(
        Path(args.dataset_root),
        splits=args.splits,
        mode=str(args.mode),
        overwrite=bool(args.overwrite),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

