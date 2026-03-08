#!/usr/bin/env python3
"""
快速抽样检查数据集样本的张量/标注格式，主要用于排查：
- coco_rgb / coco_rgb_msi 的 boxes 是否为 cxcywh 且归一化到 [0,1]
- target 中是否包含 orig_size/size/image_id 等关键字段
- dual_stream 输出(dict) 的 rgb/ms 形状是否一致
"""

from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace
from typing import Any, Dict

import torch
import yaml

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from datasets import build_dataset  # noqa: E402


def _merge_cfg(cfg: Dict[str, Any]) -> SimpleNamespace:
    data_cfg = cfg.get("data", {}) or {}
    train_cfg = cfg.get("train", {}) or {}
    model_cfg = cfg.get("model", {}) or {}

    sentinel = object()

    def pick(key: str, default=None):
        for src in (data_cfg, train_cfg, model_cfg):
            if isinstance(src, dict) and key in src:
                return src[key]
        return default

    args = SimpleNamespace()
    for src in (data_cfg, train_cfg, model_cfg):
        if isinstance(src, dict):
            for k, v in src.items():
                if not hasattr(args, k):
                    setattr(args, k, v)

    args.dataset_file = pick("dataset_file", getattr(args, "dataset_file", None))
    args.dataset_dir = pick("dataset_dir", getattr(args, "dataset_dir", None))
    args.ms_dataset_dir = pick("ms_dataset_dir", getattr(args, "ms_dataset_dir", None))
    args.img_size = int(pick("img_size", getattr(args, "img_size", 640)))
    args.patch_size = int(pick("patch_size", getattr(args, "patch_size", 16)))
    args.num_windows = int(pick("num_windows", getattr(args, "num_windows", 4)))
    args.segmentation_head = bool(pick("segmentation_head", getattr(args, "segmentation_head", False)))
    args.class_names = pick("class_names", getattr(args, "class_names", None))
    args.remap_mscoco_category = bool(pick("remap_mscoco_category", getattr(args, "remap_mscoco_category", False)))

    legacy_map = {
        "coco_ms": "coco_rgb_msi",
        "coco_ms_rgb": "coco_rgb",
    }
    dataset_key = legacy_map.get(args.dataset_file, args.dataset_file)
    if (
        args.class_names
        and dataset_key in {"coco_rgb", "coco_msi", "coco_rgb_msi"}
        and not args.remap_mscoco_category
    ):
        args.remap_mscoco_category = True
    return args


def _fmt_tensor(x: torch.Tensor) -> str:
    return f"shape={tuple(x.shape)} dtype={x.dtype} min={float(x.min()):.4g} max={float(x.max()):.4g}"


def _inspect_sample(sample: Any, target: Dict[str, Any], *, idx: int) -> None:
    print(f"\n=== sample[{idx}] ===")
    if isinstance(sample, dict):
        for k, v in sample.items():
            print(f"sample[{k}]: {_fmt_tensor(v)}")
    else:
        print(f"sample: {_fmt_tensor(sample)}")

    print("target keys:", sorted(target.keys()))
    if "image_id" in target:
        try:
            print("image_id:", int(target["image_id"].item()))
        except Exception:
            print("image_id:", target["image_id"])
    for key in ("orig_size", "size"):
        if key in target:
            try:
                print(f"{key}:", [int(x) for x in target[key].tolist()])
            except Exception:
                print(f"{key}:", target[key])

    boxes = target.get("boxes")
    if torch.is_tensor(boxes):
        print("boxes:", _fmt_tensor(boxes))
        if boxes.numel():
            cx, cy, w, h = boxes.unbind(-1)
            print(
                "boxes range:",
                f"cx[{float(cx.min()):.4g},{float(cx.max()):.4g}]",
                f"cy[{float(cy.min()):.4g},{float(cy.max()):.4g}]",
                f"w[{float(w.min()):.4g},{float(w.max()):.4g}]",
                f"h[{float(h.min()):.4g},{float(h.max()):.4g}]",
            )
            out_of_range = ((boxes < 0) | (boxes > 1)).any().item()
            print("boxes_out_of_[0,1]:", bool(out_of_range))

    labels = target.get("labels")
    if torch.is_tensor(labels):
        uniq = labels.unique().tolist()
        print("labels:", f"shape={tuple(labels.shape)} unique={uniq[:20]}{'...' if len(uniq) > 20 else ''}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolved-config", type=str, required=True, help="outputs/**/config.yaml 路径")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--num", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args_cli = parser.parse_args()

    torch.manual_seed(int(args_cli.seed))

    with open(args_cli.resolved_config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    args = _merge_cfg(cfg)
    if not getattr(args, "dataset_file", None):
        raise ValueError("resolved config 中未找到 data.dataset_file / dataset_file。")

    ds = build_dataset(args_cli.split, args, args.img_size)
    print(f"dataset_file={args.dataset_file} split={args_cli.split} len={len(ds)} remap_mscoco_category={getattr(args, 'remap_mscoco_category', False)}")
    for i in range(min(int(args_cli.num), len(ds))):
        sample, target = ds[i]
        _inspect_sample(sample, target, idx=i)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
