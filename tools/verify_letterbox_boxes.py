#!/usr/bin/env python3
"""
验证 shared_transform=letterbox 时 bbox 是否与图像同步缩放/回映射。

做法：
1) 先取原始 target（ConvertCoco 产出的 xyxy 像素坐标）
2) 走一次 dataset 内部的 _apply_shared_transforms_letterbox（得到归一化 cxcywh）
3) 将归一化 cxcywh -> letterbox 像素 xyxy，再按 (pad, scale) 反算回原图坐标
4) 与步骤 1 的原始 xyxy 做误差对比
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
from utils.box_ops import box_cxcywh_to_xyxy  # noqa: E402


def _merge_cfg(cfg: Dict[str, Any]) -> SimpleNamespace:
    data_cfg = cfg.get("data", {}) or {}
    train_cfg = cfg.get("train", {}) or {}
    model_cfg = cfg.get("model", {}) or {}

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
    return args


def _de_letterbox_xyxy(boxes_xyxy: torch.Tensor, pad_xy: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    pad = pad_xy.to(device=boxes_xyxy.device, dtype=boxes_xyxy.dtype).flatten()
    scale = scale.to(device=boxes_xyxy.device, dtype=boxes_xyxy.dtype).flatten()
    if pad.numel() < 2 or scale.numel() < 1:
        raise ValueError("invalid letterbox metadata")
    s = scale[0].clamp_min(1e-12)
    boxes = boxes_xyxy.clone()
    boxes[:, 0::2] -= pad[0]
    boxes[:, 1::2] -= pad[1]
    return boxes / s


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolved-config", type=str, required=True, help="outputs/**/config.yaml 路径")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--num", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args_cli = parser.parse_args()

    torch.manual_seed(int(args_cli.seed))

    with open(args_cli.resolved_config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    args = _merge_cfg(cfg)
    ds = build_dataset(args_cli.split, args, args.img_size)

    if not hasattr(ds, "_load_prepared_item") or not hasattr(ds, "_apply_shared_transforms_letterbox"):
        raise TypeError(f"dataset={type(ds)} 不支持该验证脚本（需要 CocoRgbMultispectralDataset）。")

    # 验证目的仅关注 resize/pad 与 bbox 同步，避免训练 split 的随机翻转引入额外误差。
    try:
        if hasattr(ds, "cfg"):
            ds.cfg.random_horizontal_flip = False  # type: ignore[attr-defined]
            ds.cfg.flip_prob = 0.0  # type: ignore[attr-defined]
    except Exception:
        pass

    n = min(int(args_cli.num), len(ds))
    max_err = 0.0
    mean_err = 0.0
    counted = 0

    for i in range(n):
        rgb_img0, ms0, target0 = ds._load_prepared_item(i)  # type: ignore[attr-defined]
        if target0 is None or target0.get("boxes") is None:
            continue

        rgb_img1, ms1, target1 = ds._apply_shared_transforms_letterbox(rgb_img0, ms0, target0)  # type: ignore[attr-defined]
        boxes0 = target0["boxes"].detach().cpu()
        boxes1 = target1["boxes"].detach().cpu()

        if boxes0.numel() == 0 and boxes1.numel() == 0:
            continue
        if boxes1.numel() == 0 and boxes0.numel() > 0:
            raise RuntimeError("letterbox 后 boxes 变为空（疑似 bbox 同步异常）")

        # normalized cxcywh -> letterbox pixel xyxy
        h1, w1 = [int(x) for x in target1["size"].tolist()]
        xyxy = box_cxcywh_to_xyxy(boxes1)
        scale_hw = torch.tensor([w1, h1, w1, h1], dtype=xyxy.dtype)
        xyxy_pix = xyxy * scale_hw

        pad = target1.get("letterbox_pad")
        s = target1.get("letterbox_scale")
        if pad is None or s is None:
            raise RuntimeError("target 缺少 letterbox_pad/letterbox_scale")
        xyxy_back = _de_letterbox_xyxy(xyxy_pix, pad, s)

        # compare
        err = (xyxy_back - boxes0).abs().max().item() if boxes0.numel() else 0.0
        max_err = max(max_err, float(err))
        mean_err += float(err)
        counted += 1

    mean_err = mean_err / max(1, counted)
    print(f"checked={counted}/{n} max_abs_err_px={max_err:.4g} mean_max_abs_err_px={mean_err:.4g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
