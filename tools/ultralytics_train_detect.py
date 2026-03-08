#!/usr/bin/env python
"""
Small wrapper to train Ultralytics YOLO (v5/v8/YOLO11) from this repo without installing as a package.

Why this exists:
- Avoid writing configs under ~/.config (sandbox may block). We force YOLO_CONFIG_DIR to a writable repo path.
- Avoid needing `pip install -e third_party/ultralytics`; we add it to sys.path at runtime.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _repo_root() -> Path:
    # tools/ultralytics_train_detect.py -> repo root is parent of tools/
    return Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True, help="e.g. yolov5s6.pt / yolov8s.pt / yolo11s.pt or a local .pt")
    p.add_argument("--data", type=str, required=True, help="Ultralytics dataset YAML path")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", type=str, default="0", help="CUDA device id(s) like '0'/'0,1' or 'cpu'")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--project", type=str, default=None, help="Output dir root (Ultralytics 'project=')")
    p.add_argument("--name", type=str, default=None, help="Run name (Ultralytics 'name=')")
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--cache", action="store_true", help="Cache images for faster training")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--pretrained", action="store_true", help="Force pretrained=True (useful when training from a .yaml model)")
    g.add_argument("--no-pretrained", action="store_true", help="Force pretrained=False (train from scratch)")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    root = _repo_root()

    # Ensure Ultralytics writes settings to a writable location.
    os.environ.setdefault("YOLO_CONFIG_DIR", str(root / "tmp" / "ultralytics_config"))

    # Use vendored ultralytics without pip install.
    sys.path.insert(0, str(root / "third_party" / "ultralytics"))

    from ultralytics import YOLO  # noqa: E402

    project = args.project or str(root / "outputs" / "ultralytics")
    name = args.name or Path(args.model).stem

    model = YOLO(args.model)
    train_kwargs = dict(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        patience=args.patience,
        cache=args.cache,
        project=project,
        name=name,
    )
    # Only override Ultralytics defaults when explicitly requested.
    if args.pretrained:
        train_kwargs["pretrained"] = True
    elif args.no_pretrained:
        train_kwargs["pretrained"] = False

    model.train(**train_kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
