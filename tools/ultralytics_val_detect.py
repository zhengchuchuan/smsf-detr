#!/usr/bin/env python
"""
Ultralytics YOLO validation/test wrapper using the vendored ultralytics package.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np


def _repo_root() -> Path:
    # tools/ultralytics_val_detect.py -> repo root is parent of tools/
    return Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True, help="Path to .pt weights")
    p.add_argument("--data", type=str, required=True, help="Ultralytics dataset YAML path")
    p.add_argument("--split", type=str, default="test", help="Dataset split: test/val/train")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", type=str, default="0", help="CUDA device id(s) like '0'/'0,1' or 'cpu'")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--project", type=str, default=None, help="Output dir root (Ultralytics 'project=')")
    p.add_argument("--name", type=str, default=None, help="Run name (Ultralytics 'name=')")
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
    name = args.name or f"{Path(args.model).stem}-{args.split}"

    model = YOLO(args.model)
    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=project,
        name=name,
    )

    # Summarize metrics and write to output folder.
    save_dir = Path(project) / name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Params and FLOPs (best-effort).
    try:
        from ultralytics.utils.torch_utils import get_flops, get_num_params

        params = int(get_num_params(model.model))
        gflops = float(get_flops(model.model, args.imgsz) or 0.0)
    except Exception:
        params = None
        gflops = None

    # Overall metrics.
    mp = float(metrics.box.mp) if metrics else 0.0
    mr = float(metrics.box.mr) if metrics else 0.0
    map50 = float(metrics.box.map50) if metrics else 0.0
    map5095 = float(metrics.box.map) if metrics else 0.0
    if metrics and len(metrics.box.f1):
        mf1 = float(np.mean(metrics.box.f1))
    else:
        mf1 = float(2 * mp * mr / (mp + mr)) if (mp + mr) > 0 else 0.0

    summary = {
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "precision": mp,
        "recall": mr,
        "map50": map50,
        "map50_95": map5095,
        "f1": mf1,
        "params": params,
        "gflops": gflops,
    }

    # Per-class metrics.
    per_class = []
    names = metrics.names if metrics else {}
    for i in sorted(names.keys()):
        name_i = names[i]
        p_i = float(metrics.box.p[i]) if len(metrics.box.p) else 0.0
        r_i = float(metrics.box.r[i]) if len(metrics.box.r) else 0.0
        f1_i = float(metrics.box.f1[i]) if len(metrics.box.f1) else 0.0
        ap50_i = float(metrics.box.ap50[i]) if len(metrics.box.ap50) else 0.0
        ap_i = float(metrics.box.ap[i]) if len(metrics.box.ap) else 0.0
        per_class.append(
            {
                "class_id": int(i),
                "class_name": str(name_i),
                "precision": p_i,
                "recall": r_i,
                "f1": f1_i,
                "map50": ap50_i,
                "map50_95": ap_i,
            }
        )

    # Write JSON.
    with open(save_dir / f"metrics_{args.split}.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_class": per_class}, f, indent=2)

    # Write CSV (summary).
    with open(save_dir / f"metrics_{args.split}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)

    # Write CSV (per-class).
    if per_class:
        with open(save_dir / f"metrics_{args.split}_per_class.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(per_class[0].keys()))
            w.writeheader()
            w.writerows(per_class)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
