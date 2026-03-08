#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

DATA_ROOT="$ROOT/data/HOD-1"
RGB_ROOT="$DATA_ROOT/rgb"
ANN_ROOT="$DATA_ROOT/annotations"
OUT_ROOT="$DATA_ROOT/ultralytics_rgb_HOD1_v2"

mkdir -p "$OUT_ROOT/images" "$OUT_ROOT/labels"

for split in train val test; do
  python tools/ultralytics_prepare_coco.py \
    --ann "$ANN_ROOT/${split}.json" \
    --images-dir "$RGB_ROOT/$split" \
    --out-images "$OUT_ROOT/images/$split" \
    --out-labels "$OUT_ROOT/labels/$split" \
    --class-names "Toyblock_screen,Photo_screen,Pen_screen,Photo_real,Toyblock_real,Pen_real,Leaf_screen,Leaf_real"
done

echo "Prepared Ultralytics dataset wrapper at: $OUT_ROOT"

