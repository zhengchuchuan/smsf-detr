#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

DATA_ROOT="$ROOT/data/HOD3K"
RGB_ROOT="$DATA_ROOT/rgb"
ANN_ROOT="$DATA_ROOT/annotations"
OUT_ROOT="$DATA_ROOT/ultralytics_rgb_HOD3K"

mkdir -p "$OUT_ROOT/images" "$OUT_ROOT/labels"

for split in train val test; do
  python tools/ultralytics_prepare_coco.py \
    --ann "$ANN_ROOT/${split}.json" \
    --images-dir "$RGB_ROOT/$split" \
    --out-images "$OUT_ROOT/images/$split" \
    --out-labels "$OUT_ROOT/labels/$split" \
    --class-names "people,bike,car"
done

echo "Prepared Ultralytics dataset wrapper at: $OUT_ROOT"

