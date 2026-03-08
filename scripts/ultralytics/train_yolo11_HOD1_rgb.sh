#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export YOLO_CONFIG_DIR="$ROOT/tmp/ultralytics_config"

# ./scripts/ultralytics/prepare_HOD1_rgb.sh >/dev/null

# Timestamp suffix to avoid run name collisions.
TS="$(date +%Y%m%d-%H%M)"

MODEL="${MODEL:-yolo11l.pt}"
DATA="${DATA:-$ROOT/configs/ultralytics/HOD1_rgb.yaml}"

EPOCHS="${EPOCHS:-100}"
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-16}"
DEVICE="${DEVICE:-0}"
WORKERS="${WORKERS:-8}"
SEED="${SEED:-0}"

python tools/ultralytics_train_detect.py \
  --model "$MODEL" \
  --data "$DATA" \
  --epochs "$EPOCHS" \
  --imgsz "$IMGSZ" \
  --batch "$BATCH" \
  --device "$DEVICE" \
  --workers "$WORKERS" \
  --seed "$SEED" \
  --project "$ROOT/outputs/ultralytics" \
  --name "HOD1_rgb_${MODEL%.*}-${TS}" \
  --cache

