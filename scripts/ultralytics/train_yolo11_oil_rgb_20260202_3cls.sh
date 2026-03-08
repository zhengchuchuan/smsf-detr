#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export YOLO_CONFIG_DIR="$ROOT/tmp/ultralytics_config"

# ./scripts/ultralytics/prepare_oil_rgb_20260202_3cls.sh >/dev/null

# Timestamp suffix to avoid run name collisions.
TS="$(date +%Y%m%d-%H%M)"

# ~30M params baseline for YOLO11 (measured with this vendored Ultralytics):
# - yolo11l: ~25.4M params (closest to 30M among 11 n/s/m/l/x)
MODEL="${MODEL:-yolo11l.pt}"
DATA="${DATA:-$ROOT/configs/ultralytics/oil_rgb_20260202_3cls.yaml}"

EPOCHS="${EPOCHS:-300}"
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-16}"
DEVICE="${DEVICE:-2}"
WORKERS="${WORKERS:-8}"
SEED="${SEED:-42}"

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
  --name "oil_rgb_20260202_3cls_${MODEL%.*}-${TS}" \
  --cache
