#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export YOLO_CONFIG_DIR="$ROOT/tmp/ultralytics_config"

# ./scripts/ultralytics/prepare_oil_rgb_20260202_3cls.sh >/dev/null

# Timestamp suffix to avoid run name collisions.
TS="$(date +%Y%m%d-%H%M)"

# Default to a recent YOLO11L run if not specified.
MODEL="${MODEL:-$ROOT/outputs/ultralytics/oil_rgb_20260202_3cls_yolo11l-20260208-2131/weights/best.pt}"
DATA="${DATA:-$ROOT/configs/ultralytics/oil_rgb_20260202_3cls.yaml}"
SPLIT="${SPLIT:-test}"

IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-16}"
DEVICE="${DEVICE:-0}"
WORKERS="${WORKERS:-8}"

RUN_DIR="$(cd "$(dirname "$MODEL")/.." && pwd)"
PROJECT="${PROJECT:-$RUN_DIR}"

BASE="$(basename "${MODEL%.*}")"
RUN_NAME="${RUN_NAME:-test-${BASE}-${SPLIT}-${TS}}"

python tools/ultralytics_val_detect.py \
  --model "$MODEL" \
  --data "$DATA" \
  --split "$SPLIT" \
  --imgsz "$IMGSZ" \
  --batch "$BATCH" \
  --device "$DEVICE" \
  --workers "$WORKERS" \
  --project "$PROJECT" \
  --name "$RUN_NAME"
