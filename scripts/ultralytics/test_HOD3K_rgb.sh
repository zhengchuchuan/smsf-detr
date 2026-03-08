#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export YOLO_CONFIG_DIR="$ROOT/tmp/ultralytics_config"

# ./scripts/ultralytics/prepare_HOD3K_rgb.sh >/dev/null

# Timestamp suffix to avoid run name collisions.
TS="$(date +%Y%m%d-%H%M)"

# If MODEL not set, pick the latest HOD3K run under outputs/ultralytics.
if [[ -z "${MODEL:-}" ]]; then
  MODEL="$(ls -t "$ROOT"/outputs/ultralytics/HOD3K_rgb_*/weights/best.pt 2>/dev/null | head -n 1 || true)"
fi
if [[ -z "$MODEL" ]]; then
  echo "MODEL not found. Please set MODEL=/path/to/weights/best.pt" 1>&2
  exit 2
fi

DATA="${DATA:-$ROOT/configs/ultralytics/HOD3K_rgb.yaml}"
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

