#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
S2ADET_ROOT="${REPO_ROOT}/third_party/S2ADet"

DATA="${DATA:-${REPO_ROOT}/third_party/S2ADet/data/hsi/oil_rgb_msi_20260202_test.yaml}"
TRAIN_PROJECT="${TRAIN_PROJECT:-${REPO_ROOT}/outputs/s2adet/train/oil_rgb_20260202_3cls}"
TRAIN_NAME="${TRAIN_NAME:-}"
if [[ -z "${WEIGHTS:-}" ]]; then
  if [[ -n "${TRAIN_NAME}" ]]; then
    WEIGHTS="${TRAIN_PROJECT}/${TRAIN_NAME}/weights/best.pt"
  else
    LATEST_RUN="$(ls -1dt "${TRAIN_PROJECT}"/* 2>/dev/null | head -n 1 || true)"
    if [[ -z "${LATEST_RUN}" ]]; then
      echo "[ERROR] no run found under: ${TRAIN_PROJECT}"
      exit 1
    fi
    WEIGHTS="${LATEST_RUN}/weights/best.pt"
    echo "[INFO] auto-selected latest run: ${LATEST_RUN}"
  fi
fi
if [[ -z "${WEIGHTS}" ]]; then
  echo "[ERROR] failed to resolve weights path"
  exit 1
fi

PROJECT="${PROJECT:-${REPO_ROOT}/outputs/s2adet/test/oil_rgb_msi_20260202_3cls}"
NAME="${NAME:-test}"
BATCH="${BATCH:-32}"
IMG="${IMG:-640}"
DEVICE="${DEVICE:-0}"
LOG_FILE="${LOG_FILE:-/tmp/s2adet_oil_eval.log}"

if [[ ! -d "${S2ADET_ROOT}" ]]; then
  echo "[ERROR] S2ADet root not found: ${S2ADET_ROOT}"
  exit 1
fi
if [[ ! -f "${DATA}" ]]; then
  echo "[ERROR] data yaml not found: ${DATA}"
  exit 1
fi
if [[ ! -f "${WEIGHTS}" ]]; then
  echo "[ERROR] weights not found: ${WEIGHTS}"
  exit 1
fi

mkdir -p "${PROJECT}"
cd "${S2ADET_ROOT}"

python test.py \
  --data "${DATA}" \
  --weights "${WEIGHTS}" \
  --img-size "${IMG}" \
  --batch-size "${BATCH}" \
  --device "${DEVICE}" \
  --project "${PROJECT}" \
  --name "${NAME}" \
  | tee "${LOG_FILE}"

LOG_FILE="${LOG_FILE}" python - <<'PY'
import os
from pathlib import Path

log = Path(os.environ["LOG_FILE"])
if not log.exists():
    raise SystemExit(f"log file not found: {log}")
line = None
for text in reversed(log.read_text().splitlines()):
    if text.strip().startswith("all"):
        line = text
        break
if not line:
    raise SystemExit("metric line not found")
parts = line.split()
p = float(parts[3])
r = float(parts[4])
map50 = float(parts[5])
map5095 = float(parts[7])
f1 = 2 * p * r / (p + r + 1e-12)
print("\nComputed metrics:")
print(f"P={p:.6f} R={r:.6f} F1={f1:.6f} mAP@0.5={map50:.6f} mAP@0.5:0.95={map5095:.6f}")
PY
