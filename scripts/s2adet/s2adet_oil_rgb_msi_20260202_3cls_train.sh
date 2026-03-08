#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
S2ADET_ROOT="${REPO_ROOT}/third_party/S2ADet"

DATA="${DATA:-${REPO_ROOT}/third_party/S2ADet/data/hsi/oil_rgb_msi_20260202_train.yaml}"
DATA_TEST="${DATA_TEST:-${REPO_ROOT}/third_party/S2ADet/data/hsi/oil_rgb_msi_20260202_test.yaml}"
OUT_ROOT="${OUT_ROOT:-third_party/S2ADet/dataset/oil_20260202_3cls}"

CFG="${CFG:-${S2ADET_ROOT}/models/hsi/yolov5l_fusion_transformerx3_hsi.yaml}"
WEIGHTS="${WEIGHTS:-${S2ADET_ROOT}/yolo_weight/yolov5l.pt}"
HYP="${HYP:-${S2ADET_ROOT}/data/hyp.finetune.yaml}"

PROJECT="${PROJECT:-${REPO_ROOT}/outputs/s2adet/train/oil_rgb_20260202_3cls}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
NAME="${NAME:-exp_${RUN_TAG}}"
BATCH="${BATCH:-8}"
EPOCHS="${EPOCHS:-200}"
IMG="${IMG:-640}"
DEVICE="${DEVICE:-0}"
WORKERS="${WORKERS:-8}"
DRY_RUN="${DRY_RUN:-0}"
PREPARE="${PREPARE:-0}"
NOTEST="${NOTEST:-0}"
EXIST_OK="${EXIST_OK:-0}"

if [[ ! -d "${S2ADET_ROOT}" ]]; then
  echo "[ERROR] S2ADet root not found: ${S2ADET_ROOT}"
  exit 1
fi
if [[ ! -f "${CFG}" ]]; then
  echo "[ERROR] cfg not found: ${CFG}"
  exit 1
fi
if [[ ! -f "${WEIGHTS}" ]]; then
  echo "[ERROR] weights not found: ${WEIGHTS}"
  exit 1
fi
if [[ ! -f "${HYP}" ]]; then
  echo "[ERROR] hyp not found: ${HYP}"
  exit 1
fi

if [[ "${PREPARE}" == "1" || ! -f "${DATA}" || ! -f "${DATA_TEST}" ]]; then
  echo "[INFO] preparing S2ADet oil dataset and yaml files..."
  bash "${REPO_ROOT}/scripts/s2adet/s2adet_prepare_dataset.sh" \
    --out-root "${OUT_ROOT}" \
    --data-yaml "third_party/S2ADet/data/hsi/oil_rgb_msi_20260202_train.yaml" \
    --data-yaml-test "third_party/S2ADet/data/hsi/oil_rgb_msi_20260202_test.yaml"
fi

if [[ ! -f "${DATA}" ]]; then
  echo "[ERROR] train data yaml not found: ${DATA}"
  exit 1
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[INFO] data yaml: ${DATA}"
  echo "[INFO] run dir: ${PROJECT}/${NAME}"
  echo "[INFO] DRY_RUN=1, skip training."
  exit 0
fi

mkdir -p "${PROJECT}"
cd "${S2ADET_ROOT}"

EXTRA_ARGS=()
if [[ "${NOTEST}" == "1" ]]; then
  EXTRA_ARGS+=(--notest)
fi
if [[ "${EXIST_OK}" == "1" ]]; then
  EXTRA_ARGS+=(--exist-ok)
fi

echo "[INFO] run dir: ${PROJECT}/${NAME}"

python train.py \
  --data "${DATA}" \
  --cfg "${CFG}" \
  --weights "${WEIGHTS}" \
  --hyp "${HYP}" \
  --img-size "${IMG}" "${IMG}" \
  --batch-size "${BATCH}" \
  --epochs "${EPOCHS}" \
  --workers "${WORKERS}" \
  --device "${DEVICE}" \
  --project "${PROJECT}" \
  --name "${NAME}" \
  "${EXTRA_ARGS[@]}"
