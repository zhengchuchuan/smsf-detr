#!/usr/bin/env bash
set -euo pipefail

# Visualize feature/score heatmaps on TEST split using vis/featuremap_heatmap_rtmsfdetr.py.
# This script uses the same manual image loading + normalization logic as that visualization script
# (i.e., NOT BaseTrainer/datasets pipeline), which is useful for sanity-checking preprocessing.
#
# Default preset (what you asked for):
# - Decoder cross-attention class heatmap ("attn_cam")
# - Target class: oil (class_index=0)
# - Decoder layer: last (-1)
# - Query selection: top-50 queries with score >= 0.2
# You can override any of these via env vars below.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

RUN_DIR="${RUN_DIR:-${REPO_ROOT}/outputs/oil_msi_20260202_3cls/rtmsfdetr/rtv4_hgnetv2_m_msi7/baseline5/260204-131232-rtmsfdetr_oil_msi_20260202_det_rtv4_hgnetv2_m}"

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/oil_20260202}"
SPLIT="test"

RGB_DIR="${RGB_DIR:-${DATA_ROOT}/rgb/${SPLIT}}"
MSI_DIR="${MSI_DIR:-${DATA_ROOT}/msi/${SPLIT}}"

# Output root; actual run will be under ${OUTPUT_DIR}/${RUN_NAME}.
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR%/}/feature_map}"
TIME_STAMP="$(date +%Y%m%d-%H%M)"
RUN_NAME="${RUN_NAME:-oil_20260202_${SPLIT}_attncam}"
RUN_NAME="${RUN_NAME}-${TIME_STAMP}"

DEVICE="${DEVICE:-cuda}"
AMP="${AMP:-0}"
USE_EMA="${USE_EMA:-1}"
IMG_SIZE="${IMG_SIZE:-0}"
LIMIT="${LIMIT:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
RECURSIVE="${RECURSIVE:-0}"
STRICT_PAIRS="${STRICT_PAIRS:-0}"
OPTS="${OPTS:-model.disable_distill=true}"

# Heatmap settings (override via env)
FEAT_SOURCE="${FEAT_SOURCE:-fpn}"          # fpn/backbone
FEAT_LEVEL="${FEAT_LEVEL:-0}"              # 0/1/2 -> stride 8/16/32 (typically)
HEAT_SOURCE="${HEAT_SOURCE:-attn_cam}"     # feat/attn_cam
CLASS_NAME="${CLASS_NAME:-oil}"
CLASS_INDEX="${CLASS_INDEX:-0}"
NORMALIZE="${NORMALIZE:-minmax}"
BLUR="${BLUR:-5}"
ALPHA="${ALPHA:-0.45}"

ATTN_LAYER="${ATTN_LAYER:--1}"
ATTN_TOPK="${ATTN_TOPK:-50}"
ATTN_SCORE_THR="${ATTN_SCORE_THR:-0.2}"

GRADCAM_TOPK="${GRADCAM_TOPK:-100}"
GRADCAM_SCORE_THR="${GRADCAM_SCORE_THR:-0.0}"
GRADCAM_SOURCE="${GRADCAM_SOURCE:-enc_score}"
GRADCAM_SCORE_MODE="${GRADCAM_SCORE_MODE:-logit}"
GRADCAM_BACKWARD_TYPE="${GRADCAM_BACKWARD_TYPE:-class}"
GRADCAM_CONF_THR="${GRADCAM_CONF_THR:-0.01}"

DET_SCORE_THR="${DET_SCORE_THR:-0.25}"
DET_TOPK="${DET_TOPK:-100}"
DET_AGG="${DET_AGG:-max}"
DET_SCORE_MODE="${DET_SCORE_MODE:-sigmoid}"

RESOLVED_CFG="${RUN_DIR%/}/config.yaml"
if [[ ! -f "${RESOLVED_CFG}" ]]; then
  echo "ERROR: missing resolved config: ${RESOLVED_CFG}" >&2
  exit 2
fi

if [[ -f "${RUN_DIR%/}/checkpoint_best.pth" ]]; then
  CHECKPOINT_PATH="${RUN_DIR%/}/checkpoint_best.pth"
elif [[ -f "${RUN_DIR%/}/checkpoint.pth" ]]; then
  CHECKPOINT_PATH="${RUN_DIR%/}/checkpoint.pth"
else
  echo "ERROR: no checkpoint found in ${RUN_DIR}" >&2
  exit 2
fi

if [[ ! -d "${RGB_DIR}" ]]; then
  echo "ERROR: missing RGB dir: ${RGB_DIR}" >&2
  exit 2
fi
if [[ ! -d "${MSI_DIR}" ]]; then
  echo "ERROR: missing MSI dir: ${MSI_DIR}" >&2
  exit 2
fi

CMD=(python "${REPO_ROOT}/vis/featuremap_heatmap_rtmsfdetr.py"
  --resolved-config "${RESOLVED_CFG}"
  --checkpoint "${CHECKPOINT_PATH}"
  --rgb-dir "${RGB_DIR}"
  --msi-dir "${MSI_DIR}"
  --split "${SPLIT}"
  --device "${DEVICE}"
  --batch-size "${BATCH_SIZE}"
  --img-size "${IMG_SIZE}"
  --feat-source "${FEAT_SOURCE}"
  --feat-level "${FEAT_LEVEL}"
  --heat-source "${HEAT_SOURCE}"
  --class-name "${CLASS_NAME}"
  --class-index "${CLASS_INDEX}"
  --attn-layer "${ATTN_LAYER}"
  --attn-topk "${ATTN_TOPK}"
  --attn-score-thr "${ATTN_SCORE_THR}"
  --gradcam-topk "${GRADCAM_TOPK}"
  --gradcam-score-thr "${GRADCAM_SCORE_THR}"
  --gradcam-source "${GRADCAM_SOURCE}"
  --gradcam-score-mode "${GRADCAM_SCORE_MODE}"
  --gradcam-backward-type "${GRADCAM_BACKWARD_TYPE}"
  --gradcam-conf-thr "${GRADCAM_CONF_THR}"
  --det-score-thr "${DET_SCORE_THR}"
  --det-topk "${DET_TOPK}"
  --det-agg "${DET_AGG}"
  --det-score-mode "${DET_SCORE_MODE}"
  --normalize "${NORMALIZE}"
  --blur "${BLUR}"
  --alpha "${ALPHA}"
  --output-dir "${OUTPUT_DIR}"
  --run-name "${RUN_NAME}"
)

if [[ -n "${OPTS}" ]]; then
  # Split by spaces: e.g. OPTS="model.disable_distill=true data.ms_center_to_rgb_range=false"
  read -r -a _opts_arr <<< "${OPTS}"
  CMD+=(--opts "${_opts_arr[@]}")
fi
if [[ "${USE_EMA}" == "1" ]]; then
  CMD+=(--use-ema)
fi

if [[ "${LIMIT}" != "0" ]]; then
  CMD+=(--limit "${LIMIT}")
fi
if [[ "${AMP}" == "1" ]]; then
  CMD+=(--amp)
fi
if [[ "${RECURSIVE}" == "1" ]]; then
  CMD+=(--recursive)
fi
if [[ "${STRICT_PAIRS}" == "1" ]]; then
  CMD+=(--strict-pairs)
fi

exec "${CMD[@]}"
