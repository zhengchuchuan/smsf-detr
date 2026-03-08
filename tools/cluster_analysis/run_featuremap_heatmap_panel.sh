#!/usr/bin/env bash
set -euo pipefail

# One-shot pipeline: visualize *actual* RTMSFDETR feature maps (RGB+MSI) as heatmaps,
# and generate side-by-side panels (RGB+GT vs feature-heatmap overlay).
#
# It runs:
#   1) vis/featuremap_heatmap_rtmsfdetr.py
#      -> outputs/repr/rtmsfdetr/<run_name>/selection/{heatmap,overlay_heatmap}
#   2) tools/cluster_analysis/repr_oil_heatmap_panel.py
#      -> outputs/repr/rtmsfdetr/<run_name>/selection/panel
#
# Usage:
#   bash tools/cluster_analysis/run_featuremap_heatmap_panel.sh <train_run_dir_or_config.yaml> <dataset_root> [split] [device] [run_name]
#
# Examples:
#   bash tools/cluster_analysis/run_featuremap_heatmap_panel.sh \
#     outputs/oil_rgb_msi_20260101/rtmsfdetr/.../260107-210530-... \
#     data/oil_20260101 val cuda:0
#
#   bash tools/cluster_analysis/run_featuremap_heatmap_panel.sh \
#     outputs/oil_rgb_msi_20260101/rtmsfdetr/.../config.yaml \
#     data/oil_20260101 val cuda:0 oil_rgb_msi_val_featmap_fpn0
#
# Tunables (env vars):
#   FEAT_SOURCE=fpn|backbone          (default: fpn)
#   FEAT_LEVEL=0                     (default: 0)
#   HEAT_SOURCE=feat|enc_score       (default: feat; enc_score 更接近“oilness/objectness”)
#   CLASS_NAME=oil                   (default: oil; enc_score 且多类别时用于推导索引)
#   CLASS_INDEX=-1                   (default: -1; enc_score 时可显式指定类别索引)
#   REDUCE=l2|meanabs                (default: l2)
#   NORMALIZE=none|minmax|clip01     (default: minmax)
#   BLUR=5                           (default: 5; 0 disables blur; must be odd)
#   ALPHA=0.45                       (default: 0.45)
#   AMP=1                            (default: 1; 1 enables --amp)
#   BATCH_SIZE=1                     (default: 1)
#   IMG_SIZE=0                       (default: 0; 0 => auto by cfg or 640)
#   LIMIT=0                          (default: 0; 0 => no limit; passed to heatmap export)
#   STRICT_PAIRS=0                   (default: 0; 1 enables --strict-pairs)
#
#   PANEL_SHOW=oil|all               (default: oil)
#   DRAW_BOXES_ON_RIGHT=1            (default: 1)
#   PANEL_LIMIT=0                    (default: 0; 0 => no limit)
#
#   DRAW_PREDS=0                     (default: 0; 1 enables model predictions overlay)
#   PRED_SHOW=oil|all                (default: oil)
#   PRED_SCORE_THR=0.3               (default: 0.3)
#   PRED_MAX_DETS=100                (default: 100)
#   PRED_BOX_WIDTH=2                 (default: 2)
#   PRED_COLOR="0,255,255"           (default: cyan)
#   PRED_SHOW_SCORE=0                (default: 0; 1 draws score text)
#   PRED_BATCH_SIZE=1                (default: 1)
#   PRED_AMP=0                       (default: 0; 1 enables --pred-amp)
#   PRED_USE_EMA=0                   (default: 0; 1 prefers ema weights if present)
#
#   REPR_ROOT=outputs/repr/rtmsfdetr (default: outputs/repr/rtmsfdetr)
#   CHECKPOINT=...                   (optional override; default auto-pick from train_run_dir)
#   ANN=...                          (optional override; default dataset_root/annotations/<split>.json)
#
# Outputs:
#   - <run_dir>/selection/overlay_heatmap/*.png   heatmap overlay on RGB (resized to model input)
#   - <run_dir>/selection/panel/*.png             side-by-side panels with GT bboxes
#   - <run_dir>/pipeline_featmap.log              combined log

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

RUN_OR_CFG="${1:-}"
DATASET_ROOT="${2:-}"
SPLIT="${3:-val}"
DEVICE="${4:-cuda}"
RUN_NAME_ARG="${5:-}"

if [[ -z "${RUN_OR_CFG}" || -z "${DATASET_ROOT}" ]]; then
  echo "Usage: $0 <train_run_dir_or_config.yaml> <dataset_root> [split] [device] [run_name]" >&2
  exit 2
fi

DATASET_ROOT="${DATASET_ROOT%/}"

if [[ -d "${RUN_OR_CFG}" ]]; then
  TRAIN_RUN_DIR="${RUN_OR_CFG%/}"
  RESOLVED_CFG="${TRAIN_RUN_DIR}/config.yaml"
elif [[ -f "${RUN_OR_CFG}" ]]; then
  RESOLVED_CFG="${RUN_OR_CFG}"
  TRAIN_RUN_DIR="$(cd "$(dirname "${RESOLVED_CFG}")" && pwd)"
else
  echo "ERROR: not a directory or file: ${RUN_OR_CFG}" >&2
  exit 2
fi

if [[ ! -f "${RESOLVED_CFG}" ]]; then
  echo "ERROR: missing resolved config: ${RESOLVED_CFG}" >&2
  exit 2
fi

CHECKPOINT_PATH="${CHECKPOINT:-}"
if [[ -n "${CHECKPOINT_PATH}" ]]; then
  if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
    if [[ -f "${REPO_ROOT}/${CHECKPOINT_PATH}" ]]; then
      CHECKPOINT_PATH="${REPO_ROOT}/${CHECKPOINT_PATH}"
    else
      echo "ERROR: CHECKPOINT not found: ${CHECKPOINT_PATH}" >&2
      exit 2
    fi
  fi
else
  if [[ -f "${TRAIN_RUN_DIR}/checkpoint_best.pth" ]]; then
    CHECKPOINT_PATH="${TRAIN_RUN_DIR}/checkpoint_best.pth"
  elif [[ -f "${TRAIN_RUN_DIR}/checkpoint.pth" ]]; then
    CHECKPOINT_PATH="${TRAIN_RUN_DIR}/checkpoint.pth"
  else
    echo "ERROR: no checkpoint found in ${TRAIN_RUN_DIR} (checkpoint_best.pth / checkpoint.pth)." >&2
    echo "       Set CHECKPOINT=/path/to/ckpt.pth to override." >&2
    exit 2
  fi
fi

RGB_DIR="${RGB_DIR:-"${DATASET_ROOT}/rgb/${SPLIT}"}"
MSI_DIR="${MSI_DIR:-"${DATASET_ROOT}/msi/${SPLIT}"}"
ANN_PATH="${ANN:-"${DATASET_ROOT}/annotations/${SPLIT}.json"}"

if [[ ! -d "${RGB_DIR}" ]]; then
  echo "ERROR: missing RGB dir: ${RGB_DIR}" >&2
  exit 2
fi
if [[ ! -d "${MSI_DIR}" ]]; then
  echo "ERROR: missing MSI dir: ${MSI_DIR}" >&2
  exit 2
fi
if [[ ! -f "${ANN_PATH}" ]]; then
  echo "ERROR: missing COCO annotations: ${ANN_PATH}" >&2
  exit 2
fi

REPR_ROOT="${REPR_ROOT:-outputs/repr/rtmsfdetr}"
FEAT_SOURCE="${FEAT_SOURCE:-fpn}"
FEAT_LEVEL="${FEAT_LEVEL:-0}"
HEAT_SOURCE="${HEAT_SOURCE:-feat}"
CLASS_NAME="${CLASS_NAME:-oil}"
CLASS_INDEX="${CLASS_INDEX:--1}"
REDUCE="${REDUCE:-l2}"
NORMALIZE="${NORMALIZE:-minmax}"
BLUR="${BLUR:-5}"
ALPHA="${ALPHA:-0.45}"
AMP="${AMP:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
IMG_SIZE="${IMG_SIZE:-0}"
LIMIT="${LIMIT:-0}"
STRICT_PAIRS="${STRICT_PAIRS:-0}"

PANEL_SHOW="${PANEL_SHOW:-oil}"
DRAW_BOXES_ON_RIGHT="${DRAW_BOXES_ON_RIGHT:-1}"
PANEL_LIMIT="${PANEL_LIMIT:-0}"
DRAW_PREDS="${DRAW_PREDS:-0}"
PRED_SHOW="${PRED_SHOW:-oil}"
PRED_SCORE_THR="${PRED_SCORE_THR:-0.3}"
PRED_MAX_DETS="${PRED_MAX_DETS:-100}"
PRED_BOX_WIDTH="${PRED_BOX_WIDTH:-2}"
PRED_COLOR="${PRED_COLOR:-0,255,255}"
PRED_SHOW_SCORE="${PRED_SHOW_SCORE:-0}"
PRED_BATCH_SIZE="${PRED_BATCH_SIZE:-1}"
PRED_AMP="${PRED_AMP:-0}"
PRED_USE_EMA="${PRED_USE_EMA:-0}"

if [[ -n "${RUN_NAME_ARG}" ]]; then
  RUN_NAME="${RUN_NAME_ARG}"
else
  TS="$(date +%Y%m%d-%H%M%S)"
  RUN_NAME="featmap_${SPLIT}_${FEAT_SOURCE}${FEAT_LEVEL}_${TS}"
fi

RUN_DIR="${REPR_ROOT%/}/${RUN_NAME}"
mkdir -p "${RUN_DIR}"
LOG_FILE="${LOG_FILE:-"${RUN_DIR}/pipeline_featmap.log"}"
: > "${LOG_FILE}"
exec > >(tee -a "${LOG_FILE}") 2>&1

export PYTHONUNBUFFERED=1

echo "==== rtmsfdetr featuremap heatmap pipeline ===="
echo "repo_root      : ${REPO_ROOT}"
echo "resolved_cfg   : ${RESOLVED_CFG}"
echo "checkpoint     : ${CHECKPOINT_PATH}"
echo "dataset_root   : ${DATASET_ROOT}"
echo "split          : ${SPLIT}"
echo "rgb_dir        : ${RGB_DIR}"
echo "msi_dir        : ${MSI_DIR}"
echo "ann            : ${ANN_PATH}"
echo "device         : ${DEVICE}"
echo "run_dir        : ${RUN_DIR}"
echo "log_file       : ${LOG_FILE}"
echo "params         : feat=${FEAT_SOURCE}[${FEAT_LEVEL}] reduce=${REDUCE} norm=${NORMALIZE} blur=${BLUR} alpha=${ALPHA}"
echo "==============================================="

echo "[0/2] dependency check"
python - <<'PY'
import importlib.util
import sys

mods = ["torch", "torchvision", "numpy", "omegaconf", "PIL", "tifffile", "cv2"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    print("Missing python modules:", ", ".join(missing), file=sys.stderr)
    sys.exit(1)
print("OK")
PY

echo "[1/2] export featuremap heatmap -> ${RUN_DIR}/selection/overlay_heatmap"
CMD1=(python vis/featuremap_heatmap_rtmsfdetr.py
  --resolved-config "${RESOLVED_CFG}"
  --checkpoint "${CHECKPOINT_PATH}"
  --device "${DEVICE}"
  --rgb-dir "${RGB_DIR}"
  --msi-dir "${MSI_DIR}"
  --split "${SPLIT}"
  --batch-size "${BATCH_SIZE}"
  --img-size "${IMG_SIZE}"
  --feat-source "${FEAT_SOURCE}"
  --feat-level "${FEAT_LEVEL}"
  --heat-source "${HEAT_SOURCE}"
  --class-name "${CLASS_NAME}"
  --class-index "${CLASS_INDEX}"
  --reduce "${REDUCE}"
  --normalize "${NORMALIZE}"
  --blur "${BLUR}"
  --alpha "${ALPHA}"
  --output-dir "${REPR_ROOT}"
  --run-name "${RUN_NAME}"
)
if [[ "${AMP}" == "1" ]]; then
  CMD1+=(--amp)
fi
if [[ "${LIMIT}" != "0" ]]; then
  CMD1+=(--limit "${LIMIT}")
fi
if [[ "${STRICT_PAIRS}" == "1" ]]; then
  CMD1+=(--strict-pairs)
fi
printf 'cmd: %q ' "${CMD1[@]}"; echo
"${CMD1[@]}"

echo "[2/2] panel (RGB+GT vs feature-heatmap) -> ${RUN_DIR}/selection/panel"
TITLE_RIGHT="${TITLE_RIGHT:-"${FEAT_SOURCE}${FEAT_LEVEL} Feature Heatmap (RGB+MSI)"}"
TITLE_LEFT="${TITLE_LEFT:-"RGB + GT"}"

CMD2=(python tools/cluster_analysis/repr_oil_heatmap_panel.py
  --run-dir "${RUN_DIR}"
  --ann "${ANN_PATH}"
  --show "${PANEL_SHOW}"
  --title-right "${TITLE_RIGHT}"
  --title-left "${TITLE_LEFT}"
)
if [[ "${DRAW_BOXES_ON_RIGHT}" == "1" ]]; then
  CMD2+=(--draw-boxes-on-right)
fi
if [[ "${PANEL_LIMIT}" != "0" ]]; then
  CMD2+=(--limit "${PANEL_LIMIT}")
fi
if [[ "${DRAW_PREDS}" == "1" ]]; then
  CMD2+=(--draw-preds
    --pred-device "${DEVICE}"
    --pred-show "${PRED_SHOW}"
    --pred-score-thr "${PRED_SCORE_THR}"
    --pred-max-dets "${PRED_MAX_DETS}"
    --pred-box-width "${PRED_BOX_WIDTH}"
    --pred-color "${PRED_COLOR}"
    --pred-batch-size "${PRED_BATCH_SIZE}"
  )
  if [[ "${PRED_AMP}" == "1" ]]; then
    CMD2+=(--pred-amp)
  fi
  if [[ "${PRED_SHOW_SCORE}" == "1" ]]; then
    CMD2+=(--pred-show-score)
  fi
  if [[ "${PRED_USE_EMA}" == "1" ]]; then
    CMD2+=(--pred-use-ema)
  fi
fi
printf 'cmd: %q ' "${CMD2[@]}"; echo
"${CMD2[@]}"

echo "==== done ===="
echo "run_dir  : ${RUN_DIR}"
echo "heatmap  : ${RUN_DIR}/selection/overlay_heatmap"
echo "panel    : ${RUN_DIR}/selection/panel"
echo "log      : ${LOG_FILE}"
