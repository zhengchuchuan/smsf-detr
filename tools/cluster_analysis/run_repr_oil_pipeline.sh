#!/usr/bin/env bash
set -euo pipefail

# One-shot pipeline for tools/cluster_analysis:
#   1) export dense features (per-image npz)
#   2) two-stage clustering + pseudo-color/overlay
#   3) auto-select "oil clusters" from COCO bbox
#   4) apply selected clusters -> binary mask + overlay
#   5) generate oil heatmap + overlay
#   6) generate side-by-side panel (RGB+GT vs heatmap)
#   7) generate analysis charts/reports
#
# Usage:
#   bash tools/cluster_analysis/run_repr_oil_pipeline.sh <train_run_dir> <dataset_root> [split] [device] [run_name]
#
# Args:
#   train_run_dir : outputs/<run_dir> that contains config.yaml and checkpoint_best.pth (or checkpoint.pth)
#   dataset_root  : data/<dataset> that contains rgb/<split>/ and annotations/<split>.json
#   split         : train|val|test (default: val)
#   device        : cuda|cuda:0|cpu (default: cuda)
#   run_name      : output subdir name under outputs/repr/rtmsfdetr (default: auto)
#
# Tunables (env vars):
#   FEAT_SOURCE=fpn|backbone          (default: fpn)
#   FEAT_LEVEL=0                     (default: 0)
#   K1=32                            (default: 32)
#   K2=16                            (default: 16)
#   SAMPLE_PER_IMAGE=2000            (default: 2000)
#   MBK_BATCH_SIZE=4096              (default: 4096)
#   L2NORM=1                         (default: 1; 1 enables --l2norm)
#   ALPHA=0.45                       (default: 0.45)
#   AMP=1                            (default: 1; 1 enables --amp in export)
#   BATCH_SIZE=1                     (default: 1)
#   IMG_SIZE=0                       (default: 0; 0 => auto)
#   SAVE_DTYPE=float16|float32       (default: float16)
#   LIMIT=0                          (default: 0; 0 => no limit; passed to export step)
#   RECURSIVE=0                      (default: 0; 1 enables --recursive in export)
#   CHECKPOINT=...                   (optional override; default auto-pick from train_run_dir)
#
#   AUTO_METRIC=iou|f1               (default: iou)
#   AUTO_MAX_CLUSTERS=32             (default: 32)
#   AUTO_MIN_SUPPORT=50              (default: 50)
#   AUTO_MIN_TP=10                   (default: 10)
#   OIL_NAMES="oil"                  (default: oil; space-separated list for COCO categories.name)
#
#   HEATMAP_SCORE_KEY=precision|coverage|lift|selected  (default: precision)
#   HEATMAP_NORMALIZE=none|minmax|clip01                (default: minmax)
#   HEATMAP_BLUR=5                                      (default: 5; 0 disables blur)
#
#   PANEL_SHOW=oil|all               (default: oil)
#   DRAW_BOXES_ON_RIGHT=1            (default: 1)
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
#   TOP_N=30                         (default: 30; analysis top-n charts)
#   ANALYSIS_LIMIT=0                 (default: 0; 0 => no limit)
#
# Outputs (under $RUN_DIR):
#   - viz/                    clustering pseudo-color + overlay
#   - selection/              selected clusters, masks, heatmaps, panels
#   - analysis/analysis_summary.json  one-page index of charts & key stats

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TRAIN_RUN_DIR="${1:-}"
DATASET_ROOT="${2:-}"
SPLIT="${3:-val}"
DEVICE="${4:-cuda}"
RUN_NAME_ARG="${5:-}"

if [[ -z "${TRAIN_RUN_DIR}" || -z "${DATASET_ROOT}" ]]; then
  echo "Usage: $0 <train_run_dir> <dataset_root> [split] [device] [run_name]" >&2
  echo "Example:" >&2
  echo "  bash $0 outputs/<run_dir> data/oil_20260101 val cuda:0" >&2
  exit 2
fi

TRAIN_RUN_DIR="${TRAIN_RUN_DIR%/}"
DATASET_ROOT="${DATASET_ROOT%/}"

RESOLVED_CFG="${TRAIN_RUN_DIR}/config.yaml"
if [[ ! -f "${RESOLVED_CFG}" ]]; then
  echo "ERROR: missing config.yaml: ${RESOLVED_CFG}" >&2
  exit 2
fi

CHECKPOINT_PATH="${CHECKPOINT:-}"
if [[ -n "${CHECKPOINT_PATH}" ]]; then
  if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
    # allow relative to repo root
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

IMG_DIR="${IMG_DIR:-"${DATASET_ROOT}/rgb/${SPLIT}"}"
ANN_PATH="${ANN:-"${DATASET_ROOT}/annotations/${SPLIT}.json"}"
if [[ ! -d "${IMG_DIR}" ]]; then
  echo "ERROR: missing image dir: ${IMG_DIR}" >&2
  exit 2
fi
if [[ ! -f "${ANN_PATH}" ]]; then
  echo "ERROR: missing COCO annotations: ${ANN_PATH}" >&2
  exit 2
fi

REPR_ROOT="${REPR_ROOT:-outputs/repr/rtmsfdetr}"
FEAT_SOURCE="${FEAT_SOURCE:-fpn}"
FEAT_LEVEL="${FEAT_LEVEL:-0}"
K1="${K1:-32}"
K2="${K2:-16}"
SAMPLE_PER_IMAGE="${SAMPLE_PER_IMAGE:-2000}"
MBK_BATCH_SIZE="${MBK_BATCH_SIZE:-4096}"
L2NORM="${L2NORM:-1}"
ALPHA="${ALPHA:-0.45}"
AMP="${AMP:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
IMG_SIZE="${IMG_SIZE:-0}"
SAVE_DTYPE="${SAVE_DTYPE:-float16}"
LIMIT="${LIMIT:-0}"
RECURSIVE="${RECURSIVE:-0}"

AUTO_METRIC="${AUTO_METRIC:-iou}"
AUTO_MAX_CLUSTERS="${AUTO_MAX_CLUSTERS:-32}"
AUTO_MIN_SUPPORT="${AUTO_MIN_SUPPORT:-50}"
AUTO_MIN_TP="${AUTO_MIN_TP:-10}"
OIL_NAMES="${OIL_NAMES:-oil}"

HEATMAP_SCORE_KEY="${HEATMAP_SCORE_KEY:-precision}"
HEATMAP_NORMALIZE="${HEATMAP_NORMALIZE:-minmax}"
HEATMAP_BLUR="${HEATMAP_BLUR:-5}"

PANEL_SHOW="${PANEL_SHOW:-oil}"
DRAW_BOXES_ON_RIGHT="${DRAW_BOXES_ON_RIGHT:-1}"
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

TOP_N="${TOP_N:-30}"
ANALYSIS_LIMIT="${ANALYSIS_LIMIT:-0}"

if [[ -n "${RUN_NAME_ARG}" ]]; then
  RUN_NAME="${RUN_NAME_ARG}"
else
  TS="$(date +%Y%m%d-%H%M%S)"
  RUN_NAME="oil_${SPLIT}_${FEAT_SOURCE}${FEAT_LEVEL}_k${K1}x${K2}_${TS}"
fi

RUN_DIR="${REPR_ROOT%/}/${RUN_NAME}"
mkdir -p "${RUN_DIR}"
LOG_FILE="${LOG_FILE:-"${RUN_DIR}/pipeline.log"}"
: > "${LOG_FILE}"
exec > >(tee -a "${LOG_FILE}") 2>&1

export PYTHONUNBUFFERED=1

echo "==== repr oil pipeline ===="
echo "repo_root      : ${REPO_ROOT}"
echo "train_run_dir  : ${TRAIN_RUN_DIR}"
echo "resolved_cfg   : ${RESOLVED_CFG}"
echo "checkpoint     : ${CHECKPOINT_PATH}"
echo "dataset_root   : ${DATASET_ROOT}"
echo "split          : ${SPLIT}"
echo "img_dir        : ${IMG_DIR}"
echo "ann            : ${ANN_PATH}"
echo "device         : ${DEVICE}"
echo "output_run_dir : ${RUN_DIR}"
echo "log_file       : ${LOG_FILE}"
echo "params         : feat=${FEAT_SOURCE}[${FEAT_LEVEL}] k1=${K1} k2=${K2} l2norm=${L2NORM} sample_per_image=${SAMPLE_PER_IMAGE}"
echo "==========================="

echo "[0/7] dependency check"
python - <<'PY'
import importlib.util
import sys

mods = [
    "torch",
    "torchvision",
    "numpy",
    "omegaconf",
    "PIL",
    "sklearn",
    "joblib",
    "cv2",
    "pandas",
    "matplotlib",
]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    print("Missing python modules:", ", ".join(missing), file=sys.stderr)
    sys.exit(1)
print("OK")
PY

OIL_NAMES_ARR=()
read -r -a OIL_NAMES_ARR <<< "${OIL_NAMES}"

echo "[1/7] export dense features -> ${RUN_DIR}/features"
EXPORT_CMD=(python tools/cluster_analysis/repr_export_rtmsfdetr.py
  --resolved-config "${RESOLVED_CFG}"
  --checkpoint "${CHECKPOINT_PATH}"
  --device "${DEVICE}"
  --input-dir "${IMG_DIR}"
  --split "${SPLIT}"
  --batch-size "${BATCH_SIZE}"
  --img-size "${IMG_SIZE}"
  --feat-source "${FEAT_SOURCE}"
  --feat-level "${FEAT_LEVEL}"
  --save-dtype "${SAVE_DTYPE}"
  --output-dir "${REPR_ROOT}"
  --run-name "${RUN_NAME}"
)
if [[ "${AMP}" == "1" ]]; then
  EXPORT_CMD+=(--amp)
fi
if [[ "${RECURSIVE}" == "1" ]]; then
  EXPORT_CMD+=(--recursive)
fi
if [[ "${LIMIT}" != "0" ]]; then
  EXPORT_CMD+=(--limit "${LIMIT}")
fi
printf 'cmd: %q ' "${EXPORT_CMD[@]}"; echo
"${EXPORT_CMD[@]}"

echo "[2/7] two-stage clustering -> ${RUN_DIR}/clusters + ${RUN_DIR}/labels + ${RUN_DIR}/viz"
CLUSTER_CMD=(python tools/cluster_analysis/repr_cluster_rtmsfdetr.py
  --run-dir "${RUN_DIR}"
  --k1 "${K1}"
  --k2 "${K2}"
  --sample-per-image "${SAMPLE_PER_IMAGE}"
  --mbk-batch-size "${MBK_BATCH_SIZE}"
  --save-vis
  --alpha "${ALPHA}"
)
if [[ "${L2NORM}" == "1" ]]; then
  CLUSTER_CMD+=(--l2norm)
fi
printf 'cmd: %q ' "${CLUSTER_CMD[@]}"; echo
"${CLUSTER_CMD[@]}"

echo "[3/7] auto-select oil clusters -> ${RUN_DIR}/selection/selected_clusters.json"
AUTO_CMD=(python tools/cluster_analysis/repr_auto_select_oil_clusters.py
  --run-dir "${RUN_DIR}"
  --ann "${ANN_PATH}"
  --metric "${AUTO_METRIC}"
  --max-clusters "${AUTO_MAX_CLUSTERS}"
  --min-support "${AUTO_MIN_SUPPORT}"
  --min-tp "${AUTO_MIN_TP}"
)
if [[ "${#OIL_NAMES_ARR[@]}" -gt 0 ]]; then
  AUTO_CMD+=(--oil-names "${OIL_NAMES_ARR[@]}")
fi
printf 'cmd: %q ' "${AUTO_CMD[@]}"; echo
"${AUTO_CMD[@]}"

echo "[4/7] apply selected clusters -> ${RUN_DIR}/selection/pseudo_mask + overlay"
python tools/cluster_analysis/repr_apply_oil_clusters.py --run-dir "${RUN_DIR}" --alpha "${ALPHA}"

echo "[5/7] oil heatmap -> ${RUN_DIR}/selection/heatmap + overlay_heatmap"
python tools/cluster_analysis/repr_oil_heatmap.py \
  --run-dir "${RUN_DIR}" \
  --score-key "${HEATMAP_SCORE_KEY}" \
  --normalize "${HEATMAP_NORMALIZE}" \
  --blur "${HEATMAP_BLUR}" \
  --alpha "${ALPHA}"

echo "[6/7] panel (RGB+GT vs heatmap) -> ${RUN_DIR}/selection/panel"
PANEL_CMD=(python tools/cluster_analysis/repr_oil_heatmap_panel.py
  --run-dir "${RUN_DIR}"
  --ann "${ANN_PATH}"
  --show "${PANEL_SHOW}"
)
if [[ "${DRAW_BOXES_ON_RIGHT}" == "1" ]]; then
  PANEL_CMD+=(--draw-boxes-on-right)
fi
if [[ "${#OIL_NAMES_ARR[@]}" -gt 0 ]]; then
  PANEL_CMD+=(--oil-names "${OIL_NAMES_ARR[@]}")
fi
if [[ "${DRAW_PREDS}" == "1" ]]; then
  PANEL_CMD+=(--draw-preds
    --pred-device "${DEVICE}"
    --pred-show "${PRED_SHOW}"
    --pred-score-thr "${PRED_SCORE_THR}"
    --pred-max-dets "${PRED_MAX_DETS}"
    --pred-box-width "${PRED_BOX_WIDTH}"
    --pred-color "${PRED_COLOR}"
    --pred-batch-size "${PRED_BATCH_SIZE}"
  )
  if [[ "${PRED_AMP}" == "1" ]]; then
    PANEL_CMD+=(--pred-amp)
  fi
  if [[ "${PRED_SHOW_SCORE}" == "1" ]]; then
    PANEL_CMD+=(--pred-show-score)
  fi
  if [[ "${PRED_USE_EMA}" == "1" ]]; then
    PANEL_CMD+=(--pred-use-ema)
  fi
fi
printf 'cmd: %q ' "${PANEL_CMD[@]}"; echo
"${PANEL_CMD[@]}"

echo "[7/7] analysis charts/reports -> ${RUN_DIR}/analysis"
ANALYSIS_CMD=(python tools/cluster_analysis/repr_analysis_viz.py
  --run-dir "${RUN_DIR}"
  --ann "${ANN_PATH}"
  --top-n "${TOP_N}"
)
if [[ "${ANALYSIS_LIMIT}" != "0" ]]; then
  ANALYSIS_CMD+=(--limit "${ANALYSIS_LIMIT}")
fi
printf 'cmd: %q ' "${ANALYSIS_CMD[@]}"; echo
"${ANALYSIS_CMD[@]}"

echo "==== done ===="
echo "run_dir  : ${RUN_DIR}"
echo "viz      : ${RUN_DIR}/viz/overlay"
echo "mask     : ${RUN_DIR}/selection/pseudo_mask"
echo "heatmap  : ${RUN_DIR}/selection/overlay_heatmap"
echo "panel    : ${RUN_DIR}/selection/panel"
echo "analysis : ${RUN_DIR}/analysis/analysis_summary.json"
echo "log      : ${LOG_FILE}"
