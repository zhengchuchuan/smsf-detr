#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="outputs/HOD3K_rgb_msi/rtmsfdetr/rtv4_hgnetv2_m_distill_det_dualstream_c2former_postblock_add_wbadd_c3c4c5_msbandsep_c2align_infonce_reg_globalkv_pos2d/260203-131159-rtmsfdetr_hod3k_rgb_msi_20260115_det_rtv4_hgnetv2_m_distill_dualstream_c2former_postblock_add_wbadd_c3c4c5_msbandsep_c2align_infonce_reg_globalkv_pos2d"

CKPT_BEST="${RUN_DIR}/checkpoint_best.pth"
CKPT_LAST="${RUN_DIR}/checkpoint.pth"

if [[ -f "${CKPT_BEST}" ]]; then
  CKPT="${CKPT_BEST}"
elif [[ -f "${CKPT_LAST}" ]]; then
  CKPT="${CKPT_LAST}"
else
  echo "[ERROR] No checkpoint found in ${RUN_DIR}" >&2
  exit 1
fi

DEVICE="${DEVICE:-cuda}"
SPLIT="${SPLIT:-test}"
SCORE_THR="${SCORE_THR:-0.5}"
MAX_DETS="${MAX_DETS:-100}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SAVE_MSI_VIS="${SAVE_MSI_VIS:-1}"
MSI_CHANNEL="${MSI_CHANNEL:--1}"   # -1 默认中间通道

# Provide either DATASET_DIR or both RGB_DIR/MSI_DIR.
DATASET_DIR="${DATASET_DIR:-"data/HOD3K"}"
RGB_DIR="${RGB_DIR:-}"
MSI_DIR="${MSI_DIR:-}"

OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}/infer_dir}"

ARGS=(
  --config "${RUN_DIR}/config.yaml"
  --checkpoint "${CKPT}"
  --device "${DEVICE}"
  --split "${SPLIT}"
  --batch-size "${BATCH_SIZE}"
  --score-thr "${SCORE_THR}"
  --max-dets "${MAX_DETS}"
  --save-vis
  --output-dir "${OUTPUT_DIR}"
)
if [[ "${SAVE_MSI_VIS}" != "0" ]]; then
  ARGS+=(--save-msi-vis --msi-channel "${MSI_CHANNEL}")
fi

if [[ -n "${DATASET_DIR}" ]]; then
  ARGS+=(--dataset-dir "${DATASET_DIR}")
else
  if [[ -z "${RGB_DIR}" || -z "${MSI_DIR}" ]]; then
    echo "[ERROR] Please set DATASET_DIR or both RGB_DIR and MSI_DIR." >&2
    exit 1
  fi
  ARGS+=(--rgb-dir "${RGB_DIR}" --msi-dir "${MSI_DIR}")
fi

python infer/infer_rtmsfdetr_dir.py "${ARGS[@]}"
