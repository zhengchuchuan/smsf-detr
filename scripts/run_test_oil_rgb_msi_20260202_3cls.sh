#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="outputs/oil_rgb_msi_20260202_3cls/rtmsfdetr/rtv4_hgnetv2_m_distill_det_dualstream_c2former_postblock_add_wbadd_c3c4c5_msbandsep_c2align_infonce_reg_globalkv_pos2d/260202-170850-rtmsfdetr_oil_rgb_msi_20260202_det_rtv4_hgnetv2_m_distill_dualstream_c2former_postblock_add_wbadd_c3c4c5_msbandsep_c2align_infonce_reg_globalkv_pos2d"

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
# 可选：覆盖数据集根目录（需符合 COCO 结构：<root>/rgb/<split>, <root>/msi/<split>, <root>/annotations/<split>.json）
DATASET_DIR="${DATASET_DIR:-data/oil_20260202}"
# 可选：指定推理 split（默认 test）
SPLIT="${SPLIT:-test}"
# 可选：保存全部样例可视化（默认给一个很大的数）
SAMPLES_NUM="${SAMPLES_NUM:-999999}"
SCORE_THR="${SCORE_THR:-0.5}"
MAX_DETS="${MAX_DETS:-100}"

EXTRA_OPTS=(
  "train.metric_charts.samples.enabled=true"
  "train.metric_charts.samples.num_samples=${SAMPLES_NUM}"
  "train.metric_charts.samples.score_threshold=${SCORE_THR}"
  "train.metric_charts.samples.max_dets=${MAX_DETS}"
  "data.test_split=${SPLIT}"
)
if [[ -n "${DATASET_DIR}" ]]; then
  EXTRA_OPTS+=("data.dataset_dir=${DATASET_DIR}")
fi

python infer/run_test.py \
  --config "${RUN_DIR}/config.yaml" \
  --checkpoint "${CKPT}" \
  --output-root "${RUN_DIR}" \
  --device "${DEVICE}" \
  --opts "${EXTRA_OPTS[@]}"
