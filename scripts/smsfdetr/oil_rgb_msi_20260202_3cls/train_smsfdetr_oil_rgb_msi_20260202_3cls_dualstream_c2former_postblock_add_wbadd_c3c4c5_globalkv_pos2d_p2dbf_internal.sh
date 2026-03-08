#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"

python main.py --mode train \
  --config configs/task/smsfdetr/oil_rgb_msi_20260202_3cls/smsfdetr_oil_rgb_msi_20260202_det_rtv4_hgnetv2_m_distill_dualstream_c2former_postblock_add_wbadd_c3c4c5_globalkv_pos2d_p2dbf_internal.yaml \
  --opts "runtime.device_ids=[$GPU_ID]" "train.batch_size=$BATCH_SIZE"
