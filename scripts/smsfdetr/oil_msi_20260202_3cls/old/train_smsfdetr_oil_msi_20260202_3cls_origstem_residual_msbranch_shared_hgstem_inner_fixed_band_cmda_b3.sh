#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-1}"
BATCH_SIZE="${BATCH_SIZE:-8}"

python main.py --mode train \
  --config configs/task/smsfdetr/oil_msi_20260202_3cls/smsfdetr_oil_msi_20260202_det_rtv4_hgnetv2_m_origstem_residual_msbranch_shared_hgstem_inner_fixed_band_cmda_b3.yaml \
  --opts "runtime.device_ids=[$GPU_ID]" "train.batch_size=$BATCH_SIZE" "$@"
