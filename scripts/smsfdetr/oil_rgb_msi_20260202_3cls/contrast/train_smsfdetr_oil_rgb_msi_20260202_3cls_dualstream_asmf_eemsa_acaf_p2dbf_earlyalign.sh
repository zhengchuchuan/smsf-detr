#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-2}"
BATCH_SIZE="${BATCH_SIZE:-10}"

python main.py --mode train \
  --config configs/task/smsfdetr/oil_rgb_msi_20260202_3cls/smsfdetr_oil_rgb_msi_20260202_det_rtv4_hgnetv2_m_distill_dualstream_asmf_eemsa_acaf_p2dbf_earlyalign.yaml \
  --opts "runtime.device_ids=[$GPU_ID]" "train.batch_size=$BATCH_SIZE" "$@"
