#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-3}"
BATCH_SIZE="${BATCH_SIZE:-6}"

python main.py --mode train \
  --config configs/task/smsfdetr/oil_rgb_msi_20260202_3cls/smsfdetr_oil_rgb_msi_20260202_det_rtv4_hgnetv2_m_distill_dualstream_asmf_postblock_add_wbadd_c3c4c5_globalkv_pos2d_acaf_p2dbf_internal_eemsa_s1s2s3s4.yaml \
  --opts "runtime.device_ids=[$GPU_ID]" "train.batch_size=$BATCH_SIZE"
