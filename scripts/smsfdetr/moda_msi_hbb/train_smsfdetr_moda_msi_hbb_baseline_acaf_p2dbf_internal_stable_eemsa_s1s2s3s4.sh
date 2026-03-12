#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-1}"
BATCH_SIZE="${BATCH_SIZE:-8}"

python main.py --mode train \
  --config configs/task/smsfdetr/moda_msi_hbb/smsfdetr_moda_msi_hbb_det_rtv4_hgnetv2_m_baseline_acaf_p2dbf_internal_stable_eemsa_s1s2s3s4.yaml \
  --opts "runtime.device_ids=[$GPU_ID]" "train.batch_size=$BATCH_SIZE"
