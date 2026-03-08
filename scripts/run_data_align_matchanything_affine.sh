#!/usr/bin/env bash
set -euo pipefail

# Run data alignment with MatchAnything stage1 using a full affine (6-DoF) RANSAC fit.
#
# Usage:
#   sh tools/run_data_align_matchanything_affine.sh <dataset_root> [device]
#
# Examples:
#   sh tools/run_data_align_matchanything_affine.sh /path/to/raw_dataset cuda:0
#   sh tools/run_data_align_matchanything_affine.sh /path/to/raw_dataset cpu

DATASET_ROOT="${1:-}"
DEVICE="${2:-cuda:0}"

if [[ -z "${DATASET_ROOT}" ]]; then
  echo "Usage: $0 <dataset_root> [device]" >&2
  exit 2
fi

python data_align/data_align_matchanything_two_stage.py \
  --rgb-dir "${DATASET_ROOT%/}/images" \
  --spectral-dir "${DATASET_ROOT%/}/spectral" \
  --alignment-reference jpg \
  --stage1-method matchanything \
  --stage2-method none \
  --matchanything-model matchanything_eloftr \
  --matchanything-estimator affine \
  --matchanything-device "${DEVICE}" \
  --matchanything-imgresize 832 \
  --overwrite
