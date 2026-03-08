#!/usr/bin/env bash
set -euo pipefail

RGB_PATH="${RGB_PATH:-tmp/crgga_vis/MAX_20240612_MAX_0201_Color_D_slice_0_0.jpg}"
MS_PATH="${MS_PATH:-tmp/crgga_vis/MAX_20240612_MAX_0201_Color_D_slice_0_0.tif}"
CHECKPOINT="${CHECKPOINT:-outputs/oil_rgb_msi_20260202_3cls/rtmsfdetr/rtv4_hgnetv2_m_distill_det_dualstream_c2former_postblock_add_wbadd_c3c4c5_msbandsep_c2align_infonce_reg_globalkv_pos2d/baseline5/260203-230950-rtmsfdetr_oil_rgb_msi_20260202_det_rtv4_hgnetv2_m_distill_dualstream_c2former_postblock_add_wbadd_c3c4c5_msbandsep_c2align_infonce_reg_globalkv_pos2d/checkpoint_best.pth}"
OUTDIR="${OUTDIR:-}"
DEVICE="${DEVICE:-auto}"
MS_FIXED_SCALE="${MS_FIXED_SCALE:-65535}"

usage() {
  cat <<'USAGE'
Usage:
  RGB_PATH=tmp/crgga_vis/MAX_20240612_MAX_0201_Color_D_slice_0_0.jpg \
  MS_PATH=tmp/crgga_vis/MAX_20240612_MAX_0201_Color_D_slice_0_0.tif \
  CHECKPOINT=outputs/oil_rgb_msi_20260202_3cls/rtmsfdetr/rtv4_hgnetv2_m_distill_det_dualstream_c2former_postblock_add_wbadd_c3c4c5_msbandsep_c2align_infonce_reg_globalkv_pos2d/baseline5/260203-230950-rtmsfdetr_oil_rgb_msi_20260202_det_rtv4_hgnetv2_m_distill_dualstream_c2former_postblock_add_wbadd_c3c4c5_msbandsep_c2align_infonce_reg_globalkv_pos2d/checkpoint_best.pth \
  DEVICE=auto \
  MS_FIXED_SCALE=65535 \
  OUTDIR=outputs/vis_crgga/MAX_20240612_MAX_0201_Color_D_slice_0_0 \
  bash scripts/vis/vis_crgga_on_pair.sh

Notes:
  - RGB_PATH is optional.
  - OUTDIR is optional; if omitted, a timestamped folder under outputs/vis_crgga/ is used.
USAGE
}

if [[ -z "${MS_PATH}" || -z "${CHECKPOINT}" ]]; then
  usage
  exit 1
fi

if [[ -z "${OUTDIR}" ]]; then
  tag=$(basename "${MS_PATH}")
  tag="${tag%.*}"
  ts=$(date +%Y%m%d-%H%M)
  OUTDIR="outputs/vis_crgga/${tag}_${ts}"
fi

python vis/vis_crgga_on_pair.py \
  ${RGB_PATH:+--rgb "${RGB_PATH}"} \
  --ms "${MS_PATH}" \
  --checkpoint "${CHECKPOINT}" \
  --device "${DEVICE}" \
  --ms_fixed_scale "${MS_FIXED_SCALE}" \
  --outdir "${OUTDIR}"

echo "[OK] Output dir: ${OUTDIR}"
echo "[FIG] Paper-ready: ${OUTDIR}/paper_rgb_overlays.png"
echo "[FIG] Alt:         ${OUTDIR}/paper_flow_quiver_mean.png"
