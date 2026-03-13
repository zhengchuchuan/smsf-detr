#!/usr/bin/env bash
set -euo pipefail


python main.py --mode train --opts runtime.device_ids=[3] --config 'configs/task/rtmsfdetr/oil_rgb_msi_20260202_3cls/rtmsfdetr_oil_rgb_msi_20260202_det_rtv4_hgnetv2_m_distill_dualstream_c2former_postblock_add_wbadd_c3c4c5_msbandsep_c2fixedbandcmda_b3_infonce_reg_globalkv_pos2d.yaml'