#!/usr/bin/env bash
set -euo pipefail

# Align the sample under tmp/register using MatchAnything-ROMA (two-stage pipeline).
#
# Usage:
#   sh tools/run_tmp_register_matchanything_roma.sh [device] [imgresize] [save_match_vis] [save_pseudo_vis] [vis_layout] [no_config_translation]
#
# Examples:
#   sh tools/run_tmp_register_matchanything_roma.sh cuda:0 1440 1 1 lr 0
#   sh tools/run_tmp_register_matchanything_roma.sh cpu 832 0 1 tb 1
#
# Args:
#   device: MatchAnything 推理设备（如 cuda:0 / cpu）。
#   imgresize: MatchAnything 匹配时的最长边缩放尺寸（<=0 表示不缩放）。
#   save_match_vis/save_pseudo_vis: 1 开启可视化，0 关闭。
#   vis_layout: lr（左右）或 tb（上下）。
#   no_config_translation: 1 禁用配置文件粗平移校准，0 使用配置文件校准。
#
# Notes:
# - This repo does NOT vendor MatchAnything by default. Set MATCHANYTHING_REPO to the
#   MatchAnything repo root (must contain configs/, src/, weights/).
# - Outputs will be written under tmp/register/ aligned_*_{run_tag}_...

DEVICE="${1:-cuda:0}"              # MatchAnything 推理设备
IMGRESIZE="${2:-1440}"             # 匹配时最长边缩放尺寸
SAVE_MATCH_VIS="${3:-1}"           # 是否保存匹配线可视化
SAVE_PSEUDO_VIS="${4:-1}"          # 是否保存伪彩图前后对比
VIS_LAYOUT="${5:-lr}"              # 可视化布局 lr/tb
NO_CONFIG_TRANSLATION="${6:-1}"    # 是否禁用配置文件粗校准
MATCHANYTHING_REPO="/home/ubuntu/Documents/newdisk_22T/zcc/msifp-detr/third_party/MatchAnything/imcui/third_party/MatchAnything"
DATA_DIR="tmp/register"
CONFIG_DIR="data/config/oil/my-conf"

# Point this to your local MatchAnything checkout (repo root).
: "${MATCHANYTHING_REPO:=}"
if [[ -z "${MATCHANYTHING_REPO}" ]]; then
  echo "ERROR: MATCHANYTHING_REPO is not set." >&2
  echo "Set it to your MatchAnything repo root (contains configs/, src/, weights/)." >&2
  exit 2
fi
if [[ ! -d "${MATCHANYTHING_REPO}" ]]; then
  echo "ERROR: MATCHANYTHING_REPO not found: ${MATCHANYTHING_REPO}" >&2
  exit 2
fi

EXTRA_ARGS=()
if [[ "${SAVE_MATCH_VIS}" == "1" ]]; then
  EXTRA_ARGS+=(--save-match-vis --match-vis-layout "${VIS_LAYOUT}" --match-vis-dir "${DATA_DIR}/match_vis")
fi
if [[ "${SAVE_PSEUDO_VIS}" == "1" ]]; then
  EXTRA_ARGS+=(--save-pseudo-vis --pseudo-vis-layout "${VIS_LAYOUT}" --pseudo-vis-dir "${DATA_DIR}/pseudo_vis")
fi
if [[ "${NO_CONFIG_TRANSLATION}" == "1" ]]; then
  EXTRA_ARGS+=(--no-config-translation)
fi

ARGS=(
  # RGB 图像目录
  --rgb-dir "${DATA_DIR}"
  # 光谱 HDR 目录
  --spectral-dir "${DATA_DIR}"
  # 配置文件目录（粗平移校准）
  --config-dir "${CONFIG_DIR}"
  # 参考通道（以 jpg 为基准）
  --alignment-reference jpg
  # 二次配准拓扑（链式: chain）
  --secondary-topology single
  # 可见光使用对应通道作为参考
  --chain-visible-rgb-ref channels
  # 阶段1方法：MatchAnything
  --stage1-method matchanything
  # MatchAnything 仓库路径
  --matchanything-repo "${MATCHANYTHING_REPO}"
  # MatchAnything 模型类型
  --matchanything-model matchanything_roma
  # RANSAC 估计器（相似变换）
  --matchanything-estimator similarity
  # MatchAnything 设备
  --matchanything-device "${DEVICE}"
  # MatchAnything 缩放尺寸
  --matchanything-imgresize "${IMGRESIZE}"
  # 全量输出目录（RGB+光谱）
  --output-full "${DATA_DIR}/aligned_full_tif"
  # 光谱输出目录（仅光谱）
  --output-nir "${DATA_DIR}/aligned_nir_tif"
  # 覆盖已有输出
  --overwrite
)

python data_align/data_align_matchanything_two_stage.py \
  "${ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
