#!/usr/bin/env bash
  set -euo pipefail

  # 可选：指定 GPU
  # export CUDA_VISIBLE_DEVICES=0
  # 可选：指定 tuning 权重（存在才会启用 -t）
  # export TUNING_CKPT="third_party/MRT-DETR/rtdetr_pytorch/rtdetr_r50vd_6x_coco_from_paddle.pth"

# CUDA_VISIBLE_DEVICES=3
# TUNING_CKPT=third_party/MRT-DETR/rtdetr_pytorch/rtdetr_r50vd_6x_coco_from_paddle.pth
# bash scripts/mrt-detr/train_oil_20260202_3cls.sh



  # 必须在 msifp-detr 仓库根目录运行（这里应当能看到 main.py）
  if [[ ! -f "main.py" ]]; then
    echo "ERROR: please run this script from the msifp-detr repo root (where main.py exists)." >&2
    exit 1
  fi

  REPO_ROOT="$(pwd)"
  # 使用 msifp-detr 的 data config 来生成 3cls COCO 标注（annotations_3cls/*.json）
  DATA_CFG="${REPO_ROOT}/configs/data/oil_rgb_msi_20260202_3cls.yaml"
  # MRT-DETR 输出目录（与 rtdetr config 里的 output_dir 保持一致）
  OUTDIR="${REPO_ROOT}/third_party/MRT-DETR/output/msifp_oil_20260202_3cls_msi7"
  MRT_DIR="${REPO_ROOT}/third_party/MRT-DETR"
  MRT_CFG_REL="rtdetr_pytorch/configs/rtdetr/rtdetr_dual_msifp_oil_20260202_3cls_msi7.yml"

  if [[ ! -f "${DATA_CFG}" ]]; then
    echo "ERROR: DATA_CFG not found: ${DATA_CFG}" >&2
    exit 1
  fi
  if [[ ! -f "${MRT_DIR}/${MRT_CFG_REL}" ]]; then
    echo "ERROR: MRT config not found: ${MRT_DIR}/${MRT_CFG_REL}" >&2
    exit 1
  fi

  echo "[1/4] Build 3cls annotations..."
  python "${REPO_ROOT}/utils/data_process/filter_coco_categories.py" \
    --config "${DATA_CFG}" \
    --drop-empty-images \
    --overwrite

  echo "[2/4] Backup old output dir (restart from scratch)..."
  if [[ -d "${OUTDIR}" ]]; then
    ts="$(date +%Y%m%d_%H%M%S)"
    mv "${OUTDIR}" "${OUTDIR}.bak_${ts}"
    echo "  moved -> ${OUTDIR}.bak_${ts}"
  fi

  echo "[3/4] Start training..."
  mkdir -p "${MRT_DIR}/output"
  LOG_FILE="${MRT_DIR}/output/train_msifp_oil_20260202_3cls_msi7_$(date +%Y%m%d_%H%M%S).log"

  cd "${MRT_DIR}"
  cmd=(python rtdetr_pytorch/tools/train.py -c "${MRT_CFG_REL}")
  if [[ -n "${TUNING_CKPT:-}" && -f "${TUNING_CKPT}" ]]; then
    cmd+=(-t "${TUNING_CKPT}")
  fi
  "${cmd[@]}" 2>&1 | tee "${LOG_FILE}"

  echo "[4/4] Plot metrics..."
  cd "${REPO_ROOT}"
  python third_party/MRT-DETR/rtdetr_pytorch/tools/plot_metrics.py \
    --output-dir third_party/MRT-DETR/output/msifp_oil_20260202_3cls_msi7

  echo "[OK] Test metrics (on split=test) -> ${OUTDIR}/metrics.json"
  echo "[OK] Test overall chart          -> ${OUTDIR}/metric_charts/test/test_overall_metrics.png"
