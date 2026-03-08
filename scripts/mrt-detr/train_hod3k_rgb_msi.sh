#!/usr/bin/env bash
  set -euo pipefail

  # 可选：指定 GPU
  # export CUDA_VISIBLE_DEVICES=0
  # 可选：指定 tuning 权重（存在才会启用 -t）
  # export TUNING_CKPT="third_party/MRT-DETR/rtdetr_pytorch/rtdetr_r50vd_6x_coco_from_paddle.pth"

  # 必须在 msifp-detr 仓库根目录运行（这里应当能看到 main.py）
  if [[ ! -f "main.py" ]]; then
    echo "ERROR: please run this script from the msifp-detr repo root (where main.py exists)." >&2
    exit 1
  fi

  REPO_ROOT="$(pwd)"
  DATA_CFG="${REPO_ROOT}/configs/data/HOD3K_rgb_msi.yaml"
  # 训练前生成 annotations_3cls/{train,val,test}.json（确保只包含 data.class_names）
  DST_ANN_DIR="${REPO_ROOT}/data/HOD3K/annotations_3cls"

  OUTDIR="${REPO_ROOT}/third_party/MRT-DETR/output/msifp_hod3k_rgb_msi3"
  MRT_DIR="${REPO_ROOT}/third_party/MRT-DETR"
  MRT_CFG_REL="rtdetr_pytorch/configs/rtdetr/rtdetr_dual_msifp_hod3k_rgb_msi3.yml"

  if [[ ! -f "${DATA_CFG}" ]]; then
    echo "ERROR: DATA_CFG not found: ${DATA_CFG}" >&2
    exit 1
  fi
  if [[ ! -f "${MRT_DIR}/${MRT_CFG_REL}" ]]; then
    echo "ERROR: MRT config not found: ${MRT_DIR}/${MRT_CFG_REL}" >&2
    exit 1
  fi

  echo "[1/4] Build filtered annotations (annotations_3cls)..."
  python "${REPO_ROOT}/utils/data_process/filter_coco_categories.py" \
    --config "${DATA_CFG}" \
    --dst-ann-dir "${DST_ANN_DIR}" \
    --drop-empty-images \
    --overwrite

  echo "[2/4] Backup old output dir (restart from scratch)..."
  if [[ -d "${OUTDIR}" ]]; then
    ts="$(date +%Y%m%d_%H%M%S)"
    mv "${OUTDIR}" "${OUTDIR}.bak_${ts}"
    echo "  moved -> ${OUTDIR}.bak_${ts}"
  fi

  echo "[3/4] Start training (epoches=80)..."
  mkdir -p "${MRT_DIR}/output"
  LOG_FILE="${MRT_DIR}/output/train_msifp_hod3k_rgb_msi3_$(date +%Y%m%d_%H%M%S).log"

  cd "${MRT_DIR}"
  cmd=(python rtdetr_pytorch/tools/train.py -c "${MRT_CFG_REL}")
  if [[ -n "${TUNING_CKPT:-}" && -f "${TUNING_CKPT}" ]]; then
    cmd+=(-t "${TUNING_CKPT}")
  fi
  "${cmd[@]}" 2>&1 | tee "${LOG_FILE}"

  echo "[4/4] Plot metrics..."
  cd "${REPO_ROOT}"
  python third_party/MRT-DETR/rtdetr_pytorch/tools/plot_metrics.py \
    --output-dir third_party/MRT-DETR/output/msifp_hod3k_rgb_msi3

  echo "[OK] Test metrics (on split=test) -> ${OUTDIR}/metrics.json"
  echo "[OK] Test overall chart          -> ${OUTDIR}/metric_charts/test/test_overall_metrics.png"

