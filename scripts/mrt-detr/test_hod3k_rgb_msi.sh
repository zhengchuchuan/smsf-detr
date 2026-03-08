#!/usr/bin/env bash
  set -euo pipefail

  # 必须在 msifp-detr 仓库根目录运行（这里应当能看到 main.py）
  if [[ ! -f "main.py" ]]; then
    echo "ERROR: please run this script from the msifp-detr repo root (where main.py exists)." >&2
    exit 1
  fi

  REPO_ROOT="$(pwd)"
  MRT_DIR="${REPO_ROOT}/third_party/MRT-DETR"
  MRT_CFG_REL="rtdetr_pytorch/configs/rtdetr/rtdetr_dual_msifp_hod3k_rgb_msi3.yml"
  OUTDIR="${REPO_ROOT}/third_party/MRT-DETR/output/msifp_hod3k_rgb_msi3"
  CKPT="${OUTDIR}/checkpoint_best.pth"

  # NOTE: 某些受限环境（容器/沙盒）下多进程 DataLoader 会 PermissionError；
  # 如遇到类似错误，可保持 NUM_WORKERS=0（默认）进行测试。
  export NUM_WORKERS="${NUM_WORKERS:-0}"

  if [[ ! -f "${MRT_DIR}/${MRT_CFG_REL}" ]]; then
    echo "ERROR: MRT config not found: ${MRT_DIR}/${MRT_CFG_REL}" >&2
    exit 1
  fi
  if [[ ! -f "${CKPT}" ]]; then
    echo "ERROR: checkpoint not found: ${CKPT}" >&2
    exit 1
  fi

  echo "[1/2] Evaluate on split=test (NUM_WORKERS=${NUM_WORKERS})..."
  REPO_ROOT="${REPO_ROOT}" MRT_CFG_REL="${MRT_CFG_REL}" CKPT="${CKPT}" NUM_WORKERS="${NUM_WORKERS}" \
    python - <<'PY'
import os
import sys
from pathlib import Path

repo_root = Path(os.environ["REPO_ROOT"])
mrt_dir = repo_root / "third_party" / "MRT-DETR"
cfg_rel = os.environ["MRT_CFG_REL"]
ckpt = os.environ["CKPT"]
num_workers = int(os.environ.get("NUM_WORKERS", "0"))

os.chdir(mrt_dir)
sys.path.insert(0, str(mrt_dir / "rtdetr_pytorch"))

from src.core import YAMLConfig
from src.solver import TASKS

cfg = YAMLConfig(
    str(mrt_dir / cfg_rel),
    resume=ckpt,
    use_amp=False,
    val_dataloader={"num_workers": num_workers},
    test_dataloader={"num_workers": num_workers},
)
solver = TASKS[cfg.yaml_cfg["task"]](cfg)
solver.val(split="test")
PY

  echo "[2/2] Plot metrics (if log.txt exists)..."
  if [[ -f "${OUTDIR}/log.txt" ]]; then
    python third_party/MRT-DETR/rtdetr_pytorch/tools/plot_metrics.py --output-dir "${OUTDIR}"
  fi

  echo "[OK] Test metrics -> ${OUTDIR}/metrics.json"
  echo "[OK] Test chart   -> ${OUTDIR}/metric_charts/test/test_overall_metrics.png"

