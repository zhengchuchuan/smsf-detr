# MODA-hbb 集成计划与执行清单

## Phase 0: 数据读取一致性（必须先通过）

目标：让 `smsf-detr` 的读取行为与 `msifp-detr` 在 MODA-hbb 上一致，避免“同配置不同输入张量”。

执行项：
1. `datasets/multispectral_coco.py` 支持 `.npy/.npz`。
2. 增加 `data.ms_npy_layout`（`auto/chw/cwh/hwc`），并在加载时生效。
3. MODA-hbb 数据配置固定：
   - `ms_suffix=.npy`
   - `ms_npy_layout=cwh`
   - `ms_normalize_mode=fixed_scale`
   - `ms_fixed_scale=255`
   - `val_split=test`

冒烟命令：
```bash
python - <<'PY'
from datasets.multispectral_coco import _load_msi_as_tensor
from pathlib import Path
p = Path("data/MODA-hbb-coco/msi/train/20000101000049140_00-00059.npy")
t = _load_msi_as_tensor(p, expected_channels=8, npy_layout="cwh")
print(t.shape, t.dtype, float(t.min()), float(t.max()))
PY
```

## Phase 1: baseline（不加模块）

配置：
- `configs/task/smsfdetr/moda_msi_hbb/smsfdetr_moda_msi_hbb_det_rtv4_hgnetv2_m_baseline_nomodule.yaml`

脚本：
- `scripts/smsfdetr/moda_msi_hbb/train_smsfdetr_moda_msi_hbb_baseline_nomodule.sh`

建议先跑 1 epoch 验证：
```bash
bash scripts/smsfdetr/moda_msi_hbb/train_smsfdetr_moda_msi_hbb_baseline_nomodule.sh \
  && true
```

## Phase 2: 单模块 EEMSA

配置：
- `configs/task/smsfdetr/moda_msi_hbb/smsfdetr_moda_msi_hbb_det_rtv4_hgnetv2_m_baseline_eemsa_s1s2s3s4.yaml`

脚本：
- `scripts/smsfdetr/moda_msi_hbb/train_smsfdetr_moda_msi_hbb_baseline_eemsa_s1s2s3s4.sh`

## Phase 3: ACAF + P2DBF 稳定版

配置：
- `configs/task/smsfdetr/moda_msi_hbb/smsfdetr_moda_msi_hbb_det_rtv4_hgnetv2_m_baseline_acaf_p2dbf_internal_stable.yaml`

脚本：
- `scripts/smsfdetr/moda_msi_hbb/train_smsfdetr_moda_msi_hbb_baseline_acaf_p2dbf_internal_stable.sh`

## Phase 4: 全组合（EEMSA + ACAF + P2DBF）

配置：
- `configs/task/smsfdetr/moda_msi_hbb/smsfdetr_moda_msi_hbb_det_rtv4_hgnetv2_m_baseline_acaf_p2dbf_internal_stable_eemsa_s1s2s3s4.yaml`

脚本：
- `scripts/smsfdetr/moda_msi_hbb/train_smsfdetr_moda_msi_hbb_baseline_acaf_p2dbf_internal_stable_eemsa_s1s2s3s4.sh`

## 验证要求（每阶段）

1. 训练侧：
   - 首先 `epochs=1` 冒烟无报错；
   - 再跑完整训练并记录 best 指标。
2. 评估侧：
   - 使用同一 `config.yaml` + `checkpoint_best.pth`；
   - 记录 `map50/map50_95`、每类 AP。
3. 对照侧：
   - 与上一阶段仅改一处变量；
   - 固定 `seed` 与 batch size（显存允许时）。
