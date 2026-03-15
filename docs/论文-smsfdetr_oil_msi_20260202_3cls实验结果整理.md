# SMSFDETR Oil MSI 20260202 3cls 实验结果整理

本文整理 `outputs/oil_msi_20260202_3cls/smsfdetr` 目录下截至 `2026-03-14` 的主要实验结果，重点分析以下问题：

1. 当前最有效的方法是哪一类；
2. 这些方法的收益来自哪里；
3. 哪些结论可以用于论文撰写，哪些结论还不能下得太满；
4. 下一步应该继续优化什么。

---

## 1. 实验范围

本次结果整理分为两部分：

1. **当前主目录中的最新实验**
   - `smsfdetr_rtv4_hgnetv2_m_oil_msi20260202_baseline_nomodule`
   - `smsfdetr_rtv4_hgnetv2_m_oil_msi20260202_origstem_residual_msbranch_shared_hgstem_inner_fixed_band_cmda_b3_stem_cf_interactive`
   - `smsfdetr_rtv4_hgnetv2_m_oil_msi20260202_origstem_residual_msbranch_shared_hgstem_inner_fixed_band_cmda_b3_stem_cf_interactive_infonce_lw004`
   - `smsfdetr_rtv4_hgnetv2_m_oil_msi20260202_origstem_residual_msbranch_shared_hgstem_inner_fixed_band_cmda_b3_stem_cf_interactive_infonce_lw006`
   - `smsfdetr_rtv4_hgnetv2_m_oil_msi20260202_origstem_residual_msbranch_shared_hgstem_inner_fixed_band_cmda_b3_stem_cf_interactive_noloss`

2. **`old/` 目录中的历史尝试**
   - CRGGA / FixedBandCMDA / shallow relation / residual ms branch 等旧方案
   - 用于横向判断当前新方案在整个探索过程中的位置

说明：

- 本文优先使用 `Auto test metric summary` 中的测试集结果作为最终比较指标。
- 对于没有完整自动测试结果的实验，不纳入主结论。
- 对于存在多次重复的实验，同时给出“最佳单次结果”和“重复实验均值”。

---

## 2. 当前主结论

先给出直接结论：

1. **当前最值得继续写论文和优化的方案，是 `origstem_residual_msbranch_shared_hgstem_inner_fixed_band_cmda_b3_stem_cf_interactive` 的原始 InfoNCE 版本。**
2. **它拿到了目前全部尝试中的最好单次测试结果**，但重复实验波动较大，暂时还不能写成“稳定优于 baseline”。
3. **适度的 InfoNCE 对齐监督是有帮助的**；完全去掉损失或者继续增大损失权重，效果都会下降。
4. **收益主要来自 `machine` 类别**，而不是所有类别同时显著提升。

---

## 3. 最新实验主对比

### 3.1 测试集主结果

| 方法 | 重复次数 | Test mAP50:95 均值 | Test mAP50:95 最好值 | Test mAP50 最好值 | 结论 |
|---|---:|---:|---:|---:|---|
| `baseline_nomodule` | 5 | `0.6311 ± 0.0077` | `0.6438` | `0.8139` | 最稳定的参考基线 |
| `stem_cf_interactive + InfoNCE(0.02)` | 2 | `0.6254 ± 0.0254` | `0.6508` | `0.8069` | 最佳单次结果，但波动较大 |
| `stem_cf_interactive + no-loss` | 1 | `0.6225` | `0.6225` | `0.7992` | 结构本身有一定作用，但不如最佳 InfoNCE |
| `stem_cf_interactive + InfoNCE(0.04)` | 1 | `0.6160` | `0.6160` | `0.7900` | 对齐损失偏强，开始压制主任务 |
| `stem_cf_interactive + InfoNCE(0.06)` | 1 | `0.5879` | `0.5879` | `0.7711` | 明显过约束，性能下降 |

### 3.2 结果解读

从上述表格可以看出：

1. `baseline_nomodule` 的平均性能最高，说明它仍然是**当前最可靠的稳定基线**。
2. `stem_cf_interactive + InfoNCE(0.02)` 的**最好单次结果**达到 `0.6508`，高于 baseline 的最好单次 `0.6438`，说明这条路线**有真实潜力**。
3. 但同一方法第二次重复实验只有 `0.6000` 左右，因此它的均值反而低于 baseline，说明**当前主要问题是稳定性不足，而不是完全无效**。
4. `no-loss` 版本低于最佳 InfoNCE，也低于 baseline 最好值，说明**仅靠结构前向本身不够，适度的显式对齐监督是有价值的**。
5. `InfoNCE` 权重从 `0.02` 增加到 `0.04`、`0.06` 后持续变差，说明当前模块不是“监督越强越好”，而是存在明显的最佳工作点。

---

## 4. 关键运行实例

### 4.1 Baseline 最佳单次

- 运行目录：
  - `outputs/oil_msi_20260202_3cls/smsfdetr/smsfdetr_rtv4_hgnetv2_m_oil_msi20260202_baseline_nomodule/260312-000008-smsfdetr_oil_msi_20260202_det_rtv4_hgnetv2_m_baseline_nomodule`
- 测试结果：
  - `test_map50_95 = 0.6437717163`
  - `test_map50 = 0.8094342309`
  - `test_precision = 0.8756068539`
  - `test_recall = 0.69`
  - `test_f1 = 0.7718013340`

### 4.2 当前最佳方法单次结果

- 方法：
  - `origstem_residual_msbranch_shared_hgstem_inner_fixed_band_cmda_b3_stem_cf_interactive`
- 运行目录：
  - `outputs/oil_msi_20260202_3cls/smsfdetr/smsfdetr_rtv4_hgnetv2_m_oil_msi20260202_origstem_residual_msbranch_shared_hgstem_inner_fixed_band_cmda_b3_stem_cf_interactive/260314-164912-smsfdetr_oil_msi_20260202_det_rtv4_hgnetv2_m_origstem_residual_msbranch_shared_hgstem_inner_fixed_band_cmda_b3_stem_cf_interactive`
- 测试结果：
  - `test_map50_95 = 0.6508072619`
  - `test_map50 = 0.7986297989`
  - `test_precision = 0.8886298886`
  - `test_recall = 0.72`
  - `test_f1 = 0.7954763545`

### 4.3 同方法第二次重复结果

- 运行目录：
  - `outputs/oil_msi_20260202_3cls/smsfdetr/smsfdetr_rtv4_hgnetv2_m_oil_msi20260202_origstem_residual_msbranch_shared_hgstem_inner_fixed_band_cmda_b3_stem_cf_interactive/260314-205640-smsfdetr_oil_msi_20260202_det_rtv4_hgnetv2_m_origstem_residual_msbranch_shared_hgstem_inner_fixed_band_cmda_b3_stem_cf_interactive`
- 测试结果：
  - `test_map50_95 = 0.5999552907`
  - `test_map50 = 0.8069461452`
  - `test_precision = 0.8911467946`
  - `test_recall = 0.68`
  - `test_f1 = 0.7713853631`

这个重复结果直接说明：**该方法存在较明显的训练波动问题。**

---

## 5. 类别级分析

为了判断收益来自哪里，选取：

- baseline 最佳单次
- `stem_cf_interactive + InfoNCE(0.02)` 最佳单次

做类别对比。

| 类别 | Baseline mAP50:95 | Stem-CF+InfoNCE mAP50:95 | 变化 |
|---|---:|---:|---:|
| `oil` | `0.5598` | `0.5613` | `+0.0015` |
| `building` | `0.8062` | `0.7995` | `-0.0067` |
| `machine` | `0.5652` | `0.5917` | `+0.0265` |
| `all` | `0.6438` | `0.6508` | `+0.0070` |

### 5.1 类别分析结论

1. `oil` 类几乎持平。
2. `building` 类略有下降。
3. `machine` 类提升最明显，是整体提升的主要来源。

这说明当前模块的收益不是“均匀地改善所有类别”，而更像是：

> **它更擅长帮助那些对浅层错位或局部几何扰动更敏感的类别。**

这一点在论文中反而是可信的，因为真实模块改进通常不会让所有类别无差别同步提升。

---

## 6. 对损失函数与损失权重的判断

### 6.1 `no-loss` 的含义

`no-loss` 版本保留了：

- 原始 stem 主分支
- 浅层 residual MS 分支
- inner fixed-band CMDA
- stem-CF interactive

但关闭了全部辅助监督：

- `loss_weight = 0`
- `loss_offset_weight = 0`
- `loss_attn_entropy_weight = 0`

测试结果为：

- `test_map50_95 = 0.6224704281`

这说明：

1. 结构本身并非完全无效；
2. 但如果没有显式监督，残差分支提供的几何补充不足以稳定超过 baseline；
3. 因此**“结构 + 适度监督”优于“结构单独前向”**。

### 6.2 为什么 `InfoNCE(0.02)` 最合理

与 `no-loss` 相比：

- `InfoNCE(0.02)` 最佳单次 `0.6508`
- `no-loss` 为 `0.6225`

这说明适度的对齐判别损失是有帮助的。

与更强损失相比：

- `InfoNCE(0.04)` 为 `0.6160`
- `InfoNCE(0.06)` 为 `0.5879`

说明在当前结构下，继续强化对齐损失会：

1. 使残差分支对主干的约束过强；
2. 抢占主任务优化空间；
3. 导致检测性能下降。

因此可以得出一个相对可靠的判断：

> **当前方法需要的是“轻量辅助对齐监督”，而不是“强监督驱动的硬对齐”。**

---

## 7. `old/` 历史实验中的位置

如果把 `old/` 目录中的历史实验一起看，当前已经出现了更清晰的结果排序。

### 7.1 历史实验中测试集较好的结果

| 排名 | 方法 | Test mAP50:95 |
|---|---|---:|
| 1 | `origstem_residual_msbranch_shared_hgstem_inner_fixed_band_cmda_b3_stem_cf_interactive` | `0.6508` |
| 2 | `baseline_nomodule` 最佳单次 | `0.6438` |
| 3 | `origstem_crgga_fixed_band_input_loss` | `0.6426` |
| 4 | `origstem_residual_msbranch_shared_hgstem_inner_fixed_band_cmda_b3` | `0.6380` |
| 5 | `origstem_shallow_group_relation_c2` | `0.6318` |
| 6 | `origstem_cmda_b3_plus_group_relation_c2_weak` | `0.6268` |
| 7 | `origstem_residual_msbranch_shared_hgstem_postalign_concatproj` | `0.6256` |
| 8 | `origstem_residual_msbranch_shared_hgstem_noalign` | `0.6230` |

### 7.2 这说明什么

1. 当前主方法已经是**全部尝试中的最好单次结果**。
2. 但它的领先幅度并不大：
   - 相比 baseline 最佳只高 `+0.0070`
   - 相比旧的 `origstem_crgga_fixed_band_input_loss` 只高 `+0.0082`
3. 因此现阶段更适合写成：
   - “当前提出的方法取得了最好结果”
   - “相较于现有若干浅层对齐与残差建模方案具有更优上限”

而不适合写成：

- “大幅领先”
- “稳定显著优于所有基线”

---

## 8. 哪些结论可以写进论文

### 8.1 可以较有把握写的结论

1. **直接重写或大幅替换 stem 并不稳定。**
2. **保留原始 stem 主路径，同时引入显式 band-wise 浅层残差分支，是更合理的设计方向。**
3. **在残差分支内部做固定锚点对齐，比早期一些 CRGGA/直接输入级对齐方案更有潜力。**
4. **适度的 InfoNCE 对齐监督优于无监督或过强监督。**
5. **收益主要集中在 `machine` 类别，说明该模块对错位敏感类别更有效。**

### 8.2 目前还不能写得太满的结论

1. 不能直接写“该方法稳定优于 baseline”。
2. 不能直接写“该方法对所有类别都有效”。
3. 不能直接写“对齐监督越强越好”。
4. 不能直接写“CRGGA 完全无意义”。

更准确的表述应当是：

> 当前实验表明，在保留原始 stem 稳定性的前提下，显式 band-wise 残差分支配合轻量对齐监督，能够在最佳设定下取得当前最优精度；但其性能增益仍主要体现为更高的结果上限，而非稳定的均值优势。

---

## 9. 当前最可信的论文叙述方式

建议把论文主线写成以下逻辑：

1. **问题动机**  
   直接改 stem 破坏稳定性；完全不显式建模 band 错位又难以利用多光谱互补性。

2. **核心设计**  
   保留原始 stem 作为主分支；额外增加浅层显式 band-wise residual MS 分支；在该分支内部以固定波段为锚点进行轻量对齐；再通过 stem-level cross interaction 与保守残差方式注入主干。

3. **消融结论**  
   - 只保留结构但不加监督，效果有限；
   - 适度 InfoNCE 最优；
   - 更强的对齐损失会破坏检测性能；
   - 说明需要的是“弱而有效”的对齐先验，而不是强配准约束。

4. **实验发现**  
   当前收益主要来自 `machine` 类别，说明该设计更适合解决对局部几何错位更敏感的样本。

---

## 10. 下一步优化建议

当前最值得继续投入的方向不是“继续增大损失权重”，而是**提升方法稳定性**。

### 10.1 优先级最高

1. 对 `InfoNCE(0.02)` 再补 `2~3` 次重复实验。
2. 确认其重复实验均值能否真正超过 baseline 的均值 `0.6311`。
3. 若均值也能超过 baseline，则该方法就同时具备：
   - 最佳单次优势
   - 平均性能优势

### 10.2 不建议的方向

1. 不建议继续把 `InfoNCE` 权重往上加。
2. 不建议再直接引入 `LNCC` 这类强结构相关损失到当前浅层 learned feature 上。
3. 不建议为了显式 band 分离而大幅重写原始 stem。

### 10.3 可以继续尝试的方向

1. 固定 `InfoNCE(0.02)`，优先优化训练稳定性。
2. 让残差注入更保守，例如更小的残差缩放初始化。
3. 让 stem-CF 交互更轻量，减少其对主分支统计的扰动。
4. 如果要加额外监督，优先考虑更弱的结构约束，而不是直接增大主对齐损失。

---

## 11. 补充说明

以下实验当前不纳入最终定量结论：

1. `stem_cf_interactive_cosine`
   - 当前没有完整可用的自动测试结果。

2. `stem_cf_interactive_infonce_lncc`
   - 已确认不是运行报错，而是优化过程从早期就陷入停滞；
   - 因此不能作为“有效实验结果”纳入主对比，只能作为失败消融说明。

---

## 12. 最终总结

截至当前，`smsfdetr` 在 `oil_msi_20260202_3cls` 上的实验结果可以概括为一句话：

> **保留原始 stem，增加浅层显式 residual MS 分支，并在分支内部做固定锚点对齐与 stem-level cross interaction，是目前最有潜力的方向；适度 InfoNCE 监督可以把该方向推到当前最好结果，但其稳定性仍需进一步验证。**

因此，现阶段最合理的策略不是继续大规模改结构，而是：

1. 锁定当前最佳结构；
2. 重点补重复实验；
3. 证明其均值性能与类别收益是否真正成立；
4. 在此基础上再决定是否继续做轻量优化。
