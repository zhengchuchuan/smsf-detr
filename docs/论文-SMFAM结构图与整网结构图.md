# SMFAM 结构图与整网结构图

本文档给出当前方法对应的两张可编辑 Mermaid 结构图：

1. `SMFAM` 模块细粒度结构图
2. 当前 `MSI-only SMSF-DETR` 整体网络结构图

对应配置：

- `configs/task/smsfdetr/oil_msi_20260202_3cls/smsfdetr_oil_msi_20260202_det_rtv4_hgnetv2_m_origstem_residual_msbranch_shared_hgstem_inner_fixed_band_cmda_b3_stem_cf_interactive.yaml`

对应主干与检测框架：

- Backbone：`HGNetv2-M (B2)` + `SMFAM`
- Encoder：`HybridEncoder`
- Decoder：`DFINETransformer`

---

## 1. SMFAM 模块细粒度结构图

建议论文图题：

- 图 X SMFAM 模块结构图

```mermaid
flowchart TB
    X["MSI Input<br/>X ∈ R^{B×7×H×W}"]

    subgraph Main["OrigStem Main Path"]
        M1["Original HGNetv2 StemBlock"]
        M2["Main stem feature<br/>F_main ∈ R^{B×32×H/4×W/4}"]
        M1 --> M2
    end

    subgraph Residual["Residual Multispectral Branch"]
        R0["Band reshape<br/>(B,7,H,W) → (B·7,1,H,W)"]
        R1["Shared HGStem encoder"]
        R2["Explicit band features<br/>Z ∈ R^{B×7×32×H/4×W/4}"]

        subgraph Align["FixedBandCMDA"]
            A0["Anchor band<br/>b4 (anchor)"]
            A1["Support bands<br/>b1,b2,b3,b5,b6,b7"]
            A2["Offset / attention prediction<br/>DeformableAlign2D"]
            A3["Deformable resampling"]
            A4["Anchor-aware fusion"]
            A5["Corrected band features<br/>Z~ ∈ R^{B×7×32×H/4×W/4}"]

            A0 --> A2
            A1 --> A2
            A2 --> A3 --> A4 --> A5
            A0 --> A4
        end

        R0 --> R1 --> R2 --> Align
        Align --> R3["Flatten bands + 1×1 merge<br/>(GN + identity act)"]
        R3 --> R4["Residual feature<br/>F_res ∈ R^{B×32×H/4×W/4}"]
    end

    subgraph Fusion["Shallow Feature Alignment"]
        C1["Query projection on F_main"]
        C2["Memory projection on F_res"]
        C3["Single-scale MSDeformAttn<br/>reference lattice + learnable shift"]
        C4["delta_fuse + output scale γ_cf"]
        C5["Aligned main feature<br/>F_cf = F_main + γ_cf ⊙ Δ"]

        C1 --> C3
        C2 --> C3
        C3 --> C4 --> C5
    end

    X --> M1
    X --> R0
    M2 --> C1
    R4 --> C2

    C5 --> O["Residual injection<br/>F_out = F_cf + γ_res ⊙ F_res"]
    R4 --> O
    O --> Y["To HGNetv2 Stage1-Stage4"]
```

### 1.1 图示说明

- 主路径保留原始 `HGNetv2 StemBlock`，提供稳定的浅层主特征 `F_main`。
- 残差支路保留显式 band 维，在 `FixedBandCMDA` 中完成“固定锚点波段引导的波段校正”。
- `FixedBandCMDA` 的核心是“对齐 + 锚点条件融合”，不是单纯的 deformable warping。
- `StemCF` 负责跨支路浅层交互修正：主路径作为 query，残差支路作为 memory。
- 最终通过小权重残差注入得到 `F_out`，再送入后续 `HGNetv2` stage。

---

## 2. 当前整网结构图

建议论文图题：

- 图 Y 基于 SMFAM 的 MSI-only SMSF-DETR 整体网络结构图

```mermaid
flowchart TB
    I0["MSI input<br/>X ∈ R^{B×7×H×W}"]
    I1["Per-channel min-max normalization"]

    subgraph Backbone["Backbone: HGNetv2-M(B2) + SMFAM"]
        B0["OrigStem + SMFAM align<br/>output: R^{B×32×H/4×W/4}"]
        B1["Stage1 / C2<br/>96 channels, stride 4"]
        B2["Stage2 / C3<br/>384 channels, stride 8"]
        B3["Stage3 / C4<br/>768 channels, stride 16"]
        B4["Stage4 / C5<br/>1536 channels, stride 32"]

        B0 --> B1 --> B2 --> B3 --> B4
    end

    C3O["Backbone output 1<br/>C3 ∈ R^{B×384×H/8×W/8}"]
    C4O["Backbone output 2<br/>C4 ∈ R^{B×768×H/16×W/16}"]
    C5O["Backbone output 3<br/>C5 ∈ R^{B×1536×H/32×W/32}"]

    subgraph Encoder["HybridEncoder"]
        E1["1×1 channel projection<br/>[384,768,1536] → [256,256,256]"]
        E2["Top-level Transformer encoder<br/>(use_encoder_idx = [2])"]
        E3["FPN top-down + PAN bottom-up fusion"]
        E4["Multi-scale encoded features<br/>[P3,P4,P5], each 256 channels"]

        E1 --> E2 --> E3 --> E4
    end

    subgraph Decoder["DFINETransformer"]
        D0["300 object queries<br/>+ denoising queries"]
        D1["6 decoder layers<br/>multi-scale deformable cross-attention"]
        D2["Classification head"]
        D3["Box regression head"]

        D0 --> D1
        D1 --> D2
        D1 --> D3
    end

    subgraph Output["Detection Output"]
        O1["pred_logits + pred_boxes"]
        O2["PostProcessor"]
        O3["Final detections"]

        O1 --> O2 --> O3
    end

    subgraph TrainOnly["Train-only supervision"]
        T1["RTv4Criterion<br/>loss_vfl + loss_bbox + loss_giou<br/>+ loss_fgl + loss_ddf"]
        T2["SMFAM auxiliary losses<br/>InfoNCE align + offset reg + attn entropy"]
    end

    I0 --> I1 --> B0
    B2 --> C3O
    B3 --> C4O
    B4 --> C5O

    C3O --> E1
    C4O --> E1
    C5O --> E1

    E4 --> D1
    D2 --> O1
    D3 --> O1

    B0 -. train only .-> T2
    O1 -. train only .-> T1
    T2 -. merged into criterion .-> T1
```

### 2.1 图示说明

- 当前方法不是双流 RGB+MS，而是单流 `MSI-only` 配置。
- `SMFAM` 只作用在 backbone 最浅层，即 `Stem/C2` 附近。
- `HGNetv2-M(B2)` 主干在当前配置下输出三个尺度给检测头：
  - `C3`: `384` channels, stride `8`
  - `C4`: `768` channels, stride `16`
  - `C5`: `1536` channels, stride `32`
- `HybridEncoder` 将三层主干特征统一投影到 `256` 维，并通过 top-level encoder + FPN/PAN 生成 `[P3,P4,P5]`。
- `DFINETransformer` 基于多尺度特征和查询向量输出分类与边界框预测。
- 训练时，`SMFAM` 内部的辅助对齐损失会作为额外项并入 `RTv4Criterion`。

---

## 3. 使用建议

- 如果你后续要把它们放进论文，建议先在 Mermaid Live Editor 中微调节点位置，再导出为 `svg`。
- 这两张图当前是“结构准确优先”的版本，适合继续人工美化成论文终稿。
- 如果你需要，我下一步可以继续帮你做两个版本：
  - 论文简洁版：节点更少，更适合正文插图
  - 答辩详细版：保留通道数、stride 和内部子模块
