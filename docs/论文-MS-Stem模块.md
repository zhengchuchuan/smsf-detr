# 论文写法：MS-Stem 模块（Band-Separated Stem + 可选 CRGGA 对齐）

本节给出项目中 **MS-Stem**（多光谱输入端的 stem）模块的论文级描述，覆盖其用途、端到端公式流程与实现对应关系。

> 代码对应：
> - MS-Stem 主体：`engines/models/rtmsfdetr/rtdetrv4/engine/backbone/ms_band_sep.py` 中的 `MSBandSeparatedStemAlign`
> - 可选对齐器：`engines/models/rtmsfdetr/rtdetrv4/engine/backbone/group_deform_align.py` 中的 `CRGGA`（内部调用 `DeformableAlign2D`/KADW）
> - 插入位置：`engines/models/rtmsfdetr/rtdetrv4/engine/backbone/hgnetv2_dualstream.py` 中替换 `ms_backbone.stem`（配置项：`model.backbone_ms_band_sep`）

---

## 1. 模块用途与动机

在 RGB+MS（多光谱）双流检测中，多光谱 MS 模态通常由 \(N\) 个波段组成（例如 \(N=7\)）。由于成像光路差异、传感器响应差异与轻微时序差异等因素，**MS 的各波段在几何上往往“不严格对齐”**。若直接将 \(\mathbf{I}^{ms}\in\mathbb{R}^{B\times N\times H\times W}\) 视为 \(N\) 通道输入并用普通卷积做早期融合，卷积会在空间位置上对齐地混合各波段，导致“错位叠加”（ghosting）并污染浅层特征。

因此，MS-Stem 的设计目标是：

1) **避免对齐前的跨波段早期融合**：先对每个波段独立编码，再做（可选）对齐，最后再融合。  
2) **在浅层（stride=4 / C2 级别）完成 MS 内部对齐**：用较低分辨率减少计算/显存开销，同时仍保留足够的边缘与几何信息。  
3) **输出与主干网络 stage-1 输入兼容的特征**：将对齐后的多波段特征映射为 backbone 期望的 \(C_{c2}\) 通道特征图，作为后续 HGNetv2 MS 分支的输入。

---

## 2. 符号定义与输入输出

设 batch size 为 \(B\)，多光谱波段数为 \(N\)，输入空间分辨率为 \(H\times W\)。

- 多光谱输入：
  \[
  \mathbf{I}^{ms}\in\mathbb{R}^{B\times N\times H\times W}.
  \]
- MS-Stem 输出（对齐/融合后的 C2 级别特征）：
  \[
  \mathbf{F}^{ms}_{c2}\in\mathbb{R}^{B\times C_{c2}\times \tfrac{H}{4}\times \tfrac{W}{4}}.
  \]

其中 \(\tfrac{H}{4},\tfrac{W}{4}\) 来自 stem 内部两次 stride=2 下采样（实现为两层 \(3\times 3\) 卷积，stride=2）。

---

## 3. 结构与公式流程（从输入到输出）

MS-Stem 可以概括为三个顺序子模块：

1) **逐波段共享嵌入（Per-band Shared Embedding）**  
2) **可选：CRGGA 对齐（Canonical Reference Guided Groupwise Alignment）**  
3) **跨波段融合与通道压缩（Flatten + 1×1 Merge to C2）**

下面按论文写法给出完整公式流程。

### 3.1 逐波段共享嵌入（不跨 band 混合）

将 band 维显式保留，并对每个波段使用同一套轻量 CNN（参数共享）：

首先将 \(\mathbf{I}^{ms}\) reshape，把 band 维并入 batch：
\[
\overline{\mathbf{I}}=\mathrm{reshape}(\mathbf{I}^{ms})\in\mathbb{R}^{(BN)\times 1\times H\times W}.
\]

用共享的嵌入网络 \(f_{\text{emb}}\) 提取每个 band 的浅层特征（stride=4）：
\[
\overline{\mathbf{X}}=f_{\text{emb}}(\overline{\mathbf{I}})\in\mathbb{R}^{(BN)\times C_e\times \tfrac{H}{4}\times \tfrac{W}{4}},
\]
其中 \(C_e\) 为每个波段的 embedding 通道数（实现中为 `embed_channels`）。

再 reshape 回显式 band 维：
\[
\mathbf{X}=\mathrm{reshape}^{-1}(\overline{\mathbf{X}})\in\mathbb{R}^{B\times N\times C_e\times \tfrac{H}{4}\times \tfrac{W}{4}}.
\]

实现细节对齐（写入论文时可作为注释说明）：
- \(f_{\text{emb}}\) 由两层 \(3\times 3\) 卷积构成，每层 stride=2，并使用 GN + ReLU（代码中字段名 `embed_use_bn` 仍保留，但实际使用 GN 以避免 reshape 后 BN 的统计混合）。

### 3.2 可选：CRGGA 对齐（MS 内部波段对齐）

对齐模块的输入为 \(\mathbf{X}\in\mathbb{R}^{B\times N\times C_e\times H_4\times W_4}\)（其中 \(H_4=\tfrac{H}{4}, W_4=\tfrac{W}{4}\)）。

#### 3.2.1 构建 canonical reference（空间加权参考）

为避免选定固定参考波段，CRGGA 从所有波段构建 canonical reference feature：

对每个位置 \(p\)（位于 stride=4 网格上），先用 \(1\times 1\) 卷积对每个波段打分：
\[
s_n(p)=g(\mathbf{X}_n)(p),\quad g:\mathbb{R}^{C_e}\rightarrow\mathbb{R}.
\]

在 band 维做 softmax 得到空间位置相关的权重：
\[
w_n(p)=\frac{\exp(s_n(p))}{\sum_{j=1}^{N}\exp(s_j(p))}.
\]

canonical reference 定义为加权和：
\[
\mathbf{R}(p)=\sum_{n=1}^{N}w_n(p)\mathbf{X}_n(p),\qquad
\mathbf{R}\in\mathbb{R}^{B\times C_e\times H_4\times W_4}.
\]

实现中可选 `ref_detach`，即用于预测对齐参数时采用 \(\mathbf{R}_{\text{pred}}=\mathrm{stopgrad}(\mathbf{R})\)，以提升训练稳定性。

#### 3.2.2 Keypoint-Attentive Deformable Warping（KADW）对齐每个 band

对每个波段特征 \(\mathbf{X}_n\)，利用 pairwise 对齐器估计其到 reference 的局部可变形采样参数。

令关键点数为 \(K\)（实现中对应 `num_keypoints`），对齐器预测每个像素位置 \(p\) 的 \(K\) 个 2D 偏移与对应权重：
\[
\{\Delta_n^k(p)\}_{k=1}^{K},\quad \Delta_n^k(p)\in\mathbb{R}^2,
\qquad
\{\alpha_n^k(p)\}_{k=1}^{K},\quad \sum_k \alpha_n^k(p)=1.
\]

采用双线性采样（`grid_sample`）从源特征图取值：
\[
\mathbf{S}_n^k(p)=\mathbf{X}_n(p+\Delta_n^k(p)).
\]

并用注意力权重进行融合，得到对齐后的 band 特征：
\[
\widetilde{\mathbf{X}}_n(p)=\sum_{k=1}^{K}\alpha_n^k(p)\mathbf{S}_n^k(p).
\]

将所有波段堆叠回显式 band 维，得到
\[
\widetilde{\mathbf{X}}\in\mathbb{R}^{B\times N\times C_e\times H_4\times W_4}.
\]

训练时可选引入对齐辅助损失（实现支持 cosine 或 patch InfoNCE），用于约束 \(\widetilde{\mathbf{X}}_n\) 与 \(\mathbf{R}\) 的一致性，从而在无显式几何标注下学习出合理的对齐。

### 3.3 跨波段融合与通道压缩（merge to C2）

在完成（可选）对齐后，才进行跨波段融合。先将 band 维展平并在通道维拼接：
\[
\mathbf{Z}=\mathrm{Concat}\big(\widetilde{\mathbf{X}}_1,\ldots,\widetilde{\mathbf{X}}_N\big)\in\mathbb{R}^{B\times (NC_e)\times H_4\times W_4}.
\]

再通过 \(1\times 1\) 卷积（配合 GN + ReLU）压缩到 backbone stage-1 期望的输入通道数 \(C_{c2}\)：
\[
\mathbf{F}^{ms}_{c2}=\phi(\mathbf{Z}),\qquad \phi:\mathbb{R}^{NC_e}\rightarrow\mathbb{R}^{C_{c2}}.
\]

至此，\(\mathbf{F}^{ms}_{c2}\) 作为 MS 分支进入后续 HGNetv2 stage blocks，与 RGB 分支在更高层（例如 C3/C4/C5）进行跨模态融合。

---

## 4. 讨论：为何该 MS-Stem 适合多光谱检测

1) **先对齐、后融合**：避免将几何错位直接写入卷积的跨通道混合中，有利于保留清晰边缘与结构信息。  
2) **stride=4 对齐的性价比**：以较低分辨率进行密集对齐，显著降低开销，同时对几何误差仍具敏感性。  
3) **GN 的稳定性**：由于实现中将 \((B,N)\) reshape 为 \((BN)\)，GN 更适合小 batch 且避免 BN 的跨 band 统计混合。  

---

## 5. 实现侧注意事项（写论文时可不展开）

- 当启用 MS-Stem 替换 HGNetv2 的 `ms_backbone.stem` 时，原 stem 的参数会被冻结，避免 DDP 下 unused-parameter 问题（见 `hgnetv2_dualstream.py`）。
- 训练时若 CRGGA 返回辅助损失字典，训练器会将其聚合进总损失（见 `hgnetv2_dualstream.py` 的 `stem_losses` 处理分支）。

