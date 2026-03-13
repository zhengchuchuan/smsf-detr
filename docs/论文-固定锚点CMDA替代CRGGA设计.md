# 设计文档：固定锚点 CMDA 替代 CRGGA

本文面向当前项目里的 7 通道多光谱输入场景，分析 `third_party/CF-Deformable-DETR` 中 CF-CMDA 思路能否替代现有 `CRGGA`，并给出一个更适合本仓库的落地方案：**固定波段锚点（fixed-band anchor）版 Band-CMDA**。

---

## 0. 结论先行

先给出直接结论，避免讨论发散：

1. **不能根据当前“预测 offset 看起来不好”就直接下结论说 CRGGA 没有意义。**  
   更准确的结论是：**当前这套 CRGGA 任务定义、参考构造方式、监督方式和插入位置，在你的数据与骨干设置下没有明显带来收益，甚至可能引入负优化。**

2. **本地 `CF-Deformable-DETR` 不能直接拿来复用。**  
   仓库里只有外层 `deformable_transformer.py` 结构，缺失 `models.ops.modules` 以及 `MSDeformAttn_cross_interactive` 等关键实现，无法直接迁移。

3. **CF-CMDA 和当前 CRGGA 不是同一类模块。**  
   `CRGGA` 是在 `(B, N, C, H, W)` 上做**显式 band-wise 密集配准**；  
   CF-CMDA 本质是 decoder/query 级别的**跨分支 deformable cross-attention**。

4. **你的新方向是可行的，但不应理解为“把 CF-CMDA 原样塞进 MS stem”。**  
   更合适的做法是：  
   - 保留 `MSBandSeparatedStemAlign` 的“逐 band 提特征”；
   - 不再构造 canonical reference；
   - 选一个固定波段特征作为锚点；
   - 其余 6 个波段向该锚点坐标系对齐；
   - 再在锚点坐标系内融合。

5. **最合理的工程路线不是“复刻 CF 论文代码”，而是新写一个适配当前项目的 `FixedBandCMDA`。**  
   这会是一个**单尺度、anchor-query、band-to-anchor** 的轻量模块，放在 `MSBandSeparatedStemAlign.embed(...)` 之后、`merge` 之前。

---

## 1. 当前现象应该怎么解读

你前面的实验现象大致是：

- baseline `nomodule` 更稳；
- 打开当前 CRGGA/输入对齐后，预测框有偏移；
- 从可视化直觉看，offset 学到的东西不够可信。

这个现象更像是在说明下面几件事，而不是简单说明“CRGGA 没意义”：

### 1.1 现有任务比想象中更难

当前 CRGGA 学的是：

- 从 7 个 band 的浅层特征中构造一个 canonical reference；
- 每个 band 再对这个 reference 做可变形采样对齐；
- 用检测损失 + 辅助对齐损失，间接逼出正确 warp。

这件事同时包含了三重困难：

1. **reference 本身是学习出来的，不稳定。**
2. **offset 没有 GT 几何监督，只能靠间接损失驱动。**
3. **7 个 band 全部朝一个动态 reference 对齐，比“6 个 band 朝 1 个固定 band 对齐”更难。**

所以“偏移效果不好”很可能不是因为“对齐不需要”，而是因为**当前优化问题定义过于松、过于难**。

### 1.2 当前辅助损失不一定在教几何对齐

你现在用的辅助损失主要是 `cosine` / `InfoNCE` 一类特征一致性约束。  
这类损失有两个典型问题：

- 它鼓励的是“特征相似”，不一定是“几何位置正确”；
- 在浅层纹理上，模型可能通过改变纹理响应而不是学正确位移来降低损失。

因此模型有可能学到的是：

- 更平滑的特征；
- 更局部的注意力权重；
- 某种“看起来像对齐”的响应，

但不一定真的得到对检测最有利的几何配准。

### 1.3 canonical reference 可能反而模糊了配准目标

`CRGGA` 的强项是“不需要显式指定参考 band”，但它也带来副作用：

- 对 7 个 band 做 mean / weighted / spatial weighted 汇聚后，reference 往往是“综合体”；
- 综合体不一定对应任何真实成像波段；
- 如果不同 band 的局部响应差异很大，reference 可能变成“语义折中图”，而不是几何清晰图。

这样每个 band 对齐的目标就不够明确：

- 它不是对齐到 band 3；
- 也不是对齐到 band 5；
- 而是对齐到一个动态混合出来的 reference。

这对优化是额外负担。

### 1.4 对检测任务来说，轻微错位不一定值得大规模显式校正

检测 backbone 和 neck 自身已经具有一定平移鲁棒性。  
如果真实 band 间错位不大，而对齐模块引入了额外自由度，那么模型很可能出现：

- 收益小；
- 不稳定性大；
- 辅助损失影响主任务优化。

所以当前结果更适合得出的判断是：

> **当前版本的 CRGGA 没有形成“高置信度、低副作用”的有效对齐。**

而不是：

> **MS 场景根本不需要对齐。**

---

## 2. 当前项目里的 CRGGA 到底在做什么

当前实现位于：

- `engines/models/rtmsfdetr/rtdetrv4/engine/backbone/ms_band_sep.py`
- `engines/models/rtmsfdetr/rtdetrv4/engine/backbone/group_deform_align.py`

整体流程是：

1. `MSBandSeparatedStemAlign` 先做逐 band 共享 embedding：
   \[
   \mathbf{I}^{ms}\in\mathbb{R}^{B\times 7\times H\times W}
   \rightarrow
   \mathbf{Z}\in\mathbb{R}^{B\times 7\times C_e\times H_4\times W_4}
   \]

2. `CRGGA` 在显式 band 维上构造 reference：
   \[
   \mathbf{R}\in\mathbb{R}^{B\times C_e\times H_4\times W_4}
   \]

3. 对每个 band 特征 \(\mathbf{Z}_i\) 做 pairwise deformable alignment：
   \[
   \widetilde{\mathbf{Z}}_i = \mathrm{Align}(\mathbf{R}, \mathbf{Z}_i)
   \]

4. 再把 7 个对齐后的 band 特征 flatten + `1x1 merge` 到 backbone 的 C2 输入通道。

这说明当前 `CRGGA` 的核心不是“融合”，而是：

- **先定义一个公共目标坐标系；**
- **再把每个 band warp 到这个坐标系。**

这与 CF-CMDA 的工作点不同。

---

## 3. 本地 `CF-Deformable-DETR` 里的 CMDA 实际是什么

### 3.1 它不是一个独立的“特征图对齐器”

在本地 `third_party/CF-Deformable-DETR` 中，并没有一个单独命名为 `CMDA` 的模块文件。  
所谓 CMDA，实际上是 `deformable_transformer.py` 里 decoder 的跨分支 deformable attention 路径。

从结构上看，它更接近：

- query 来自当前解码 token；
- reference points 来自 query 的归一化采样中心；
- memory/value 来自另一分支的多尺度特征；
- 通过 deformable attention 从另一分支采样并更新 query 表示。

这本质上是：

> **query-level 的 cross-modal interaction**

而不是：

> **dense feature-map warping**

### 3.2 它依赖 Deformable DETR 风格的输入组织

CF-CMDA 需要的是一整套 flatten 后的多尺度表示：

- `src`
- `spatial_shapes`
- `level_start_index`
- `reference_points`
- `query_pos`

也就是说，它不是直接吃 `(B, C, H, W)` 或 `(B, N, C, H, W)` 的卷积特征图就能工作的。

### 3.3 本地仓库还缺关键实现

当前 `third_party/CF-Deformable-DETR` 存在一个直接问题：

- `deformable_transformer.py` 会 import
  - `MSDeformAttn`
  - `MSDeformAttn_cross`
  - `MSDeformAttn_cross_interactive`
  - `MSDeformAttn_cross_base`
- 但本地 `models/ops` 下缺少对应源码模块。

因此结论非常明确：

> **你不能把本地这个 third_party 目录里的 CF-CMDA 当成一个可直接 copy 的模块。**

你现在能借鉴的是**思路**，不是完整实现。

---

## 4. 为什么“固定波段锚点”比 canonical reference 更适合你当前阶段

你的新诉求是：

- 不想再构建 canonical reference；
- 想直接选一个固定 band 特征作为锚点；
- 其余 band 向它配准并融合。

这个方向比当前 CRGGA 更合理，主要因为它把问题明显简化了。

### 4.1 对齐目标更明确

canonical reference 的问题在于目标是动态变化的。  
固定波段锚点方案则很明确：

- 锚点 band \(a\) 就是坐标系；
- 所有 support band \(i\neq a\) 都对齐到 \(\mathbf{Z}_a\)。

优化目标从：

> “对齐到一个学习出来的混合 reference”

变成：

> “对齐到 band-a 这张真实特征图”

这通常更稳。

### 4.2 任务从“多对一动态 reference”变成“多对一固定 anchor”

当前 CRGGA 每轮都要：

- 先构 reference；
- 再依次对齐 7 个 band。

固定锚点方案则不需要 reference 构造分支，只需：

- 选 anchor；
- 其余 band 直接和 anchor 建模关系。

这减少了一个不确定源。

### 4.3 更符合“配准 + 融合一体化”的目标

如果只用 `fixed_band` 的 CRGGA 模式，你仍然是在做：

- 独立对齐每个 band；
- 最后再统一 merge。

而你现在希望借 CF-CMDA 的点，在于：

- 对齐不是单纯 warp；
- 对齐后的信息应该直接服务于融合。

固定锚点 CMDA 可以天然做成：

- anchor 发 query；
- support 提供可变形采样信息；
- 输出直接是“在 anchor 坐标系下的 support-aware 表示”；
- 最终把 anchor 和多支 support-aware 表示一起融合。

---

## 5. 需要先澄清的一个概念：reference feature 和 reference points 不是一回事

这里非常容易混淆。

### 5.1 你可以去掉的是 reference feature

也就是：

- 不再通过 mean / weighted / spatial-weighted 构造一张 canonical feature map；
- 而是直接拿固定 band 特征当锚点。

### 5.2 你不能去掉的是 deformable sampling 的 reference points

无论是 CRGGA 还是 CF-CMDA，只要是 deformable sampling，本质都需要一个采样中心。

对固定锚点方案而言，这些 reference points 应来自：

- 锚点特征图的规则网格；
- 或锚点 query 自身预测出的参考点。

所以“不要构建参考”应理解为：

> **不要构建 canonical reference feature**

而不是：

> **不要 deformable attention 的 sampling reference points**

---

## 6. 推荐的新方案：Fixed-Band Anchor CMDA

下面给出我认为最适合当前仓库的方案。

### 6.1 模块放置位置

推荐插在现有 `MSBandSeparatedStemAlign` 流程中：

1. per-band embedding  
2. **Fixed-Band Anchor CMDA**  
3. flatten + merge

也就是：

\[
\mathbf{I}^{ms}
\rightarrow
\mathbf{Z}\in\mathbb{R}^{B\times 7\times C_e\times H_4\times W_4}
\rightarrow
\widehat{\mathbf{Z}}\in\mathbb{R}^{B\times 7\times C_e\times H_4\times W_4}
\rightarrow
\mathbf{F}^{ms}_{c2}
\]

推荐原因：

- 此时 band 维仍然显式存在；
- 特征已有基础语义，比 raw band 更稳定；
- 分辨率是 stride=4，成本可控；
- 不需要改动主干大部分结构。

### 6.2 基本张量定义

设 embedding 后特征为：

\[
\mathbf{Z}=\{\mathbf{Z}_0,\mathbf{Z}_1,\dots,\mathbf{Z}_6\},\quad
\mathbf{Z}_i\in\mathbb{R}^{B\times C_e\times H_4\times W_4}
\]

选定一个固定锚点 band 索引 \(a\)，例如：

- 默认先试中间波段：`a = 3`
- 或按先验选择结构最清晰/信噪比最好的一支

记锚点特征为：

\[
\mathbf{A}=\mathbf{Z}_a
\]

其余 support band 为：

\[
\mathbf{S}_i=\mathbf{Z}_i,\quad i\neq a
\]

### 6.3 模块目标

目标不是简单得到：

\[
\mathrm{Warp}(\mathbf{S}_i \rightarrow \mathbf{A})
\]

而是得到：

\[
\mathbf{Y}_i = \mathrm{CMDA}(\mathbf{A}, \mathbf{S}_i)
\]

其中 \(\mathbf{Y}_i\) 具有两个性质：

1. 空间坐标系与 anchor \(\mathbf{A}\) 一致；
2. 内容来自 support band \(\mathbf{S}_i\) 对 anchor 的补充。

这样最终融合不是“几何纠偏后的平铺拼接”，而是“anchor 坐标系下的多 band 互补表示”。

---

## 7. 一个适合当前工程的 v1 结构

不建议第一版就追求完整多尺度 Deformable DETR 风格实现。  
最稳妥的是先做一个**单尺度、密集网格、anchor-query** 版本。

### 7.1 Anchor-to-Support 单向 cross deformable interaction

对每个 support band \(i\neq a\)，做下面的过程：

#### 第一步：生成 anchor query

从 anchor 特征生成 query：

\[
\mathbf{Q}_a = \phi_q(\mathbf{A})
\]

其中 \(\phi_q\) 可以是 `1x1 conv` 或轻量卷积块。

#### 第二步：用 anchor 与 support 的联合信息预测偏移和权重

将 anchor 与 support 拼接：

\[
\mathbf{U}_i = [\mathbf{A}; \mathbf{S}_i]
\]

经轻量 head 预测：

- \(K\) 个采样偏移：
  \[
  \Delta_i \in \mathbb{R}^{B\times K\times 2\times H_4\times W_4}
  \]
- \(K\) 个采样权重：
  \[
  \alpha_i \in \mathbb{R}^{B\times K\times H_4\times W_4}
  \]

这里的参考网格就是 anchor 网格。

#### 第三步：从 support 特征图采样

对 anchor 网格上每个位置 \(p\)，在 support 图上取 \(K\) 个采样点：

\[
\mathbf{v}_i^k(p) = \mathbf{S}_i(p + \Delta_i^k(p))
\]

并做加权聚合：

\[
\widetilde{\mathbf{S}}_i(p) = \sum_{k=1}^{K}\alpha_i^k(p)\mathbf{v}_i^k(p)
\]

#### 第四步：做一个 cross gating / residual update

为了让输出不只是“warp 后的 support”，建议再用 anchor 做一次门控融合：

\[
\mathbf{Y}_i = \phi_o\left([\mathbf{A}; \widetilde{\mathbf{S}}_i]\right)
\]

或残差式：

\[
\mathbf{Y}_i = \mathbf{A} + \psi\left([\mathbf{A}; \widetilde{\mathbf{S}}_i]\right)
\]

这样每个 \(\mathbf{Y}_i\) 都是在 anchor 坐标系下、带有第 \(i\) 个 support band 信息的增强特征。

### 7.2 最终融合

把 anchor 本身与所有 `support-aware` 特征一起融合：

\[
\mathbf{F}_{fusion} =
\mathrm{Fuse}\left(
\mathbf{A},
\mathbf{Y}_0,\dots,\mathbf{Y}_{a-1},\mathbf{Y}_{a+1},\dots,\mathbf{Y}_6
\right)
\]

可选的 `Fuse` 方式：

1. 直接 concat + `1x1 conv`
2. concat + channel attention + `1x1 conv`
3. weighted sum + residual

从工程稳健性看，第一版建议：

- `concat`
- `1x1 conv`
- `GN`
- `ReLU`

最简单可靠。

### 7.3 输出形式

有两种输出组织方式。

#### 方案 A：只输出融合后的单张特征图

\[
\mathbf{F}^{ms}_{c2} = \mathrm{Merge}(\mathbf{F}_{fusion})
\]

优点：

- 参数少；
- 简洁；
- 更接近“CMDA 直接做配准融合”。

缺点：

- 失去“显式 7 band 中间表示”；
- 可解释性稍弱。

#### 方案 B：保留 7 张对齐后的 band 表示，再统一 merge

定义：

- anchor 输出保持 \(\mathbf{A}\)
- 每个 support 输出为 \(\mathbf{Y}_i\)

得到：

\[
\widehat{\mathbf{Z}}=
\{\mathbf{Y}_0,\dots,\mathbf{A},\dots,\mathbf{Y}_6\}
\in\mathbb{R}^{B\times 7\times C_e\times H_4\times W_4}
\]

再走现有 `flatten + merge`。

我更推荐 **方案 B** 作为第一版，因为：

- 与当前 `MSBandSeparatedStemAlign` 接口更兼容；
- 更容易和现有 CRGGA 做 apples-to-apples 对比；
- 后续若效果好，再压缩为更融合的一体化版本。

---

## 8. 它和“把 CRGGA 的 ref_mode 改成 fixed_band”有什么区别

这个区别非常关键。

当前代码里，`CRGGA` 实际已经支持：

- `ref_mode="fixed_band"`
- `ref_band_index=<某一支 band>`

但这不等于你真正想要的 fixed-band CMDA。

### 8.1 `CRGGA(fixed_band)` 仍然是 pairwise warp 模型

它做的是：

\[
\widetilde{\mathbf{Z}}_i = \mathrm{Align}(\mathbf{Z}_a, \mathbf{Z}_i)
\]

然后统一堆叠、merge。

它的重点仍然是：

- 几何对齐；
- 每个 support band 单独 warp。

### 8.2 Fixed-Band CMDA 是“anchor query 驱动的对齐融合”

它更像：

\[
\mathbf{Y}_i = \mathrm{CrossDeformAttn}(\mathbf{A}\leftarrow\mathbf{S}_i)
\]

重点变成：

- anchor 发起查询；
- support 响应采样；
- 输出是融合后的 anchor-space 表示。

换句话说：

- `CRGGA(fixed_band)` 更偏 **registration-first**
- `Fixed-Band CMDA` 更偏 **registration-aware fusion**

### 8.3 建议先做一个很重要的中间对照实验

在写新模块之前，建议先跑一个低成本对照：

- 保持现有 `CRGGA` 代码不改；
- 只把 `ref_mode` 改成 `fixed_band`；
- 设 `ref_band_index=3`；
- 跑一版和当前 `spatial_weighted` 对照。

这个实验的价值是判断：

> 问题主要出在“canonical reference 构造”，还是出在“显式 deformable alignment 这件事本身”。

如果 `fixed_band` 明显优于 `spatial_weighted`，说明：

- 锚点明确化是正确方向；
- 新模块值得继续做。

如果 `fixed_band` 依然不行，说明：

- 问题可能更深，出在 offset 学习、辅助损失、插入位置，或者数据本身并不需要这么强的对齐。

---

## 9. 为什么不建议直接复刻 CF 解码器式 CMDA

即使忽略本地 third-party 缺失实现的问题，也不建议直接把 CF 的 decoder CMDA 硬迁移到这里，原因有三点。

### 9.1 工作层级不一致

CF 的 CMDA 发生在：

- object query
- decoder stage
- 高层多尺度 memory

而你当前要替代的是：

- MS 输入端
- C2/stride=4
- dense feature alignment

这是两个完全不同的层级。

### 9.2 任务输入形式不一致

CF 主要是双分支模态交互。  
你这里是 7 个波段显式并列。

如果直接照搬，就要回答：

- query 是哪一支 band？
- memory 是 6 支拼一起还是逐支处理？
- 多尺度从哪里来？
- 每支 band 都建 decoder 吗？

这会把问题复杂化。

### 9.3 第一版最好不要把“对齐”和“检测 query 语义交互”绑死

你现在首先要验证的是：

> 固定锚点是否能比 canonical reference 更稳地做 band 配准融合

这件事完全可以在 backbone 输入端验证，没必要一开始就把 decoder 级 cross-modal attention 全搬进来。

---

## 10. 建议的实现形态

### 10.1 新模块建议命名

建议新增一个模块，例如：

- `engines/models/rtmsfdetr/rtdetrv4/engine/backbone/fixed_band_cmda.py`

类名例如：

- `FixedBandCMDA`
- `BandToAnchorCMDA`
- `AnchorBandDeformFusion`

其中我更推荐 `FixedBandCMDA`，因为命名最直接。

### 10.2 与现有 `MSBandSeparatedStemAlign` 的关系

可以把当前 `MSBandSeparatedStemAlign` 的 `aligner` 概念扩成两种模式：

1. `align.type = crgga`
2. `align.type = fixed_band_cmda`

流程上保持：

- `embed`
- `align/fuse`
- `merge`

这样不会破坏现有配置体系。

### 10.3 v1 先做单尺度版

第一版不要引入：

- 多尺度 flatten
- level_start_index
- encoder/decoder query
- 复杂 iterative refinement

先做：

- 单尺度 `H/4 x W/4`
- 每个 support 单独对 anchor 做 deformable cross interaction
- 输出仍为 `(B,7,C_e,H_4,W_4)`

更务实。

### 10.4 共享参数还是独立参数

我建议第一版对 6 个 support band **共享同一套 CMDA 参数**。

原因：

- 更接近当前 `CRGGA` 的“同一 aligner 复用到每个 band”；
- 参数量更小；
- 不容易因为数据量有限而过拟合到具体 band 编号。

如果后续观察到某些 band 明显需要特殊建模，再考虑：

- 按 band 学习 band embedding；
- 或者 support-specific bias。

---

## 11. 训练与损失建议

### 11.1 主损失仍然以检测损失为主

这是最重要的原则。

`FixedBandCMDA` 的存在目的不是生成漂亮 offset，而是提升最终检测。  
所以第一优先级仍应是：

- 分类损失
- 框回归损失
- 现有 detector 的主训练目标

### 11.2 辅助损失不要过重

建议第一版只保留轻量约束：

1. **offset 正则**
   \[
   \mathcal{L}_{off}
   \]
   防止采样位移过大。

2. **attention 熵或归一化约束**
   \[
   \mathcal{L}_{attn}
   \]
   防止权重退化。

3. **弱一致性约束**
   可选地约束 \(\mathbf{Y}_i\) 与 anchor 在局部上更一致，但权重要小。

不建议第一版就重压：

- 强 InfoNCE
- 强 contrastive
- 多层复杂互信息损失

因为这会再次把优化重心从检测拉向“自监督对齐”。

### 11.3 一个更合理的辅助监督方向

如果要做特征一致性，建议约束：

\[
\mathbf{Y}_i \leftrightarrow \mathbf{A}
\]

而不是约束：

\[
\widetilde{\mathbf{S}}_i \leftrightarrow \text{某个动态 reference}
\]

因为 anchor 是真实存在、几何目标明确的。

---

## 12. 锚点波段如何选择

固定锚点方案的一个核心设计就是选哪一支 band 当 anchor。

### 12.1 最务实的第一版选择

直接选中间 band：

- 7 通道时先试 `ref_band_index = 3`

理由：

- 不带先验偏见；
- 通常比最边缘波段更稳；
- 与当前代码里 `fixed_band` 的默认“中间 band”逻辑一致。

### 12.2 更工程化的选择

如果你对数据有先验，可以优先选：

- 边缘最清晰的 band；
- 目标对比度最稳定的 band；
- 噪声最小的 band。

### 12.3 不建议第一版做“动态 anchor 选择”

虽然理论上可以按图像动态选择 anchor，但第一版不要这么做。  
否则又回到了“reference 不稳定”的老问题。

---

## 13. 风险分析

固定锚点 CMDA 比 CRGGA 更合理，但仍有几个风险需要提前预判。

### 13.1 如果 anchor band 自身局部信息很差，会把其他 band 都往错误坐标系拉

这是固定 anchor 的天然代价。  
因此第一版最好：

- 只在一个相对稳定的 band 上试；
- 并保留对照实验。

### 13.2 若 support 和 anchor 的光谱差异过大，强行几何一致性可能不稳定

有些 band 在纹理和响应模式上差异很大。  
这时模块应更偏“cross fusion”，而不是强制像素级严格相似。

这也是我建议输出 \(\mathbf{Y}_i\) 而不是只输出纯 warp 后 \(\widetilde{\mathbf{S}}_i\) 的原因。

### 13.3 如果真实错位很小，复杂对齐仍可能收益有限

需要接受一个现实：  
最终实验结果仍有可能说明：

- `nomodule` 已经足够；
- 新模块收益很小；
- 或者只在少数类别上有改善。

但这时你得到的结论会更强，因为你已经排除了：

- canonical reference 设计偏差；
- 任务定义过宽；
- third-party 直接迁移不适配等因素。

---

## 14. 建议的实验顺序

推荐按下面顺序推进，而不是一步到位重写。

### Step 1：先做最低成本诊断

直接使用现有 `CRGGA`，只改：

- `ref_mode=fixed_band`
- `ref_band_index=3`

目的：

- 判断“固定锚点”这件事是否立刻缓解当前问题。

### Step 2：如果 Step 1 有改善，再做 `FixedBandCMDA v1`

实现单尺度、共享参数、anchor-query 的版本。

和下列方法做严格对照：

1. `baseline_nomodule`
2. `CRGGA(spatial_weighted)`
3. `CRGGA(fixed_band)`
4. `FixedBandCMDA v1`

### Step 3：如果 v1 有收益，再考虑增强

可选增强方向：

1. 多尺度版 `FixedBandCMDA`
2. 融合时增加 channel attention
3. anchor + support 双向交互
4. anchor confidence / reliability gating

---

## 15. 我对这个方向的最终判断

如果目标是“替换当前效果不理想的 CRGGA”，我认为**固定锚点 CMDA 是一个值得做、而且比继续堆 canonical reference 更靠谱的方向**。

原因不是因为 CF-CMDA 可以原样迁移，而是因为它启发了一个更合适的问题重写方式：

- 不再让模型同时学习“参考长什么样”和“怎么对齐”；
- 直接给它一个明确锚点；
- 让其余 band 朝这个锚点做 registration-aware fusion。

因此更准确的路线图应当是：

> **借 CF-CMDA 的“anchor query + deformable cross interaction”思想，设计一个适配当前 7-band 显式特征场景的 `FixedBandCMDA`，而不是直接复刻 CF 源码。**

---

## 16. 一句话版本

如果你要一个最简洁的判断：

> **可以做，而且值得做；但正确做法不是“把 CF-CMDA 搬过来”，而是“在 `MSBandSeparatedStemAlign` 的 per-band embedding 之后，新写一个固定波段锚点版的单尺度 Band-CMDA，用 anchor 作为 query/reference frame，让其余 6 个波段向它做 deformable cross fusion”。**

