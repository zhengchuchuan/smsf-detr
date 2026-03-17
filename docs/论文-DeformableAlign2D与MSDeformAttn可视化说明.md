# DeformableAlign2D 与 MSDeformAttn 可视化说明

本文档对应以下两张可视化图：

- `docs/deformable_align2d_visualization_cn.svg`
- `docs/ms_deform_attn_visualization_cn.svg`

它们的目标是帮助你在毕业论文中准确绘制：

1. `DeformableAlign2D` 的结构图
2. `MSDeformAttn` 的结构图
3. 二者在“可变形采样”层面的共性与差异

---

## 1. 图一：`DeformableAlign2D` 结构图

文件：

- `docs/deformable_align2d_visualization_cn.svg`

对应代码：

- 类定义：`engines/models/rtmsfdetr/rtdetrv4/engine/backbone/deform_align.py:69-716`
- 初始化结构：`engines/models/rtmsfdetr/rtdetrv4/engine/backbone/deform_align.py:76-243`
- offset/attention 预测：`engines/models/rtmsfdetr/rtdetrv4/engine/backbone/deform_align.py:324-389`
- deformable sampling：`engines/models/rtmsfdetr/rtdetrv4/engine/backbone/deform_align.py:429-543`
- 可选损失：`engines/models/rtmsfdetr/rtdetrv4/engine/backbone/deform_align.py:662-708`

### 1.1 这张图表达的重点

这张图展示的是：

- 输入为 `x_ref` 和 `x_src`
- 先将两者拼接后预测 `offset_x`、`offset_y` 和 `attention`
- 再用 `grid_sample` 对 `x_src` 做局部可变形采样
- 最后按注意力权重聚合，得到 `fused_features`

因此，这个模块的核心语义应写成：

> **“以参考特征为条件，预测源特征的局部采样位置与聚合权重，并将源特征对齐到参考特征坐标系。”**

### 1.2 当前项目里最常见的论文口径

在你的项目实际使用中，`DeformableAlign2D` 通常不是单独裸用，而是被 `FixedBandCMDA` 调用：

- `engines/models/rtmsfdetr/rtdetrv4/engine/backbone/fixed_band_cmda.py:150-173`
- `engines/models/rtmsfdetr/rtdetrv4/engine/backbone/fixed_band_cmda.py:221-243`

因此论文中更推荐写成：

> **“可变形对齐子模块（DeformableAlign2D）”**

而不是单独把它写成整个方法名。

---

## 2. 图二：`MSDeformAttn` 结构图

文件：

- `docs/ms_deform_attn_visualization_cn.svg`

对应代码：

- 类定义：`engines/models/msifdetr/common/ops/modules/ms_deform_attn.py:40-140`
- PyTorch 核心实现：`engines/models/msifdetr/common/ops/functions/ms_deform_attn_func.py:27-50`

### 2.1 这张图表达的重点

这张图展示的是：

- `query` 直接预测 `sampling_offsets`
- `query` 直接预测 `attention_weights`
- `memory/input_flatten` 经过 `value_proj`
- 由 `reference_points + offsets` 确定采样位置
- 在 `value` 上做双线性采样并加权求和
- 再通过 `output_proj` 输出

因此，这个模块的核心语义应写成：

> **“由 query 引导的可变形特征读取模块。”**

### 2.2 在当前 `StemCFInteractive2D` 中的使用方式

当前项目里，`MSDeformAttn` 被用于：

- `engines/models/rtmsfdetr/rtdetrv4/engine/backbone/stem_cf_interactive.py:97-102`
- `engines/models/rtmsfdetr/rtdetrv4/engine/backbone/stem_cf_interactive.py:159-190`

此时它是：

- 单尺度 `n_levels=1`
- `query` 来源于主干浅层特征
- `memory` 来源于残差 MS 分支
- 输出是一个交互增量 `delta`

因此论文中更推荐写成：

> **“单尺度可变形跨分支读取模块”**

或者：

> **“query 引导的单尺度可变形交叉注意力模块”**

---

## 3. 这两张图应该如何配合论文使用

推荐的用法是：

1. 若你只想解释残差分支内部的对齐机制，用图一。
2. 若你只想解释 `StemCF` 里的跨分支交互机制，用图二。
3. 若你想说明“二者底层都属于 deformable sampling，但语义不同”，就把两张图并排放。

最适合你的正文表述是：

> `DeformableAlign2D` 主要用于分支内部显式对齐，其输出是参考坐标系下的对齐特征；  
> `MSDeformAttn` 主要用于跨分支交互，其输出是 query 引导下从 support/value 中读取的特征增量。

---

## 4. 两张图中最值得你在论文里保留的信息

如果你要自己重新绘图，最应该保留的是：

### 4.1 `DeformableAlign2D`

- 输入是 `x_ref` 和 `x_src`
- 中间要有：
  - `concat`
  - `offset_head`
  - `attention_head`
  - `base_grid`
  - `grid_sample`
  - `weighted sum`
- 维度一定要写：
  - `offset_x, offset_y: (B,K,H,W)`
  - `sampled_features: (B,K,C,H,W)`
  - `fused_features: (B,C,H,W)`

### 4.2 `MSDeformAttn`

- 输入是：
  - `query`
  - `reference_points`
  - `input_flatten`
- 中间要有：
  - `sampling_offsets(query)`
  - `attention_weights(query)`
  - `value_proj(input_flatten)`
  - `sampling_locations`
  - `grid_sample`
  - `output_proj`
- 维度一定要写：
  - `sampling_offsets: (B,Len_q,N_h,L,P,2)`
  - `attention_weights: (B,Len_q,N_h,L·P)`
  - `output: (B,Len_q,C)`

---

## 5. 推荐的论文图题

你可以直接用下面这两个图题：

### 图一题目

> **图 x.x DeformableAlign2D 的结构与局部可变形对齐过程**

### 图二题目

> **图 x.x MSDeformAttn 的结构与 query 引导的可变形特征读取过程**

如果你并排放这两张图，可以写成：

> **图 x.x DeformableAlign2D 与 MSDeformAttn 的结构对比示意图**

---

## 6. 一句话结论

这两张图共同支持的结论是：

> **两者底层都属于基于可学习偏移的局部双线性采样与加权聚合，但 `DeformableAlign2D` 偏向显式对齐，`MSDeformAttn` 偏向 query 引导的可变形特征读取。**
