# rtmsfdetr_oil_rgb_msi_20260202_3cls 配置的损失函数说明

本文档对应配置：
configs/task/rtmsfdetr/oil_rgb_msi_20260202_3cls/rtmsfdetr_oil_rgb_msi_20260202_det_rtv4_hgnetv2_m_distill_dualstream_c2former_postblock_add_wbadd_c3c4c5_msbandsep_c2align_infonce_reg_globalkv_pos2d.yaml

## 1. 损失函数清单与权重（该配置实际启用）

### 1.1 RTv4Criterion（检测 + 蒸馏）
由 RT-DETRv4 配置合并得到的最终 weight_dict：

- loss_mal: 1
- loss_bbox: 5
- loss_giou: 2
- loss_fgl: 0.15
- loss_ddf: 1.5
- loss_distill: 5（在 rtv4_hgnetv2_m_coco.yml 中覆盖了 base 的 10）

说明：losses = ["mal", "boxes", "local", "distill"]，其中 local 会产生 loss_fgl 与 loss_ddf。

蒸馏权重可能在训练中自适应变化：配置中的 distill_adaptive_params 允许根据 encoder 梯度比例动态调整 loss_distill，参数为：
- enabled: True
- rho: 3.5
- delta: 0.25
- default_weight: 15

即：初始权重为 5，但训练过程中可能被调整为接近 default_weight 或按规则缩放。

### 1.2 MS band-separated stem 对齐损失（CRGGA / GroupwiseDeformableAlign2D）
来自 backbone_ms_band_sep.align：

- loss_ms_group_align（InfoNCE 对齐损失）: 0.02
- loss_ms_group_offset（offset 正则）: 0.01
- loss_ms_group_attn_entropy（注意力熵正则）: 0.001
- loss_ms_group_attn（注意力和正则）: 0（未配置）

说明：这些权重在 GroupwiseDeformableAlign2D 内部已乘上对应系数后返回；RTv4Criterion 只在 weight_dict 里另行配置时才会再次缩放（本配置未额外设置）。

## 2. 数学公式（与实现一致的简化写法）

记号说明：
- 预测分类 logits 为 s，经过 sigmoid 为 p = σ(s)
- 匹配后的 GT 类别 one-hot 为 y
- 匹配后的 GT box 为 b*，预测 box 为 b
- IoU = IoU(b, b*)（或 GIoU）
- N 为匹配后的正样本数量（num_boxes）

### 2.1 loss_mal（MAL 分类损失）
实现中先计算 IoU，并作为 soft target：

$$
\hat{y} = IoU^{\gamma}
$$

权重项（mal_alpha 未配置时）：

$$
weight = p^{\gamma} (1 - y) + y
$$

最终为带权 BCE：

$$
L_{mal} = \frac{1}{N} \sum_{q} BCEWithLogits(s_q, \hat{y}_q; weight_q)
$$

其中 $\gamma = 1.5$（来自配置）。

### 2.2 loss_bbox（L1 回归损失）

$$
L_{bbox} = \frac{1}{N} \sum \|b - b^*\|_1
$$

### 2.3 loss_giou（GIoU 损失）

$$
L_{giou} = \frac{1}{N} \sum (1 - GIoU(b, b^*))
$$

### 2.4 loss_fgl（Fine-Grained Localization, FGL）
FGL 采用 D-FINE 的离散回归分布。对每个边界距离的 GT 值，得到左右 bin 及插值权重：

$$
(d_L, d_R, w_L, w_R) = bbox2distance(b^*)
$$

设预测分布为 p，则：

$$
L_{fgl} = \frac{1}{N} \sum w_{IoU} \cdot (w_L \cdot CE(p, d_L) + w_R \cdot CE(p, d_R))
$$

其中 w_{IoU} 为匹配后的 IoU 权重（代码中使用对角 IoU）。

### 2.5 loss_ddf（Decoupled Distillation Focal, DDF）
使用 teacher corners 与 student corners 的 KL 散度（温度 T = 5）：

$$
L_{ddf} = \frac{1}{Z} \sum w_t \cdot T^2 \cdot KL(softmax(p/T) || softmax(p^*/T))
$$

其中 w_t 来自 teacher 置信度与 IoU 组合；实现中对正负样本分别取均值再按数量加权归一化。

### 2.6 loss_distill（特征蒸馏）
对 student 和 teacher encoder feature 做 L2 归一化后计算 cosine 距离：

$$
L_{distill} = \frac{1}{M} \sum (1 - cos(\hat{f}_s, \hat{f}_t))
$$

### 2.7 loss_ms_group_align（InfoNCE 对齐损失）
在对齐后的特征上采样 patch，计算 InfoNCE：

$$
logits_{ij} = \frac{\hat{f}_i^T \hat{g}_j}{\tau}
$$

$$
L_{nce} = \frac{1}{2}(CE(logits, I) + CE(logits^T, I))
$$

其中 $\tau = 0.2$，patch 采样数量 64，patch 大小 5，支持 loss_downsample = 0.5 下采样。

### 2.8 loss_ms_group_offset（偏移正则）
对 attention 归一化后，计算融合 offset 的幅度：

$$
L_{offset} = E[\sqrt{\Delta x^2 + \Delta y^2 + 1e-8}]
$$

其中 $E[\cdot]$ 表示期望/取平均；在本项目实现中等价于对当前 mini-batch 以及空间位置（并在需要时包含 keypoint/通道等维度）做 `.mean()` 聚合，得到标量损失。

### 2.9 loss_ms_group_attn_entropy（注意力熵正则）

$$
L_{ent} = E[-\sum_k p_k \log p_k]
$$

## 3. 实现位置（代码文件与函数）

### 3.1 RTv4Criterion 检测与蒸馏
- loss_mal / loss_bbox / loss_giou / loss_fgl / loss_ddf / loss_distill：
  - engines/models/rtmsfdetr/rtdetrv4/engine/rtv4/rtv4_criterion.py
  - 关键函数：loss_labels_mal, loss_boxes, loss_local, loss_distillation, unimodal_distribution_focal_loss
- FGL 的 GT 分布构造：
  - engines/models/rtmsfdetr/rtdetrv4/engine/rtv4/dfine_utils.py
  - 关键函数：bbox2distance, translate_gt, weighting_function

### 3.2 MS band-separated stem 对齐（CRGGA）
- 模块入口：
  - engines/models/rtmsfdetr/rtdetrv4/engine/backbone/ms_band_sep.py
  - 类：MSBandSeparatedStemAlign
- 对齐与正则损失：
  - engines/models/rtmsfdetr/rtdetrv4/engine/backbone/group_deform_align.py
  - 类：CRGGA / GroupwiseDeformableAlign2D
- InfoNCE 实现与对齐损失：
  - engines/models/rtmsfdetr/rtdetrv4/engine/backbone/deform_align.py
  - 函数：_info_nce_loss, loss_calculate

## 4. 配置来源（权重和超参）

- RTv4Criterion 权重与 loss 列表：
  - engines/models/rtmsfdetr/rtdetrv4/configs/base/rtv4.yml
- loss_distill 覆盖与 distill_adaptive_params：
  - engines/models/rtmsfdetr/rtdetrv4/configs/rtv4/rtv4_hgnetv2_m_coco.yml
- MS band-separated stem 对齐超参与权重：
  - configs/task/rtmsfdetr/oil_rgb_msi_20260202_3cls/rtmsfdetr_oil_rgb_msi_20260202_det_rtv4_hgnetv2_m_distill_dualstream_c2former_postblock_add_wbadd_c3c4c5_msbandsep_c2align_infonce_reg_globalkv_pos2d.yaml

## 5. 论文小节：损失函数设计（与本项目实现一致）

我们采用多重损失的加权组合来训练模型，以同时优化检测性能与跨模态融合/对齐质量。总损失由两部分构成：RT-DETRv4 检测头的监督项，以及双流骨干中 MS band-separated 对齐模块产生的辅助对齐与正则项。其总损失可写为：

$$
L_{total} =
\lambda_{mal} L_{mal}
+ \lambda_{bbox} L_{bbox}
+ \lambda_{giou} L_{giou}
+ \lambda_{fgl} L_{fgl}
+ \lambda_{ddf} L_{ddf}
+ \lambda_{distill} L_{distill}
+ \lambda_{align} L_{align}
+ \lambda_{off} L_{off}
+ \lambda_{ent} L_{ent}. \qquad (19)
$$

在实现层面，训练时对模型输出的各 loss 项直接求和作为反向传播的标量目标（即优化目标为所有加权损失之和）。其中，各项含义如下（具体数学形式见第 2 节，实现位置见第 3 节）：

- $L_{mal}$：分类置信度监督项。本项目采用 MAL（Matching-Aware Loss）作为分类损失。其属于“质量感知（quality-aware）的 soft label + 带权 BCE”一类形式：soft label 由匹配后的 IoU（幂指数 $\gamma$）构造，并使用预测概率的幂作为负样本抑制权重。该框架也可切换为 VFL（Varifocal Loss）风格项，但本文档对应的实验配置实际启用的是 MAL。
- $L_{bbox}$：边界框坐标的 L1 回归损失。
- $L_{giou}$：广义 IoU 损失，用于衡量预测框与真值框的几何重叠，并进一步提高边界框预测质量。
- $L_{fgl}$：Fine-Grained Localization（FGL）损失，对应 D-FINE 的分布式回归监督，用于提升边界框细粒度定位精度。
- $L_{ddf}$：Decoupled Distillation Focal（DDF）损失，对应局部分布的蒸馏监督，用于强化定位分布学习。
- $L_{distill}$：特征蒸馏损失（cosine distance），对齐 teacher/student encoder 特征，以提升表征质量与收敛稳定性。
- $L_{align}$：跨模态（或跨 band）特征一致性/对齐损失。本配置在 MS band-separated stem 中启用了 CRGGA（Canonical Reference Guided Groupwise Alignment），并使用 patch InfoNCE 来定义对齐损失，以鼓励不同 band/模态特征在空间对齐后仍保持语义一致。
  - 说明：该模块同时支持 “余弦相似度一致性损失” 与 “InfoNCE 损失” 两种定义方式；本文档所对应实验配置使用 InfoNCE，因此论文中建议以 InfoNCE 形式表述为主，并补充说明“可退化为余弦一致性项”。余弦一致性写作可采用如下形式（与常见论文表述一致）：

$$
L_{align}^{cos} = 1 - \frac{1}{BHW}\sum_{i=1}^{B}\sum_{h=1}^{H}\sum_{w=1}^{W}
\frac{\tilde{\mathbf{F}}_{a}(i,h,w)\cdot\mathbf{F}_{b}(i,h,w)}
{\|\tilde{\mathbf{F}}_{a}(i,h,w)\|\ \|\mathbf{F}_{b}(i,h,w)\|}. \qquad (20)
$$

其中 $\tilde{\mathbf{F}}_{a}$ 表示对齐/偏移学习后的特征（例如 IR/MS 分支经过对齐模块后的特征），$\mathbf{F}_{b}$ 表示参考模态特征（例如 RGB/可见光特征）。通过最小化 $L_{align}^{cos}$（即最大化余弦相似度），可以鼓励跨模态特征的空间对齐与语义一致。

在本文档对应实验中，我们采用 patch InfoNCE 来定义对齐损失。设在同一批次内采样得到的特征向量对为 $\{(\mathbf{f}_i,\mathbf{g}_i)\}_{i=1}^{N}$，并记
$$
logits_{ij}=\frac{\hat{\mathbf{f}}_i^\top \hat{\mathbf{g}}_j}{\tau}, \qquad (21)
$$
其中 $\hat{\mathbf{f}},\hat{\mathbf{g}}$ 为 L2 归一化特征，$\tau$ 为温度系数，则对齐项可写为对称形式：
$$
L_{align}^{nce}=\frac{1}{2}\left(CE(logits, I)+CE(logits^\top, I)\right), \qquad (22)
$$
其中 $I$ 表示正样本匹配的对角标签（同索引为正样本，其余为负样本）。这里的 $CE(\cdot)$ 指交叉熵损失：对 $logits$ 的每一行做 softmax，并以对角项作为正确类别，具体为
$$
CE(logits, I)=\frac{1}{N}\sum_{i=1}^{N}-\log\frac{\exp(logits_{ii})}{\sum_{j=1}^{N}\exp(logits_{ij})}.
$$
实现上通常等价于设置 `labels=[0,1,2,...,N-1]` 并调用 `cross_entropy(logits, labels)`；$CE(logits^\top, I)$ 为对称项。

- $L_{off}$：对齐模块中的 offset 正则项，用于抑制过大形变并稳定训练。
- $L_{ent}$：对齐模块中的注意力熵正则项，用于避免采样注意力退化并提升稳定性。

### 5.1 本配置的权重设置（$\lambda$）

在本配置中，检测/蒸馏项的权重设置为（配置来源见第 4 节）：  
$\lambda_{mal}=1$、$\lambda_{bbox}=5$、$\lambda_{giou}=2$、$\lambda_{fgl}=0.15$、$\lambda_{ddf}=1.5$、$\lambda_{distill}=5$。

MS band-separated 对齐模块的权重直接在模块内部设置，并以损失项的形式回传：  
$\lambda_{align}=0.02$、$\lambda_{off}=0.01$、$\lambda_{ent}=0.001$（其中注意力和约束项 $\lambda_{attn}=0$，未启用）。

### 5.2 梯度引导自适应调制（GAM）在本项目中的对应实现

项目中存在“基于梯度统计动态调整 $\lambda$ 的机制”，但其作用对象是蒸馏项的权重 $\lambda_{distill}$，用于适应训练过程中不断变化的优化动态并平衡检测监督与蒸馏监督：

- 配置入口：`distill_adaptive_params`（见 `engines/models/rtmsfdetr/rtdetrv4/configs/rtv4/rtv4_hgnetv2_m_coco.yml:22`），包含 $\rho$、$\delta$ 与 `default_weight`。
- 实现逻辑：训练过程中统计 encoder transformer 的梯度占比（百分比），若偏离区间 $[\rho-\delta,\,\rho+\delta]$ 则按比例更新 $\lambda_{distill}$（更新代码见 `engines/models/rtmsfdetr/rtdetrv4/engine/solver/det_solver.py:100`）。

因此，在论文表述上可将其描述为一种 GAM（Gradient-guided Adaptive Modulation）式的自适应权重调制策略，但需要明确：本项目当前实现是对蒸馏权重 $\lambda_{distill}$ 进行自适应更新，而非对所有损失项的 $\lambda$ 同时更新。

（各损失项的具体数学形式与实现位置，可分别参考本文档第 2 节与第 3 节。）
