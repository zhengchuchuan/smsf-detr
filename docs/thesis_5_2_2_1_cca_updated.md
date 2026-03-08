# 5.2.2.1 互补交叉注意力（CCA）补充：global_kv 与 pos2d

本文档基于 `docs/毕业论文-郑楚川-2310273066-计算机技术.pdf` 的 **5.2.2.1 互补交叉注意力（Complementary Cross-Attention, CCA）** 小节，
在不改变原有叙述主线的前提下，补充当前项目实现中启用的两项增强机制：

- `global_kv`：在 DSA 采样得到的 token 之外，追加全局池化得到的 **global tokens** 作为额外 K/V 记忆；
- `pos2d`（`pos_q/pos_kv`）：对 **Q 与 K** 添加二维正余弦位置编码（V 不加），并确保 global tokens 的位置编码与 token 拼接顺序对齐。

> 注：论文中 DSA 的详细推导位于 5.2.2.2，本小节仅在 CCA 语境下说明 DSA 输出 token 如何作为 K/V，以及 global_kv/pos2d 如何插入。

---

## 逐段落插入点（建议）

1) **在“引入 DSA 作为前置模块”段落之后**（原文约在图 12 后，开始描述 q/k/v 之前）插入：
- global_kv 的动机：结构化采样可能损失全局语义与长程上下文，全局 tokens 作为“全局记忆/兜底”；
- global_kv 的实现：`AdaptiveAvgPool2d(A_h,A_w)` + flatten，按 token 维拼接到 DSA tokens 后面；
- global_kv 对 token 长度的影响：`N = Hk*Wk + A_h*A_w`。

2) **在 q/k/v 线性映射公式之后、跨模态相关性矩阵公式之前**插入：
- pos2d 的动机：二维特征 flatten 成序列后，注意力缺少显式空间先验；
- pos_q/pos_kv 的插入位置：`q = proj_q(...) + pos_q`，`k = proj_k(...) + pos_kv`（V 不加）；
- global_kv 开启时 pos_kv 的拼接方式：`pos_kv = [pos_kv_dsa, pos_kv_global]`，顺序与 token 拼接一致。

---

## 更新后的 5.2.2.1 内容（可直接替换论文同名小节）

在多光谱溢油目标检测任务中，可见光图像主要反映地表物体的反射强度与纹理结构信息，能够提供较为丰富的空间细节。然而，在复杂陆地油田环境下，其成像结果易受到光照变化、阴影遮挡、土壤颜色相似性以及地表湿度变化等因素的干扰，导致溢油区域与背景之间的视觉对比度显著降低，判别特征甚至出现弱化或缺失。相比之下，多光谱图像能够在特定波段（如近红外或短波红外）对原油表现出更为显著的反射或吸收差异，有利于增强溢油区域的可分性，但其受限于传感器空间分辨率和大气散射等因素，纹理与结构细节相对不足。因此，可见光与多光谱图像在信息表达上呈现出明显的互补特性，但同时也伴随着成像机理差异所引入的空间错位与语义不一致问题。若未能对这种互补性与不对齐性进行有效建模，简单的特征融合往往难以充分发挥多模态优势，从而限制目标检测性能的提升。

可见光与多光谱特征在成像机理和分布特性上存在显著差异，导致跨模态特征在空间对应关系和语义表达上难以直接对齐。为有效挖掘不同模态之间的对齐特征与互补信息，本文引入互补交叉注意力模块（Complementary Cross-Attention, CCA），通过 Transformer 的跨模态交叉注意力机制，在特征层面显式建模两种模态之间的相关性，并通过双向信息交互实现跨模态语义校准与互补特征增强。在前置空间对齐模块缓解模态间几何错位的基础上，CCA 模块进一步自动发现并聚合具有高度相关性的跨模态对应特征，为后续检测头提供更加稳定且判别性更强的融合表示。

为提升跨模态注意力建模的稳定性与有效性，本文在跨模态交叉注意力计算之前，引入可变形采样对齐模块（Deformable Sampling Aligner, DSA）对两种模态特征进行空间对齐与结构化降采样处理。设可见光特征与多光谱特征分别为
$$
X^{rgb} \\in \\mathbb{R}^{C\\times H\\times W}, \\quad X^{ms} \\in \\mathbb{R}^{C\\times H\\times W}.
$$
DSA 在 stride 网格上对齐并采样得到稀疏 token，其网格大小记为 $H_k\\times W_k$，对应 token 数为 $N_s=H_kW_k$。令 $\\mathcal{E}(\\cdot)$ 表示二维特征图到序列的重排操作（例如 $\\mathbb{R}^{C\\times H\\times W}\\to\\mathbb{R}^{HW\\times C}$，或对 token 网格 $\\mathbb{R}^{C\\times H_k\\times W_k}\\to\\mathbb{R}^{N_s\\times C}$），则 DSA 的输出可记为
$$
S^{rgb}=\\mathcal{E}(\\mathrm{DSA}(X^{rgb}))\\in\\mathbb{R}^{N_s\\times C},\\quad
S^{ms}=\\mathcal{E}(\\mathrm{DSA}(X^{ms}))\\in\\mathbb{R}^{N_s\\times C}.
$$

**（新增）全局键值增强：global_kv。**

结构化采样在降低计算量的同时，可能削弱对全局语义与长程上下文的表达。为此，本文在 DSA token 的基础上追加一组全局 token 作为额外的 K/V 记忆（global key-value）。具体地，对输入特征施加自适应平均池化得到固定大小的全局网格（$A_h\\times A_w$，对应 `global_vert_anchors/global_horz_anchors`）：
$$
G^{rgb}=\\mathrm{AvgPool}_{A_h,A_w}(X^{rgb})\\in\\mathbb{R}^{C\\times A_h\\times A_w},\\quad
G^{ms}=\\mathrm{AvgPool}_{A_h,A_w}(X^{ms})\\in\\mathbb{R}^{C\\times A_h\\times A_w}.
$$
将其展平为 token 序列（$N_g=A_hA_w$）：
$$
g^{rgb}=\\mathcal{E}(G^{rgb})\\in\\mathbb{R}^{N_g\\times C},\\quad
g^{ms}=\\mathcal{E}(G^{ms})\\in\\mathbb{R}^{N_g\\times C}.
$$
最终，将 DSA token 与 global token 在 token 维拼接，得到增强后的键值序列：
$$
\\tilde{S}^{rgb}=[S^{rgb};\\, g^{rgb}]\\in\\mathbb{R}^{N\\times C},\\quad
\\tilde{S}^{ms}=[S^{ms};\\, g^{ms}]\\in\\mathbb{R}^{N\\times C},
\\quad N=N_s+N_g.
$$
其中 $[\\cdot;\\cdot]$ 表示在 token 维进行拼接。由此，CCA 的 K/V 不再仅依赖稀疏采样点，同时具备“局部对齐记忆（DSA tokens）+ 全局语义记忆（global tokens）”的混合库，在跨尺度或大位移场景下更为鲁棒。

**查询/键/值描述子构建（补充：K/V 基于 $\\tilde{S}$）。**

CCA 模块对两种模态分别生成查询（Query）、键（Key）和值（Value）描述子。为与实现保持一致，本文采用 $1\\times 1$ 卷积（或等价线性层）完成线性映射，并将二维特征图重排为序列形式：
$$
Q^{rgb}=\\mathcal{E}(W_q^{rgb}*\\hat{X}^{rgb})\\in\\mathbb{R}^{HW\\times C},\\quad
Q^{ms}=\\mathcal{E}(W_q^{ms}*\\hat{X}^{ms})\\in\\mathbb{R}^{HW\\times C},
$$
$$
K^{rgb}=W_k^{rgb}*\\tilde{S}^{rgb}\\in\\mathbb{R}^{N\\times C},\\quad
K^{ms}=W_k^{ms}*\\tilde{S}^{ms}\\in\\mathbb{R}^{N\\times C},
$$
$$
V^{rgb}=W_v^{rgb}*\\tilde{S}^{rgb}\\in\\mathbb{R}^{N\\times C},\\quad
V^{ms}=W_v^{ms}*\\tilde{S}^{ms}\\in\\mathbb{R}^{N\\times C}.
$$
其中 $W_q, W_k, W_v$ 为可学习映射参数，$*$ 表示卷积/线性映射；$\\hat{X}^{rgb},\\hat{X}^{ms}$ 表示经过模态归一化（下文给出）后的特征，用于构造更稳定的 Query。

**模态归一化（Modality Normalization）。**

由于 RGB 与多光谱特征在统计分布上存在显著差异，若直接计算跨模态相似度，容易受到模态偏差影响，从而削弱注意力权重的可靠性。为缓解上述问题，CCA 在特征交互前引入模态归一化操作，将一种模态的特征分布映射至另一模态的统计空间。以多光谱分支为例，首先对特征进行实例归一化：
$$
\\bar{X}^{ms}=\\frac{X^{ms}-\\mu(X^{ms})}{\\sigma(X^{ms})+\\varepsilon},
$$
其中 $\\mu(\\cdot),\\sigma(\\cdot)$ 为通道维均值与标准差，$\\varepsilon$ 为数值稳定项。随后利用轻量卷积网络预测可学习的尺度与偏置（以 RGB 分支作为条件）：
$$
\\beta^{ms}=f_{\\beta}(X^{rgb})+\\mu(X^{rgb}),\\quad
\\gamma^{ms}=f_{\\gamma}(X^{rgb})+\\sigma(X^{rgb}),
$$
最终得到分布对齐后的特征：
$$
\\hat{X}^{ms}=\\bar{X}^{ms}\\odot\\gamma^{ms}+\\beta^{ms},
$$
其中 $\\odot$ 表示逐元素乘。可见光分支的模态归一化过程与之对称。经归一化处理后的特征用于生成查询描述子，从而有效减小模态分布差异对跨模态相似度计算的干扰。

**（新增）二维正余弦位置编码：pos2d（pos_q / pos_kv）。**

将二维特征重排为序列后，注意力计算本身对 token 顺序不敏感；若缺少显式空间先验，在纹理重复或大范围匹配时容易出现歧义。为此，本文为 Query 与 Key 引入二维正余弦位置编码（2D sine-cosine positional embedding）。设温度系数为 $T$（对应实现中的 `pos_temperature`），令 $D_p=\\lfloor C/4\\rfloor$，对一维坐标 $u$ 定义频率：
$$
\\omega_i = \\frac{1}{T^{i/D_p}},\\quad i=0,1,\\dots,D_p-1.
$$
对二维网格位置 $(u,v)$（$u\\in[0,w-1], v\\in[0,h-1]$），定义二维位置编码：
$$
\\mathrm{PE}_{2D}(u,v)=\\big[\\sin(u\\omega),\\cos(u\\omega),\\sin(v\\omega),\\cos(v\\omega)\\big]\\in\\mathbb{R}^{C},
$$
其中 $\\sin(u\\omega)$ 表示对向量 $(u\\omega_0,\\dots,u\\omega_{D_p-1})$ 逐元素取 $\\sin$，其余项同理；当通道数无法被 4 整除时，其余通道以 0 填充。

在 CCA 中，位置编码的插入方式为：
$$
\\tilde{Q}=Q+P_q,\\quad \\tilde{K}=K+P_{kv},
$$
其中 $P_q\\in\\mathbb{R}^{HW\\times C}$ 由 $(H,W)$ 网格生成并加到 Query；$P_{kv}\\in\\mathbb{R}^{N\\times C}$ 由 token 网格生成并加到 Key。需要强调的是：实现中**仅对 Q 与 K 加位置编码，V 不加**，以保持值向量的纯内容表达。

当开启 global_kv 时，Key 的 token 由两部分拼接而成，因此其位置编码也需按相同顺序拼接：
$$
P_{kv}=[P_{kv}^{dsa};\\,P_{kv}^{global}],
$$
其中 $P_{kv}^{dsa}$ 由 $(H_k,W_k)$ 生成，对应 DSA tokens；$P_{kv}^{global}$ 由 $(A_h,A_w)$ 生成，对应 global tokens；拼接顺序与 $\\tilde{S}=[S;g]$ 完全一致，从而保证位置编码与 token 一一对齐。

**跨模态相关性矩阵与 Soft Attention 融合（补充：使用 $\\tilde{Q},\\tilde{K}$）。**

在完成特征描述子构建后，CCA 通过跨模态交叉注意力机制显式建模 RGB 与多光谱特征之间的对应关系。以 RGB 查询多光谱键值为例，构建跨模态相关性矩阵：
$$
M_{rgb}^{\\varepsilon}=\\frac{\\tilde{Q}^{rgb}(\\tilde{K}^{ms})^{\\top}}{\\sqrt{d}},\\quad
M_{ms}^{\\varepsilon}=\\frac{\\tilde{Q}^{ms}(\\tilde{K}^{rgb})^{\\top}}{\\sqrt{d}},
$$
其中 $d$ 为特征维度（缩放项用于稳定梯度传播）。随后通过 softmax 对相关性矩阵进行归一化得到相似度权重：
$$
M_{rgb}=\\mathrm{Softmax}(M_{rgb}^{\\varepsilon}),\\quad
M_{ms}=\\mathrm{Softmax}(M_{ms}^{\\varepsilon}),
$$
其中 softmax 通常对 Key 维进行归一化，使每个 Query 在另一模态中的匹配权重之和为 1。上述相似度矩阵刻画了不同模态特征在 token 空间的对应关系，高权重位置表示跨模态对齐程度较高的特征点。

在获得跨模态相关性权重后，CCA 进一步采用 Soft Attention 进行跨模态特征融合，通过将相似度矩阵与对应模态的值描述子进行加权求和，聚合来自另一模态的上下文信息：
$$
Y^{rgb}=M_{rgb}V^{ms},\\quad
Y^{ms}=M_{ms}V^{rgb}.
$$
随后，将融合后的特征序列回填为二维特征图形式，并通过卷积层映射至原始特征空间，得到最终的跨模态增强特征：
$$
\\hat{Y}^{rgb}=W_m*\\mathcal{E}^{-1}(Y^{rgb}),\\quad
\\hat{Y}^{ms}=W_n*\\mathcal{E}^{-1}(Y^{ms}),
$$
其中 $W_m, W_n$ 为卷积映射参数，$\\mathcal{E}^{-1}(\\cdot)$ 表示将序列恢复为二维特征图的逆重排操作。
