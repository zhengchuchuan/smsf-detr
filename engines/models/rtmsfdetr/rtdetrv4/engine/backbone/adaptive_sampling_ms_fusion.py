from __future__ import annotations

"""
本文件实现了 RTMSF-DETR 中的 ASMF (Adaptive Sampling Multispectral Fusion) 融合模块。

历史命名：
- 早期实现/配置中该模块常被称为 “C2Former 风格融合”（因此你可能仍会看到 C2Former/DSA/CCA 的命名残留）；
- 为避免破坏旧配置，本文件仍保留了一些旧类名/别名（见文件末尾的 alias）。

核心思想（对应论文叙述，但这里用更“拆分/工程化”的方式实现）：
- DSA (Deformable Sampling Aligner)：预测跨模态的二维粗对齐 offset，并在低分辨率网格上做采样，
  用于降低后续跨注意力的 token 数量（类似论文中的 AFS：Adaptive Feature Sampling）。
- CCA (Complementary Cross Attention)：用全分辨率的 Query 去关注由 DSA 采样得到的 Key/Value，
  输出与输入同分辨率的增强特征（类似论文中的 ICA：用于对齐/互补融合）。

注意：
- 该实现是“同尺度”融合：要求两路输入特征图尺寸一致（B,C,H,W）。
- offset 是按 group 预测/采样（不是逐通道），因此“对齐”是对一组通道共享同一张采样网格。
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mrt_fusion import _build_2d_sincos_pos_embed


class LayerNormProxy(nn.Module):
    """对 BCHW 特征做 LayerNorm 的轻量包装。

    PyTorch 的 `nn.LayerNorm` 默认作用在最后一维，因此这里把张量从 BCHW 转为 BHWC，
    对 C 维做归一化后再转回 BCHW。
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(int(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # BCHW -> BHWC，LayerNorm 在最后一维(C)上做归一化。
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        # BHWC -> BCHW
        return x.permute(0, 3, 1, 2)


class ModalityNorm(nn.Module):
    """跨模态的“条件归一化/调制”模块。

    直觉：用 lr(当前模态/引导模态) 预测 (gamma, beta) 去调制 ref(被调制模态) 的归一化特征。

    计算形式接近：
        y = IN(ref) * gamma(lr) + beta(lr)

    其中：
    - `IN(ref)`：对 ref 做 InstanceNorm（不带 affine），消除 ref 的强度/风格差异；
    - `gamma/beta`：由 lr 通过卷积预测（learnable=True 时），并可选择 residual 方式
      将 lr 的统计量 (mean/std) 加回去，保证初始化更稳定（初始近似恒等映射）。
    """

    def __init__(self, channels: int, *, use_residual: bool = True, learnable: bool = True) -> None:
        super().__init__()
        self.learnable = bool(learnable)
        self.use_residual = bool(use_residual)
        self.norm_layer = nn.InstanceNorm2d(int(channels), affine=False)

        if self.learnable:
            # 用 lr 生成 gamma/beta 的“条件网络”。
            # 这里先做一个 3x3 + ReLU 提取局部上下文，再分别预测 gamma / beta。
            self.conv = nn.Sequential(
                nn.Conv2d(int(channels), int(channels), kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
            )
            self.conv_gamma = nn.Conv2d(int(channels), int(channels), kernel_size=3, padding=1, bias=True)
            self.conv_beta = nn.Conv2d(int(channels), int(channels), kernel_size=3, padding=1, bias=True)

            # 重要：将 gamma/beta 分支初始化为 0，使得模块初始接近恒等（更利于稳定训练）。
            nn.init.zeros_(self.conv_gamma.weight)
            nn.init.zeros_(self.conv_gamma.bias)
            nn.init.zeros_(self.conv_beta.weight)
            nn.init.zeros_(self.conv_beta.bias)

    def forward(self, lr: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        # 1) 对 ref 做 InstanceNorm，得到“去风格”的基底特征。
        ref_normed = self.norm_layer(ref)

        if self.learnable:
            # 2) 用 lr 预测 gamma/beta（与 ref 同形状的逐像素调制参数）。
            x = self.conv(lr)
            gamma = self.conv_gamma(x)
            beta = self.conv_beta(x)

        # 3) 计算 lr 的 per-channel 统计量（mean/std），用于 residual/非 learnable 情况。
        b, c, h, w = lr.shape
        lr_flat = lr.view(b, c, h * w)
        lr_mean = torch.mean(lr_flat, dim=-1, keepdim=True).view(b, c, 1, 1)
        lr_std = torch.std(lr_flat, dim=-1, keepdim=True).view(b, c, 1, 1)

        if self.learnable:
            if self.use_residual:
                # residual：让预测的 gamma/beta 在 lr 的统计量附近微调（更稳定）。
                gamma = gamma + lr_std
                beta = beta + lr_mean
            else:
                # 非 residual：让 gamma 初始接近 1（因为 gamma 分支 init 为 0）。
                gamma = 1.0 + gamma
        else:
            # 不学习时：退化为用 lr 的 mean/std 直接去调制 ref_normed。
            gamma = lr_std
            beta = lr_mean

        # 4) 条件调制：把 ref 的归一化特征映射到 lr 统计空间（实现跨模态“对齐/适配”）。
        return ref_normed * gamma + beta


class DeformableSamplingAligner2D(nn.Module):
    """
    DSA (Deformable Sampling Aligner，形变采样对齐器).

    作用：
    - 预测跨模态的二维 offset（粗对齐）；
    - 在低分辨率采样网格上用 `grid_sample` 从高分辨率特征中采样出少量 tokens，
      供后续跨注意力使用，从而显著降低注意力计算量。

    Notes:
    - 只对第一个输入 x_a 使用 (ref + offset) 进行采样；第二个输入 x_b 使用 ref 直接采样。
      这样就把“对齐谁/offset 作用在哪一路”的决策放到更高层（见 ASMFusion2D.offset_on）。
    - offset 是按 group 预测：每个 group 的通道共享同一个 2D 采样网格（不是逐通道 offset）。

    输入/输出：
    - 输入：x_a/x_b ∈ R^{B×C×H×W}，要求形状完全一致，且 C==d_model
    - 输出：a_sampled/b_sampled ∈ R^{B×C×1×(Hk*Wk)}，其中 Hk,Wk 由 stride=cca_stride 决定
    """

    def __init__(
        self,
        *,
        d_model: int,
        groups: int,
        cca_stride: int,
        offset_range_factor: float,
        no_offset: bool,
        offset_kernel_size: int,
        padding_mode: str,
        align_corners: bool,
    ) -> None:
        super().__init__()
        d_model = int(d_model)
        groups = int(groups)
        cca_stride = int(cca_stride)
        offset_kernel_size = int(offset_kernel_size)

        if d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {d_model}")
        if groups <= 0 or d_model % groups != 0:
            raise ValueError(f"Invalid groups={groups}; must divide d_model={d_model}")
        if cca_stride <= 0:
            raise ValueError(f"cca_stride must be > 0, got {cca_stride}")
        if offset_kernel_size <= 0:
            raise ValueError(f"offset_kernel_size must be > 0, got {offset_kernel_size}")
        if offset_kernel_size % 2 == 0:
            offset_kernel_size += 1
        padding_mode = str(padding_mode).strip().lower()
        if padding_mode not in {"zeros", "border", "reflection"}:
            raise ValueError(f"Unsupported padding_mode={padding_mode}")

        self.d_model = d_model
        self.n_groups = groups
        self.n_group_channels = d_model // groups
        self.cca_stride = cca_stride
        self.offset_range_factor = float(offset_range_factor)
        self.no_offset = bool(no_offset)
        self.padding_mode = padding_mode
        self.align_corners = bool(align_corners)

        # 将两路特征 concat 后用 1x1 压回 d_model，用于预测 offset。
        # 这样 Offset prediction head 的输入通道数与 d_model 对齐，便于后续 group 切分。
        self.proj_combinq = nn.Conv2d(2 * d_model, d_model, kernel_size=1, stride=1, padding=0)

        # Offset prediction head：预测二维 offset（dx, dy）。
        # - depthwise conv：每个通道独立卷积（但这里的通道是 group 内通道数）
        # - stride=cca_stride：在低分辨率网格上预测 offset（同时起到降采样 token 的作用）
        # - 最后 1x1 输出 2 通道，对应 (dy, dx)（后面会再 permute 成 (Hk,Wk,2)）
        self.conv_offset = nn.Sequential(
            nn.Conv2d(
                self.n_group_channels,
                self.n_group_channels,
                kernel_size=offset_kernel_size,
                stride=cca_stride,
                padding=offset_kernel_size // 2,
                groups=self.n_group_channels,
            ),
            LayerNormProxy(self.n_group_channels),
            nn.GELU(),
            nn.Conv2d(self.n_group_channels, 2, kernel_size=1, stride=1, padding=0, bias=False),
        )

    @torch.no_grad()
    def _get_ref_points(self, h: int, w: int, b: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        # 构造 grid_sample 所需的参考网格（归一化到 [-1,1]）。
        # 这里以像素中心 0.5..H-0.5 作为采样点，避免边界偏置。
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, h - 0.5, h, dtype=dtype, device=device),
            torch.linspace(0.5, w - 0.5, w, dtype=dtype, device=device),
            indexing="ij",
        )
        ref = torch.stack((ref_y, ref_x), dim=-1)
        # 注意：grid_sample 期望的坐标范围是 [-1, 1]，且最后一维顺序为 (x, y)。
        # 这里先存成 (y, x) 便于计算，再在调用 grid_sample 前做索引重排。
        ref[..., 1].div_(w).mul_(2).sub_(1)
        ref[..., 0].div_(h).mul_(2).sub_(1)
        # 扩展到 (B*groups, H, W, 2)：每个 group 单独一张采样网格。
        return ref.unsqueeze(0).expand(b * self.n_groups, -1, -1, -1)

    def forward(self, x_a: torch.Tensor, x_b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x_a.shape != x_b.shape:
            raise ValueError(f"DSA expects same shapes, got a={x_a.shape} b={x_b.shape}")
        if x_a.ndim != 4:
            raise ValueError(f"DSA expects BCHW tensors, got {x_a.shape}")
        b, c, h, w = x_a.shape
        if c != self.d_model:
            raise ValueError(f"DSA expects C={self.d_model}, got {c}")

        # 1) 拼接两模态特征 -> 1x1 压缩为 d_model，用于 offset 预测。
        x = torch.cat([x_a, x_b], dim=1)
        combin_q = self.proj_combinq(x)

        # 2) 按 group 切分通道：每个 group 独立预测 offset。
        # q_off: (B*groups, Cg, H, W)，Cg = d_model/groups
        q_off = combin_q.view(b, self.n_groups, self.n_group_channels, h, w).reshape(
            b * self.n_groups, self.n_group_channels, h, w
        )
        # offset: (B*groups, 2, Hk, Wk)，Hk/Wk 由 stride=cca_stride 决定
        offset = self.conv_offset(q_off)
        h_k, w_k = offset.shape[2], offset.shape[3]
        n_sample = h_k * w_k

        # 3) 控制 offset 的数值范围，避免训练中发散：
        #    tanh -> [-1,1]，再乘以 (1/Hk, 1/Wk) 的尺度和可配置的 range_factor。
        if self.offset_range_factor > 0:
            offset_range = torch.tensor(
                [1.0 / float(h_k), 1.0 / float(w_k)], device=offset.device, dtype=offset.dtype
            ).view(1, 2, 1, 1)
            offset = torch.tanh(offset) * offset_range * float(self.offset_range_factor)

        # 4) (B*groups, 2, Hk, Wk) -> (B*groups, Hk, Wk, 2)
        offset = offset.permute(0, 2, 3, 1)
        if self.no_offset:
            # 消融：关闭 offset，对齐退化为固定网格采样。
            offset = torch.zeros_like(offset)

        # 5) 构造参考采样网格 ref，并得到两路的采样位置。
        ref = self._get_ref_points(h_k, w_k, b, offset.dtype, offset.device)
        if self.offset_range_factor >= 0:
            a_pos = ref + offset
            b_pos = ref
        else:
            # 兼容历史实现：如果 range_factor < 0，则直接 tanh 截断到 [-1,1]。
            a_pos = torch.tanh(ref + offset)
            b_pos = torch.tanh(ref)

        # 6) 在原始高分辨率特征 (H,W) 上，按低分辨率网格 (Hk,Wk) 的坐标进行采样。
        #    grid_sample 的 grid 最后一维是 (x,y)，因此这里用 [...,(1,0)] 交换顺序。
        a_sampled = F.grid_sample(
            input=x_a.reshape(b * self.n_groups, self.n_group_channels, h, w),
            grid=a_pos[..., (1, 0)],
            mode="bilinear",
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )
        b_sampled = F.grid_sample(
            input=x_b.reshape(b * self.n_groups, self.n_group_channels, h, w),
            grid=b_pos[..., (1, 0)],
            mode="bilinear",
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )

        # 7) 将采样后的 (Hk,Wk) 展平为 token 维度 N=Hk*Wk，供后续注意力使用。
        #    输出形状固定为 (B,C,1,N)，与后续 1x1 Conv 投影兼容。
        a_sampled = a_sampled.view(b, c, 1, n_sample)
        b_sampled = b_sampled.view(b, c, 1, n_sample)

        return a_sampled, b_sampled


class ComplementaryCrossAttention2D(nn.Module):
    """
    CCA (Complementary Cross Attention，互补跨注意力).

    作用：
    - 使用“全分辨率”的 Query（H*W 个位置）去关注 DSA 提供的“低分辨率采样”Key/Value（N 个 token），
      从而在保持输出分辨率不变的前提下，把跨模态的互补信息注入到每个像素位置。

    输入/输出：
    - 输入：x_a/x_b ∈ R^{B×C×H×W}
    - 输入：a_sampled/b_sampled ∈ R^{B×C×1×N}（由 DSA 产生）
    - 输出：out_a/out_b ∈ R^{B×C×H×W}（增强分支，通常在外部以残差方式加回）
    """

    def __init__(
        self,
        *,
        d_model: int,
        nhead: int,
        attn_drop: float,
        proj_drop: float,
        use_pos_encoding: bool = False,
        pos_temperature: float = 10000.0,
    ) -> None:
        super().__init__()
        d_model = int(d_model)
        nhead = int(nhead)
        attn_drop = float(attn_drop)
        proj_drop = float(proj_drop)
        use_pos_encoding = bool(use_pos_encoding)
        pos_temperature = float(pos_temperature)

        if d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {d_model}")
        if nhead <= 0 or d_model % nhead != 0:
            raise ValueError(f"Invalid nhead/d_model: d_model={d_model} nhead={nhead}")
        if pos_temperature <= 0:
            raise ValueError(f"pos_temperature must be > 0, got {pos_temperature}")

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim ** -0.5
        self.use_pos_encoding = use_pos_encoding
        self.pos_temperature = pos_temperature

        self.proj_q_a = nn.Conv2d(d_model, d_model, kernel_size=1, stride=1, padding=0)
        self.proj_q_b = nn.Conv2d(d_model, d_model, kernel_size=1, stride=1, padding=0)
        self.proj_k_a = nn.Conv2d(d_model, d_model, kernel_size=1, stride=1, padding=0)
        self.proj_k_b = nn.Conv2d(d_model, d_model, kernel_size=1, stride=1, padding=0)
        self.proj_v_a = nn.Conv2d(d_model, d_model, kernel_size=1, stride=1, padding=0)
        self.proj_v_b = nn.Conv2d(d_model, d_model, kernel_size=1, stride=1, padding=0)
        self.proj_out_a = nn.Conv2d(d_model, d_model, kernel_size=1, stride=1, padding=0)
        self.proj_out_b = nn.Conv2d(d_model, d_model, kernel_size=1, stride=1, padding=0)

        self.attn_drop_a = nn.Dropout(attn_drop, inplace=True)
        self.attn_drop_b = nn.Dropout(attn_drop, inplace=True)
        self.proj_drop_a = nn.Dropout(proj_drop, inplace=True)
        self.proj_drop_b = nn.Dropout(proj_drop, inplace=True)

        # 用另一模态作为条件，对当前模态的 Query 做“模态自适应归一化/调制”。
        # 这一步有助于减小跨模态特征分布差异，使注意力更容易学习到对齐关系。
        self.norm_a = ModalityNorm(d_model, use_residual=True, learnable=True)
        self.norm_b = ModalityNorm(d_model, use_residual=True, learnable=True)

    def forward(
        self,
        x_a: torch.Tensor,
        x_b: torch.Tensor,
        *,
        a_sampled: torch.Tensor,
        b_sampled: torch.Tensor,
        kv_hw: tuple[int, int] | None = None,
        global_hw: tuple[int, int] | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x_a.shape != x_b.shape:
            raise ValueError(f"CCA expects same shapes, got a={x_a.shape} b={x_b.shape}")
        if x_a.ndim != 4:
            raise ValueError(f"CCA expects BCHW tensors, got {x_a.shape}")
        b, c, h, w = x_a.shape
        if c != self.d_model:
            raise ValueError(f"CCA expects C={self.d_model}, got {c}")
        if a_sampled.shape[:2] != (b, c) or b_sampled.shape[:2] != (b, c):
            raise ValueError(
                "CCA expects sampled features with shape (B,C,*,*), got "
                f"a_sampled={a_sampled.shape} b_sampled={b_sampled.shape}"
            )

        n_token = int(a_sampled.shape[-1])
        if int(b_sampled.shape[-1]) != n_token:
            raise ValueError(f"CCA expects same token length, got a={n_token} b={int(b_sampled.shape[-1])}")

        pos_q: torch.Tensor | None = None
        pos_kv: torch.Tensor | None = None
        if self.use_pos_encoding:
            if kv_hw is None:
                raise ValueError("CCA(use_pos_encoding=True) requires kv_hw=(Hk,Wk) for DSA tokens.")
            h_k, w_k = int(kv_hw[0]), int(kv_hw[1])
            if h_k <= 0 or w_k <= 0:
                raise ValueError(f"Invalid kv_hw={kv_hw} (expected >0).")
            pos_q = _build_2d_sincos_pos_embed(
                self.d_model, h, w, temperature=self.pos_temperature, device=x_a.device, dtype=x_a.dtype
            )
            pos_kv = _build_2d_sincos_pos_embed(
                self.d_model, h_k, w_k, temperature=self.pos_temperature, device=x_a.device, dtype=x_a.dtype
            )
            pos_kv = pos_kv.flatten(2).unsqueeze(2)  # (1, D, 1, Hk*Wk)
            if global_hw is not None:
                g_h, g_w = int(global_hw[0]), int(global_hw[1])
                if g_h <= 0 or g_w <= 0:
                    raise ValueError(f"Invalid global_hw={global_hw} (expected >0).")
                pos_global = _build_2d_sincos_pos_embed(
                    self.d_model, g_h, g_w, temperature=self.pos_temperature, device=x_a.device, dtype=x_a.dtype
                )
                pos_global = pos_global.flatten(2).unsqueeze(2)  # (1, D, 1, Va*Ha)
                pos_kv = torch.cat([pos_kv, pos_global], dim=-1)
            if int(pos_kv.shape[-1]) != n_token:
                raise ValueError(
                    "CCA positional encoding length mismatch: "
                    f"kv_hw={kv_hw} global_hw={global_hw} -> pos={int(pos_kv.shape[-1])}, "
                    f"but token_len={n_token}."
                )
        # 路径 A：用 x_a(引导) 调制 x_b，生成 q_b；与 a_sampled 的 k/v 做跨注意力，得到 out_a（回写到 x_a 分支）。
        # 维度约定：
        # - q_*: (B*nhead, head_dim, H*W)
        # - k_*/v_*: (B*nhead, head_dim, N)
        q_b = self.proj_q_b(self.norm_a(x_a, x_b))
        if pos_q is not None:
            q_b = q_b + pos_q
        q_b = q_b.view(b * self.nhead, self.head_dim, h * w)
        k_a = self.proj_k_a(a_sampled)
        if pos_kv is not None:
            k_a = k_a + pos_kv
        k_a = k_a.view(b * self.nhead, self.head_dim, n_token)
        v_a = self.proj_v_a(a_sampled).view(b * self.nhead, self.head_dim, n_token)

        # 路径 B：对称地，用 x_b(引导) 调制 x_a，生成 q_a；与 b_sampled 的 k/v 做跨注意力，得到 out_b。
        q_a = self.proj_q_a(self.norm_b(x_b, x_a))
        if pos_q is not None:
            q_a = q_a + pos_q
        q_a = q_a.view(b * self.nhead, self.head_dim, h * w)
        k_b = self.proj_k_b(b_sampled)
        if pos_kv is not None:
            k_b = k_b + pos_kv
        k_b = k_b.view(b * self.nhead, self.head_dim, n_token)
        v_b = self.proj_v_b(b_sampled).view(b * self.nhead, self.head_dim, n_token)

        # 注意力：attn_a = softmax(q_b^T k_a)，shape: (B*nhead, H*W, N)
        attn_a = torch.einsum("b c m, b c n -> b m n", q_b, k_a).mul(self.scale)
        attn_a = F.softmax(attn_a, dim=2)
        attn_a = self.attn_drop_a(attn_a)
        # out_a: (B*nhead, head_dim, H*W) -> (B, C, H, W)
        out_a = torch.einsum("b m n, b c n -> b c m", attn_a, v_a)
        out_a = out_a.view(b, c, h, w)
        out_a = self.proj_drop_a(self.proj_out_a(out_a))

        # 同理计算另一方向的注意力与输出 out_b。
        attn_b = torch.einsum("b c m, b c n -> b m n", q_a, k_b).mul(self.scale)
        attn_b = F.softmax(attn_b, dim=2)
        attn_b = self.attn_drop_b(attn_b)
        out_b = torch.einsum("b m n, b c n -> b c m", attn_b, v_b)
        out_b = out_b.view(b, c, h, w)
        out_b = self.proj_drop_b(self.proj_out_b(out_b))

        return out_a, out_b


class _DSACCABlock2D(nn.Module):
    """
    历史兼容包装：保持“单块 C2Former”接口不变，但内部拆分为 DSA + CCA。

    This keeps external behavior/config stable while internally splitting the logic into:
    - DSA: DeformableSamplingAligner2D
    - CCA: ComplementaryCrossAttention2D
    """

    def __init__(
        self,
        *,
        d_model: int,
        nhead: int,
        groups: int,
        cca_stride: int,
        offset_range_factor: float,
        no_offset: bool,
        attn_drop: float,
        proj_drop: float,
        offset_kernel_size: int,
        padding_mode: str,
        align_corners: bool,
        global_kv: bool = False,
        global_vert_anchors: int = 8,
        global_horz_anchors: int = 8,
        use_pos_encoding: bool = False,
        pos_temperature: float = 10000.0,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.global_kv = bool(global_kv)
        self.global_vert_anchors = int(global_vert_anchors)
        self.global_horz_anchors = int(global_horz_anchors)
        if self.global_kv:
            if self.global_vert_anchors <= 0 or self.global_horz_anchors <= 0:
                raise ValueError(
                    "global_vert_anchors/global_horz_anchors must be > 0 when global_kv=True, "
                    f"got vert={global_vert_anchors} horz={global_horz_anchors}"
                )
            self.global_pool = nn.AdaptiveAvgPool2d((self.global_vert_anchors, self.global_horz_anchors))
        else:
            self.global_pool = None
        self.dsa = DeformableSamplingAligner2D(
            d_model=d_model,
            groups=groups,
            cca_stride=cca_stride,
            offset_range_factor=offset_range_factor,
            no_offset=no_offset,
            offset_kernel_size=offset_kernel_size,
            padding_mode=padding_mode,
            align_corners=align_corners,
        )
        self.cca = ComplementaryCrossAttention2D(
            d_model=d_model,
            nhead=nhead,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            use_pos_encoding=use_pos_encoding,
            pos_temperature=pos_temperature,
        )

    def forward(self, x_a: torch.Tensor, x_b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x_a.shape != x_b.shape:
            raise ValueError(f"C2Former expects same shapes, got a={x_a.shape} b={x_b.shape}")
        if x_a.ndim != 4:
            raise ValueError(f"C2Former expects BCHW tensors, got {x_a.shape}")
        b, c, _, _ = x_a.shape
        if c != self.d_model:
            raise ValueError(f"C2Former expects C={self.d_model}, got {c}")

        # 先用 DSA 得到低分辨率采样 token，再用 CCA 做跨注意力输出增强特征。
        a_sampled, b_sampled = self.dsa(x_a, x_b)
        global_hw: tuple[int, int] | None = None
        if self.global_pool is not None:
            # Append global pooled tokens to K/V, enabling a "global memory" in addition to DSA sampled tokens.
            pooled_a = self.global_pool(x_a).flatten(2).view(b, c, 1, -1)
            pooled_b = self.global_pool(x_b).flatten(2).view(b, c, 1, -1)
            a_sampled = torch.cat([a_sampled, pooled_a], dim=-1)
            b_sampled = torch.cat([b_sampled, pooled_b], dim=-1)
            global_hw = (self.global_vert_anchors, self.global_horz_anchors)

        kv_hw: tuple[int, int] | None = None
        if self.cca.use_pos_encoding:
            # DSA predicts offsets on a stride=cca_stride grid, which yields Hk/Wk ~= ceil(H/stride).
            _, _, h, w = x_a.shape
            stride = int(self.dsa.cca_stride)
            h_k = (int(h) + stride - 1) // stride
            w_k = (int(w) + stride - 1) // stride
            kv_hw = (h_k, w_k)
        return self.cca(x_a, x_b, a_sampled=a_sampled, b_sampled=b_sampled, kv_hw=kv_hw, global_hw=global_hw)


# 兼容旧命名：历史代码里可能仍引用 `_DSA_CCABlock2D` / `_C2FormerBlock2D`。
_DSA_CCABlock2D = _DSACCABlock2D
_C2FormerBlock2D = _DSACCABlock2D


class ASMFusion2D(nn.Module):
    """
    ASMF (Adaptive Sampling Multispectral Fusion) 2D 融合模块。

    该模块内部仍是 “DSA + CCA” 的实现方式：
    - DSA: DeformableSamplingAligner2D（对齐 + 稀疏采样 K/V token）
    - CCA: ComplementaryCrossAttention2D（密集 Q 关注稀疏 K/V，输出仍为 HxW）

    输入：rgb/ms 两路同尺寸特征（B,C,H,W）
    输出：两路被互补增强后的特征（B,C,H,W）

    关键点：
    - 支持 in_channels != d_model：用 1x1 Conv 做投影（进入融合用 d_model，融合后再投回 in_channels）。
    - `offset_on` 用于控制“offset 作用在哪一路”（即 DSA 中哪一路使用 ref+offset 采样）。
      - offset_on="ms"：认为 MS 相对 RGB 存在位移，采样 MS 时使用 offset；注意最终是“交叉加回”。
      - offset_on="rgb"：相反方向。
    """

    def __init__(
        self,
        *,
        in_channels: int,
        d_model: int | None = None,
        nhead: int = 8,
        groups: int = 1,
        cca_stride: int = 3,
        offset_range_factor: float = 2.0,
        no_offset: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        offset_kernel_size: int = 3,
        padding_mode: str = "zeros",
        align_corners: bool = True,
        offset_on: str = "ms",
        global_kv: bool = False,
        global_vert_anchors: int = 8,
        global_horz_anchors: int = 8,
        use_pos_encoding: bool = False,
        pos_temperature: float = 10000.0,
    ) -> None:
        super().__init__()
        in_channels = int(in_channels)
        if d_model is None:
            d_model = in_channels
        d_model = int(d_model)
        offset_on = str(offset_on).strip().lower()

        if in_channels <= 0:
            raise ValueError(f"in_channels must be > 0, got {in_channels}")
        if d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {d_model}")
        if offset_on not in {"rgb", "ms"}:
            raise ValueError(f"Unsupported offset_on={offset_on} (expected rgb/ms)")

        self.in_channels = in_channels
        self.d_model = d_model
        self.offset_on = offset_on

        if d_model == in_channels:
            self.rgb_in_proj = nn.Identity()
            self.ms_in_proj = nn.Identity()
            self.rgb_out_proj = nn.Identity()
            self.ms_out_proj = nn.Identity()
        else:
            self.rgb_in_proj = nn.Sequential(nn.Conv2d(in_channels, d_model, kernel_size=1), nn.ReLU(inplace=True))
            self.ms_in_proj = nn.Sequential(nn.Conv2d(in_channels, d_model, kernel_size=1), nn.ReLU(inplace=True))
            self.rgb_out_proj = nn.Sequential(nn.Conv2d(d_model, in_channels, kernel_size=1), nn.ReLU(inplace=True))
            self.ms_out_proj = nn.Sequential(nn.Conv2d(d_model, in_channels, kernel_size=1), nn.ReLU(inplace=True))

        self.block = _C2FormerBlock2D(
            d_model=d_model,
            nhead=nhead,
            groups=groups,
            cca_stride=cca_stride,
            offset_range_factor=offset_range_factor,
            no_offset=no_offset,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            offset_kernel_size=offset_kernel_size,
            padding_mode=padding_mode,
            align_corners=align_corners,
            global_kv=global_kv,
            global_vert_anchors=global_vert_anchors,
            global_horz_anchors=global_horz_anchors,
            use_pos_encoding=use_pos_encoding,
            pos_temperature=pos_temperature,
        )

    def forward(self, rgb: torch.Tensor, ms: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if rgb.shape != ms.shape:
            raise ValueError(f"ASMFusion2D expects same shapes, got rgb={rgb.shape} ms={ms.shape}")
        if rgb.ndim != 4:
            raise ValueError(f"ASMFusion2D expects BCHW tensors, got {rgb.shape}")
        b, c, _, _ = rgb.shape
        if c != self.in_channels:
            raise ValueError(f"ASMFusion2D expects C={self.in_channels}, got {c}")

        if self.offset_on == "ms":
            # 情况 1：offset 作用在 MS 路（即 DSA 的第一个输入是 ms_proj）。
            # block(ms_proj, rgb_proj) 会返回 (out_ms, out_rgb)，然后“交叉残差”加回：
            #   ms <- ms + out_rgb
            #   rgb <- rgb + out_ms
            ms_proj = self.ms_in_proj(ms)
            rgb_proj = self.rgb_in_proj(rgb)
            out_ms, out_rgb = self.block(ms_proj, rgb_proj)
            out_ms = self.ms_out_proj(out_ms)
            out_rgb = self.rgb_out_proj(out_rgb)
            ms = ms + out_rgb
            rgb = rgb + out_ms
        else:
            # 情况 2：offset 作用在 RGB 路（DSA 的第一个输入是 rgb_proj）。
            rgb_proj = self.rgb_in_proj(rgb)
            ms_proj = self.ms_in_proj(ms)
            out_rgb, out_ms = self.block(rgb_proj, ms_proj)
            out_rgb = self.rgb_out_proj(out_rgb)
            out_ms = self.ms_out_proj(out_ms)
            rgb = rgb + out_ms
            ms = ms + out_rgb

        return rgb, ms


# 兼容旧命名：外部/旧配置可能仍 import/引用这些名字（不影响配置与训练）。
DSACCAFusion2D = ASMFusion2D
C2FormerFusion2D = ASMFusion2D

__all__ = [
    "ASMFusion2D",
    "DSACCAFusion2D",
    "C2FormerFusion2D",
    "DeformableSamplingAligner2D",
    "ComplementaryCrossAttention2D",
]
