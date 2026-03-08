# ------------------------------------------------------------------------
# MSI Fusion Backbone (P4-only, gated)
# ------------------------------------------------------------------------
# 目标：在不破坏 RGB(DINOv2) 预训练优势的前提下，引入 MSI 分支并仅在 P4 做弱耦合融合。
# 设计要点：
# - RGB 分支保持 3 通道，复用现有 Backbone(DINOv2 + projector)。
# - MSI 分支使用纯 PyTorch 的轻量“SPECAT-like”编码器（避免额外依赖）。
# - 融合采用 FiLM/SE 风格调制，并使用 alpha 门控（alpha 初始化为 0，训练初期等价 RGB-only）。

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from utils.misc import NestedTensor

from .specat_encoder import SpecatEncoderBackbone, SpecatEncoderConfig


class ScalarGate(nn.Module):
    """
    标量门控参数容器。

    注意：BaseTrainer 会识别名称中包含 ".fusion_alpha." 的参数并禁用 weight decay；
    因此把参数命名为 fusion_alpha.value，确保能命中该规则。
    """

    def __init__(self, init: float = 0.0):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(float(init)))

    def forward(self) -> Tensor:
        return self.value


class ConvGnAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        gn_groups: int = 8,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        # GN 对 batch size 更友好；groups 需整除通道数
        gn_groups = max(1, min(int(gn_groups), int(out_channels)))
        while out_channels % gn_groups != 0 and gn_groups > 1:
            gn_groups -= 1
        self.norm = nn.GroupNorm(gn_groups, out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.norm(self.conv(x)))


class SpecatLiteP4Backbone(nn.Module):
    """
    轻量 MSI 编码器（SPECAT-like），输出 stride≈16 的 P4 特征。

    说明：
    - 不依赖 third_party/SPECAT 的 timm/einops，便于在当前环境直接跑通；
    - 通过逐级下采样 + 通道注意力（SE 风格）增强“光谱通道的有效性”；
    - 输出通道为 hidden_dim，便于直接与 DETR 的 P4 对齐。
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        *,
        base_dim: int = 64,
        se_reduction: int = 8,
    ):
        super().__init__()
        d1 = int(base_dim)
        d2 = int(base_dim * 2)
        d3 = int(base_dim * 4)

        # 4 次 stride=2 下采样 -> stride=16
        self.stem = ConvGnAct(in_channels, d1, kernel_size=3, stride=2, gn_groups=8)
        self.stage2 = ConvGnAct(d1, d2, kernel_size=3, stride=2, gn_groups=8)
        self.stage3 = ConvGnAct(d2, d3, kernel_size=3, stride=2, gn_groups=8)
        self.stage4 = ConvGnAct(d3, d3, kernel_size=3, stride=2, gn_groups=8)

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(d3, max(1, d3 // se_reduction), kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(1, d3 // se_reduction), d3, kernel_size=1),
            nn.Sigmoid(),
        )

        self.proj = nn.Sequential(
            nn.Conv2d(d3, hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(8 if hidden_dim % 8 == 0 else 1, hidden_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = x * self.se(x)
        return self.proj(x)


class P4FiLMFusion(nn.Module):
    """
    P4 单点融合（FiLM/SE 调制 + alpha 门控）。

    输出：融合后的 rgb_p4，形状不变。
    """

    def __init__(
        self,
        hidden_dim: int,
        *,
        reduction: int = 4,
        alpha_init: float = 0.0,
        alpha_scale: float = 1.0,
        spatial: bool = False,
        kernel_size: int = 3,
    ):
        super().__init__()
        mid = max(1, int(hidden_dim // reduction))
        self.spatial = bool(spatial)
        k = int(kernel_size)
        if k <= 0 or k % 2 == 0:
            raise ValueError(f"P4FiLMFusion.kernel_size 需要为正奇数，但得到 {kernel_size}")
        if self.spatial:
            self.to_gamma_beta = nn.Sequential(
                nn.Conv2d(hidden_dim, mid, kernel_size=k, padding=k // 2),
                nn.SiLU(inplace=True),
                nn.Conv2d(mid, hidden_dim * 2, kernel_size=1),
            )
        else:
            self.to_gamma_beta = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(hidden_dim, mid, kernel_size=1),
                nn.SiLU(inplace=True),
                nn.Conv2d(mid, hidden_dim * 2, kernel_size=1),
            )
        self.fusion_alpha = ScalarGate(alpha_init)
        self.alpha_scale = float(alpha_scale)

    def forward(self, rgb_p4: Tensor, ms_p4: Tensor) -> Tensor:
        if rgb_p4.shape != ms_p4.shape:
            raise ValueError(
                f"P4FiLMFusion 需要 rgb/ms 形状一致，但收到 rgb={tuple(rgb_p4.shape)} ms={tuple(ms_p4.shape)}"
            )
        gamma_beta = self.to_gamma_beta(ms_p4)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        # 限幅，避免调制过强导致训练不稳定；
        # alpha 初始化为 0，训练初期等价 RGB-only，但仍建议对 alpha 做 tanh 以限制范围。
        alpha = (self.alpha_scale * torch.tanh(self.fusion_alpha())).to(rgb_p4.dtype)
        gamma = torch.tanh(gamma)
        beta = torch.tanh(beta)
        return rgb_p4 * (1.0 + alpha * gamma) + alpha * beta


@dataclass(frozen=True)
class MsiFusionConfig:
    # 是否启用 MSI 分支（用于生成 ms 特征或做 P4 融合）
    enabled: bool = False
    # 是否在 backbone 侧做 P4 融合（FiLM/SE + alpha gate）
    p4_fusion_enabled: bool = False
    # 是否在 forward 中返回 ms 特征（供 decoder 侧融合使用）
    return_ms_features: bool = False
    msi_backbone: str = "specat_lite"  # specat_lite | specat_encoder
    ms_input_channels: int = 7
    # 将 MSI 输入预先下采样到 RGB-P4 的空间尺寸，再送入 MSI backbone。
    # 目的：显著降低 MSI 分支的计算/显存开销，并减少残余平移带来的逐像素噪声。
    pre_downsample_to_p4: bool = False
    specat_base_dim: int = 64
    specat_stage: int = 3
    specat_num_blocks: Tuple[int, ...] = (2, 2, 2, 2)
    specat_dim_head: Optional[int] = None
    specat_attention_type: str = "base"
    film_reduction: int = 4
    film_spatial: bool = False
    film_kernel_size: int = 3
    fusion_alpha_init: float = 0.0
    fusion_alpha_scale: float = 1.0
    freeze_fusion_alpha: bool = False
    freeze_msi_backbone: bool = False
    use_ms_mask: bool = False


class MsiP4FusionJoiner(nn.Sequential):
    """
    Joiner：RGB backbone +（可选）MSI backbone +（可选）P4 融合 + 位置编码。

    兼容输入：
    - 单流：NestedTensor
    - 堆叠：NestedTensor（按 [RGB, MSI] 通道拼接；shape=(B, rgb_ch+ms_ch, H, W)）
    - 双流：{"rgb": NestedTensor, "ms": NestedTensor}
    """

    def __init__(
        self,
        *,
        rgb_backbone: nn.Module,
        position_embedding: nn.Module,
        projector_scale: List[str],
        hidden_dim: int,
        cfg: MsiFusionConfig,
    ):
        # 关键：使用 nn.Sequential(rgb_backbone, position_embedding) 保持与原 Joiner 一致的参数命名：
        # - RGB backbone 参数路径仍为 backbone.0.*
        # - pos embedding 参数路径仍为 backbone.1.*
        # 这样可以最大化复用既有预训练权重与优化器分组逻辑。
        super().__init__(rgb_backbone, position_embedding)
        self.projector_scale = list(projector_scale)
        self.hidden_dim = int(hidden_dim)
        self.cfg = cfg
        try:
            self.rgb_input_channels = int(getattr(getattr(rgb_backbone, "encoder", None), "input_channels", 3))
        except Exception:
            self.rgb_input_channels = 3

        self._export = False

        self._p4_index = self.projector_scale.index("P4") if "P4" in self.projector_scale else 0

        if cfg.enabled:
            if str(cfg.msi_backbone).lower() in {"specat_encoder", "specat"}:
                self.msi_backbone = SpecatEncoderBackbone(
                    SpecatEncoderConfig(
                        in_channels=int(cfg.ms_input_channels),
                        base_dim=int(cfg.specat_base_dim),
                        stage=int(cfg.specat_stage),
                        num_blocks=tuple(int(v) for v in cfg.specat_num_blocks),
                        dim_head=int(cfg.specat_dim_head) if cfg.specat_dim_head is not None else None,
                        attention_type=str(cfg.specat_attention_type).lower(),  # type: ignore[arg-type]
                        out_channels=self.hidden_dim,
                    )
                )
            elif str(cfg.msi_backbone).lower() in {"specat_lite", "lite"}:
                self.msi_backbone = SpecatLiteP4Backbone(
                    in_channels=int(cfg.ms_input_channels),
                    hidden_dim=self.hidden_dim,
                    base_dim=int(cfg.specat_base_dim),
                )
            else:
                raise ValueError(
                    f"未知的 msi_backbone={cfg.msi_backbone}，支持 specat_lite/specat_encoder"
                )
            if bool(cfg.p4_fusion_enabled):
                self.p4_fusion = P4FiLMFusion(
                    hidden_dim=self.hidden_dim,
                    reduction=int(cfg.film_reduction),
                    alpha_init=float(cfg.fusion_alpha_init),
                    alpha_scale=float(cfg.fusion_alpha_scale),
                    spatial=bool(cfg.film_spatial),
                    kernel_size=int(cfg.film_kernel_size),
                )
                if bool(cfg.freeze_fusion_alpha):
                    self.p4_fusion.fusion_alpha.value.requires_grad = False
            else:
                self.p4_fusion = None
            if bool(cfg.freeze_msi_backbone):
                for p in self.msi_backbone.parameters():
                    p.requires_grad = False
        else:
            self.msi_backbone = None
            self.p4_fusion = None

    def export(self):
        """
        ONNX 导出模式：
        - 将 forward 切换为 forward_export（输入为裸 Tensor，而不是 NestedTensor/dict）
        - 递归触发子模块的 export（Backbone/PositionEmbedding 等）
        """
        self._export = True
        self._forward_origin = self.forward
        self.forward = self.forward_export
        for _, m in self.named_modules():
            if (
                hasattr(m, "export")
                and isinstance(getattr(m, "export"), Callable)
                and hasattr(m, "_export")
                and not getattr(m, "_export")
            ):
                m.export()

    def _split_stacked_export_inputs(self, rgb: Tensor) -> tuple[Tensor, Optional[Tensor]]:
        if not self.cfg.enabled:
            return rgb, None
        if rgb.ndim != 4:
            return rgb, None
        ms_ch = int(getattr(self.cfg, "ms_input_channels", 0))
        rgb_ch = int(getattr(self, "rgb_input_channels", 3))
        if ms_ch <= 0 or rgb_ch <= 0:
            return rgb, None
        if int(rgb.shape[1]) != rgb_ch + ms_ch:
            return rgb, None
        return rgb[:, :rgb_ch, ...], rgb[:, rgb_ch : rgb_ch + ms_ch, ...]

    def _split_stacked_nested_tensor(self, inputs: NestedTensor) -> Optional[Dict[str, NestedTensor]]:
        if not self.cfg.enabled:
            return None
        tensors, mask = inputs.decompose()
        if tensors.ndim != 4:
            return None
        ms_ch = int(getattr(self.cfg, "ms_input_channels", 0))
        rgb_ch = int(getattr(self, "rgb_input_channels", 3))
        if ms_ch <= 0 or rgb_ch <= 0:
            return None
        if int(tensors.shape[1]) != rgb_ch + ms_ch:
            return None
        rgb = tensors[:, :rgb_ch, ...]
        ms = tensors[:, rgb_ch : rgb_ch + ms_ch, ...]
        return {"rgb": NestedTensor(rgb, mask), "ms": NestedTensor(ms, mask)}

    def forward_export(self, rgb: Tensor, ms: Optional[Tensor] = None):
        """
        导出模式的 Joiner 前向：
        - 输入：rgb (B,3,H,W)，可选 ms (B,ms_ch,H,W)
        - 兼容堆叠输入：当 ms=None 且 rgb 的通道数为 rgb_ch+ms_ch 时，自动拆分为 rgb/ms
        - 输出：rgb_feats(list[Tensor]), None(占位), poss(list[Tensor])

        注意：mask 在导出中默认为全 False（无 padding），用于 position embedding 的计算。
        """
        if ms is None:
            rgb, ms = self._split_stacked_export_inputs(rgb)

        rgb_out = self[0](rgb)
        if not (isinstance(rgb_out, (tuple, list)) and len(rgb_out) == 2):
            raise TypeError(
                "Backbone.export() 后应返回 (feats, masks)，请确认已调用 model.export() 且 backbone 支持导出。"
            )
        rgb_feats, rgb_masks = rgb_out

        # 构建 MSI 特征（可选）
        ms_feats = None
        if self.cfg.enabled and ms is not None:
            if self.msi_backbone is None:
                raise ValueError("cfg.enabled=True 但 msi_backbone 未构建。")
            p4_index = int(self._p4_index)
            p4_index = 0 if p4_index < 0 or p4_index >= len(rgb_feats) else p4_index
            rgb_p4 = rgb_feats[p4_index]

            ms_input = ms
            if bool(self.cfg.pre_downsample_to_p4) and ms_input.shape[-2:] != rgb_p4.shape[-2:]:
                ms_input = F.interpolate(ms_input, size=rgb_p4.shape[-2:], mode="bilinear", align_corners=False)

            ms_feat_base = self.msi_backbone(ms_input)
            ms_feats = []
            for rgb_feat in rgb_feats:
                target_hw = rgb_feat.shape[-2:]
                ms_feat_i = ms_feat_base
                if ms_feat_i.shape[-2:] != target_hw:
                    ms_feat_i = F.interpolate(ms_feat_i, size=target_hw, mode="bilinear", align_corners=False)
                ms_feats.append(ms_feat_i)

            # P4 融合（可选）
            if self.p4_fusion is not None:
                ms_p4 = ms_feats[p4_index]
                fused_p4 = self.p4_fusion(rgb_p4, ms_p4)
                rgb_feats = list(rgb_feats)
                rgb_feats[p4_index] = fused_p4

        # 位置编码：依赖 mask（导出时 mask 全 False，来自 backbone.export 的 out_masks）
        poss = []
        for mask in rgb_masks:
            poss.append(self[1](mask, align_dim_orders=False))

        if self.cfg.return_ms_features:
            if ms_feats is None:
                raise ValueError("cfg.return_ms_features=True 需要 MSI 输入，但 ms=None。")
            ms_poss = []
            # 复用 rgb_masks（导出时无 padding，mask 全 False）
            for mask in rgb_masks:
                ms_poss.append(self[1](mask, align_dim_orders=False))
            return rgb_feats, None, poss, ms_feats, None, ms_poss

        return rgb_feats, None, poss

    def forward(self, inputs):
        if not isinstance(inputs, dict) and isinstance(inputs, NestedTensor):
            split = self._split_stacked_nested_tensor(inputs)
            if split is not None:
                inputs = split

        if isinstance(inputs, dict):
            rgb_in: Optional[NestedTensor] = inputs.get("rgb")
            ms_in: Optional[NestedTensor] = inputs.get("ms")
            if rgb_in is None:
                raise ValueError("MsiP4FusionJoiner 需要输入 dict 中包含 'rgb'。")
            feats = self[0](rgb_in)
            ms_feats = None
            if self.cfg.enabled and ms_in is not None:
                ms_feats = self._build_ms_feats(feats, rgb_in=rgb_in, ms_in=ms_in)
                if self.p4_fusion is not None:
                    feats = self._fuse_p4(feats, ms_feats=ms_feats, rgb_in=rgb_in, ms_in=ms_in)
        else:
            feats = self[0](inputs)
            ms_feats = None

        pos = []
        for feat in feats:
            pos.append(self[1](feat, align_dim_orders=False).to(feat.tensors.dtype))
        if self.cfg.return_ms_features:
            if ms_feats is None:
                raise ValueError("decoder_ms_fusion 需要 ms 特征，但当前未提供 'ms' 输入或未启用 MSI 分支。")
            ms_pos = []
            for feat in ms_feats:
                ms_pos.append(self[1](feat, align_dim_orders=False).to(feat.tensors.dtype))
            return feats, pos, ms_feats, ms_pos
        return feats, pos

    def _fuse_p4(
        self,
        feats: List[NestedTensor],
        *,
        ms_feats: List[NestedTensor],
        rgb_in: NestedTensor,
        ms_in: NestedTensor,
    ) -> List[NestedTensor]:
        if self.msi_backbone is None or self.p4_fusion is None:
            return feats

        p4_index = int(self._p4_index)
        if p4_index < 0 or p4_index >= len(feats):
            raise IndexError(f"P4 index={p4_index} 超出特征列表长度 {len(feats)}。")

        rgb_p4, rgb_mask = feats[p4_index].decompose()
        if rgb_mask is None:
            raise ValueError("RGB P4 mask 不能为空。")
        ms_p4, _ = ms_feats[p4_index].decompose()
        fused_p4 = self.p4_fusion(rgb_p4, ms_p4)

        out_mask = rgb_mask
        if self.cfg.use_ms_mask:
            ms_mask = ms_in.mask
            if ms_mask is not None:
                ms_mask_p4 = F.interpolate(ms_mask[None].float(), size=rgb_p4.shape[-2:]).to(torch.bool)[0]
                # mask=True 表示 padding；valid = ~mask；valid_out = valid_rgb & valid_ms -> mask_out = mask_rgb | mask_ms
                out_mask = out_mask | ms_mask_p4

        fused = NestedTensor(fused_p4, out_mask)
        new_feats = list(feats)
        new_feats[p4_index] = fused
        return new_feats

    def _build_ms_feats(
        self,
        rgb_feats: List[NestedTensor],
        *,
        rgb_in: NestedTensor,
        ms_in: NestedTensor,
    ) -> List[NestedTensor]:
        if self.msi_backbone is None:
            raise ValueError("未启用 MSI backbone，无法构建 ms_feats。")
        if not rgb_feats:
            raise ValueError("rgb_feats 为空，无法构建 ms_feats。")

        p4_index = int(self._p4_index)
        if p4_index < 0 or p4_index >= len(rgb_feats):
            p4_index = 0

        rgb_p4, rgb_mask = rgb_feats[p4_index].decompose()
        if rgb_mask is None:
            raise ValueError("RGB P4 mask 不能为空。")

        ms_input = ms_in.tensors
        if bool(self.cfg.pre_downsample_to_p4) and ms_input.shape[-2:] != rgb_p4.shape[-2:]:
            ms_input = F.interpolate(ms_input, size=rgb_p4.shape[-2:], mode="bilinear", align_corners=False)

        ms_feat_base = self.msi_backbone(ms_input)

        ms_feats: List[NestedTensor] = []
        for rgb_nt in rgb_feats:
            rgb_feat, rgb_mask_i = rgb_nt.decompose()
            if rgb_mask_i is None:
                raise ValueError("RGB mask 不能为空。")
            target_hw = rgb_feat.shape[-2:]
            ms_feat_i = ms_feat_base
            if ms_feat_i.shape[-2:] != target_hw:
                ms_feat_i = F.interpolate(ms_feat_i, size=target_hw, mode="bilinear", align_corners=False)

            out_mask = rgb_mask_i
            if self.cfg.use_ms_mask:
                ms_mask = ms_in.mask
                if ms_mask is not None:
                    ms_mask_i = F.interpolate(ms_mask[None].float(), size=target_hw).to(torch.bool)[0]
                    out_mask = out_mask | ms_mask_i
            ms_feats.append(NestedTensor(ms_feat_i, out_mask))
        return ms_feats
