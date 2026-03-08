# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------

from typing import Callable, Dict, List

import torch
from torch import nn

from utils.misc import NestedTensor
from engines.models.msifdetr.common.position_encoding import build_position_encoding
from .backbone import Backbone
from .msi_fusion_backbone import MsiFusionConfig, MsiP4FusionJoiner

class Joiner(nn.Sequential):
    # 将 backbone 与位置编码器串成一个 nn.Sequential，方便统一调用与导出。
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self._export = False

    def forward(self, tensor_list):
        """返回骨干特征以及对应的位置编码列表。"""
        x = self[0](tensor_list)
        pos = []
        for x_ in x:
            pos.append(self[1](x_, align_dim_orders=False).to(x_.tensors.dtype))
        return x, pos

    def export(self):
        # 切换到导出模式，递归触发子模块的导出方法。
        self._export = True
        self._forward_origin = self.forward
        self.forward = self.forward_export
        for name, m in self.named_modules():
            if (
                hasattr(m, "export")
                and isinstance(m.export, Callable)
                and hasattr(m, "_export")
                and not m._export
            ):
                m.export()

    def forward_export(self, inputs):
        # 导出时直接接受纯张量输入（或 dict），返回特征、占位 mask 和位置编码。
        feats, masks = self[0](inputs)
        poss = []
        for feat, mask in zip(feats, masks):
            poss.append(self[1](mask, align_dim_orders=False).to(feat.dtype))
        return feats, None, poss


def build_backbone(
    encoder,
    vit_encoder_num_layers,
    pretrained_encoder,
    window_block_indexes,
    drop_path,
    out_channels,
    out_feature_indexes,
    projector_scale,
    use_cls_token,
    hidden_dim,
    position_embedding,
    freeze_encoder,
    layer_norm,
    target_shape,
    rms_norm,
    backbone_lora,
    force_no_pretrain,
    gradient_checkpointing,
    load_dinov2_weights,
    patch_size,
    num_windows,
    positional_encoding_size,
    input_channels: int = 3,
    # MSI 融合（P4 单点融合 + 门控）
    msi_fusion: bool = False,
    # Decoder 侧融合（Query 级语义对齐）需要返回 ms 特征
    decoder_ms_fusion: bool = False,
    msi_backbone: str = "specat_lite",
    ms_input_channels: int = 7,
    msi_pre_downsample_to_p4: bool = False,
    specat_base_dim: int = 64,
    specat_stage: int = 3,
    specat_num_blocks=None,
    specat_dim_head=None,
    specat_attention_type: str = "base",
    film_reduction: int = 4,
    film_spatial: bool = False,
    film_kernel_size: int = 3,
    fusion_alpha_init: float = 0.0,
    fusion_alpha_scale: float = 1.0,
    freeze_fusion_alpha: bool = False,
    freeze_msi_backbone: bool = False,
    use_ms_mask: bool = False,
):
    """
    Useful args:
        - encoder: encoder name
        - lr_encoder:
        - dilation
        - use_checkpoint: for swin only for now

    """
    # 先构建位置编码，再实例化 ViT backbone。
    position_embedding = build_position_encoding(hidden_dim, position_embedding)

    # Default: RF-DETR ViT backbone + projector（单流/堆叠输入）。
    rgb_backbone = Backbone(
        encoder,
        pretrained_encoder,
        window_block_indexes=window_block_indexes,
        drop_path=drop_path,
        out_channels=out_channels,
        out_feature_indexes=out_feature_indexes,
        projector_scale=projector_scale,
        use_cls_token=use_cls_token,
        layer_norm=layer_norm,
        freeze_encoder=freeze_encoder,
        target_shape=target_shape,
        rms_norm=rms_norm,
        backbone_lora=backbone_lora,
        gradient_checkpointing=gradient_checkpointing,
        load_dinov2_weights=load_dinov2_weights,
        patch_size=patch_size,
        num_windows=num_windows,
        positional_encoding_size=positional_encoding_size,
        input_channels=input_channels,
    )

    ms_enabled = bool(msi_fusion) or bool(decoder_ms_fusion)
    if not ms_enabled:
        # Joiner 同时输出特征与位置编码，便于 Transformer 头直接消费。
        return Joiner(rgb_backbone, position_embedding)

    # 双模态融合版 Joiner：输入可为 dict{"rgb","ms"}，仅在 P4 做弱耦合融合。
    cfg = MsiFusionConfig(
        enabled=True,
        p4_fusion_enabled=bool(msi_fusion),
        return_ms_features=bool(decoder_ms_fusion),
        msi_backbone=str(msi_backbone),
        ms_input_channels=int(ms_input_channels),
        pre_downsample_to_p4=bool(msi_pre_downsample_to_p4),
        specat_base_dim=int(specat_base_dim),
        specat_stage=int(specat_stage),
        specat_num_blocks=tuple(int(v) for v in (specat_num_blocks or (2, 2, 2, 2))),
        specat_dim_head=int(specat_dim_head) if specat_dim_head is not None else None,
        specat_attention_type=str(specat_attention_type),
        film_reduction=int(film_reduction),
        film_spatial=bool(film_spatial),
        film_kernel_size=int(film_kernel_size),
        fusion_alpha_init=float(fusion_alpha_init),
        fusion_alpha_scale=float(fusion_alpha_scale),
        freeze_fusion_alpha=bool(freeze_fusion_alpha),
        freeze_msi_backbone=bool(freeze_msi_backbone),
        use_ms_mask=bool(use_ms_mask),
    )
    return MsiP4FusionJoiner(
        rgb_backbone=rgb_backbone,
        position_embedding=position_embedding,
        projector_scale=projector_scale,
        hidden_dim=hidden_dim,
        cfg=cfg,
    )
