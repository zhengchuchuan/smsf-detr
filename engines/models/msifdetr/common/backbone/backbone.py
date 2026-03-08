# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Backbone modules.
"""
from functools import partial
import torch
import torch.nn.functional as F
from torch import nn

from transformers import AutoModel, AutoProcessor, AutoModelForCausalLM, AutoConfig, AutoBackbone
from peft import LoraConfig, get_peft_model, PeftModel

from utils.misc import NestedTensor, is_main_process

from .base import BackboneBase
from .projector import MultiScaleProjector
from .dinov2 import DinoV2

__all__ = ["Backbone"]


class Backbone(BackboneBase):
    """backbone."""
    def __init__(self,
                 name: str,
                 pretrained_encoder: str=None,
                 window_block_indexes: list=None,
                 drop_path=0.0,
                 out_channels=256,
                 out_feature_indexes: list=None,
                 projector_scale: list=None,
                 use_cls_token: bool = False,
                 freeze_encoder: bool = False,
                 layer_norm: bool = False,
                 target_shape: tuple[int, int] = (640, 640),
                 rms_norm: bool = False,
                 backbone_lora: bool = False,
                 gradient_checkpointing: bool = False,
                 load_dinov2_weights: bool = True,
                 patch_size: int = 14,
                 num_windows: int = 4,
                 positional_encoding_size: bool = False,
                 input_channels: int = 3,
                 ):
        super().__init__()
        # name 约定: dinov2[_registers][_windowed]_{size}，从名称中拆解是否启用 register token 与窗口注意力。
        name_parts = name.split("_")
        assert name_parts[0] == "dinov2"
        size = name_parts[-1]
        use_registers = False
        if "registers" in name_parts:
            use_registers = True
            name_parts.remove("registers")
        use_windowed_attn = False
        if "windowed" in name_parts:
            use_windowed_attn = True
            name_parts.remove("windowed")
        assert len(name_parts) == 2, "name should be dinov2, then either registers, windowed, both, or none, then the size"
        self.encoder = DinoV2(
            size=name_parts[-1],
            out_feature_indexes=out_feature_indexes,
            shape=target_shape,
            use_registers=use_registers,
            use_windowed_attn=use_windowed_attn,
            gradient_checkpointing=gradient_checkpointing,
            load_dinov2_weights=load_dinov2_weights,
            patch_size=patch_size,
            num_windows=num_windows,
            positional_encoding_size=positional_encoding_size,
            num_input_channels=input_channels,
        )
        # 仅冻结 encoder，projector 仍然参与训练。
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.projector_scale = projector_scale
        assert len(self.projector_scale) > 0
        # x[0]
        assert (
            sorted(self.projector_scale) == self.projector_scale
        ), "only support projector scale P2/P3/P4/P5/P6 in ascending order."
        # 特征金字塔各层对应的缩放系数
        # 约定：P4 为基准尺度（1.0）；P2/P3 上采样，P5/P6 下采样
        level2scalefactor = dict(P2=4.0, P3=2.0, P4=1.0, P5=0.5, P6=0.25)
        scale_factors = [level2scalefactor[lvl] for lvl in self.projector_scale]

        self.projector = MultiScaleProjector(
            in_channels=self.encoder._out_feature_channels,
            out_channels=out_channels,
            scale_factors=scale_factors,
            layer_norm=layer_norm,
            rms_norm=rms_norm,
        )

        # 记录各尺度输出通道，便于双流融合时对齐。
        self.out_channels = [out_channels for _ in self.projector_scale]
        self._export = False

    def export(self):
        # 导出模式：固定 forward 并在使用 LoRA 时合并权重。
        self._export = True
        self._forward_origin = self.forward
        self.forward = self.forward_export

        if isinstance(self.encoder, PeftModel):
            print("Merging and unloading LoRA weights")
            self.encoder.merge_and_unload()

    def forward(self, tensor_list: NestedTensor):
        """ """
        # encoder 输出多尺度特征，projector 做尺度对齐，最后补齐对应 mask。
        feats = self.encoder(tensor_list.tensors)
        feats = self.projector(feats)
        # x: [(B, C, H, W)]
        out = []
        for feat in feats:
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=feat.shape[-2:]).to(torch.bool)[
                0
            ]
            out.append(NestedTensor(feat, mask))
        return out

    def forward_export(self, tensors: torch.Tensor):
        feats = self.encoder(tensors)
        feats = self.projector(feats)
        out_feats = []
        out_masks = []
        for feat in feats:
            # x: [(B, C, H, W)]
            b, _, h, w = feat.shape
            out_masks.append(
                torch.zeros((b, h, w), dtype=torch.bool, device=feat.device)
            )
            out_feats.append(feat)
        return out_feats, out_masks

    def get_named_param_lr_pairs(self, args, prefix: str = "backbone.0"):
        # 为 ViT 各层设置分层 lr/weight decay，供优化器构造参数组。
        num_layers = args.out_feature_indexes[-1] + 1
        # dual-stream 场景下 prefix 可能是 "backbone.0.rgb_backbone"；
        # 必须使用 prefix 推导关键前缀，否则会导致 RGB 分支未被纳入分层 lr/weight decay 参数组。
        backbone_key = f"{prefix}.encoder"
        named_param_lr_pairs = {}
        for n, p in self.named_parameters():
            n = prefix + "." + n
            if backbone_key in n and p.requires_grad:
                lr = (
                    args.lr_encoder
                    * get_dinov2_lr_decay_rate(
                        n,
                        lr_decay_rate=args.lr_vit_layer_decay,
                        num_layers=num_layers,
                    )
                    * args.lr_component_decay**2
                )
                wd = args.weight_decay * get_dinov2_weight_decay_rate(n)
                named_param_lr_pairs[n] = {
                    "params": p,
                    "lr": lr,
                    "weight_decay": wd,
                }
        return named_param_lr_pairs


def get_dinov2_lr_decay_rate(name, lr_decay_rate=1.0, num_layers=12):
    """
    Calculate lr decay rate for different ViT blocks.

    Args:
        name (string): parameter name.
        lr_decay_rate (float): base lr decay rate.
        num_layers (int): number of ViT blocks.
    Returns:
        lr decay rate for the given parameter.
    """
    layer_id = num_layers + 1
    if name.startswith("backbone"):
        if "embeddings" in name:
            layer_id = 0
        elif ".layer." in name and ".residual." not in name:
            layer_id = int(name[name.find(".layer.") :].split(".")[2]) + 1
    return lr_decay_rate ** (num_layers + 1 - layer_id)

def get_dinov2_weight_decay_rate(name, weight_decay_rate=1.0):
    if (
        ("gamma" in name)
        or ("pos_embed" in name)
        or ("rel_pos" in name)
        or ("bias" in name)
        or ("norm" in name)
        or ("embeddings" in name)
    ):
        weight_decay_rate = 0.0
    return weight_decay_rate
