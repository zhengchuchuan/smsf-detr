import argparse
import logging
import re
from pathlib import Path
from typing import Dict, Iterable

import torch

from engines.core.parse_config import get_config
from engines.trainer.base_trainer import BaseTrainer


class MsifdetrTrainer(BaseTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)

    def _resolve_pretrain_weights(self, model_cfg: Dict) -> Path | None:
        """解析预训练权重路径（直接读取配置路径），若不存在则返回 None。"""
        weight_cfg = get_config(model_cfg, "pretrain_weights", None)
        if not weight_cfg:
            return None

        weight_path = Path(weight_cfg).expanduser()
        if not weight_path.is_file():
            logging.warning("预训练权重文件未找到，跳过加载: %s", weight_path)
            return None
        return weight_path

    def _load_pretrained(self, *, model: torch.nn.Module, path: Path, exclude_keys: Iterable[str] | None = None) -> None:
        """在不严格匹配的前提下加载预训练权重。"""
        exclude = tuple(exclude_keys or ())
        # PyTorch 2.6+ 默认 weights_only=True，需显式允许 argparse.Namespace。
        with torch.serialization.safe_globals([argparse.Namespace]):
            state = torch.load(path, map_location="cpu", weights_only=True)
        raw_state = state.get("model", state.get("state_dict", state))
        model_state = model.state_dict()

        filtered = {}
        skipped = []

        # 解析 ckpt 的 projector stage 数量（用于单尺度 ckpt -> 多尺度模型时对齐 P4 stage）
        stage_pat = re.compile(r"^backbone\.0\.projector\.(stages|stages_sampling)\.(\d+)\.")
        ckpt_max_stage = -1
        for k in raw_state.keys():
            m = stage_pat.match(k)
            if m:
                ckpt_max_stage = max(ckpt_max_stage, int(m.group(2)))
        ckpt_stage_count = ckpt_max_stage + 1 if ckpt_max_stage >= 0 else 0

        # 解析模型 projector_scale，用于确定 P4 在多尺度模型中的 index
        projector_scale = None
        try:
            backbone0 = getattr(getattr(model, "backbone"), "__getitem__")(0)
            projector_scale = getattr(backbone0, "projector_scale", None)
        except Exception:
            projector_scale = None
        if projector_scale is None:
            projector_scale = []
        projector_scale = list(projector_scale)
        p4_index = projector_scale.index("P4") if "P4" in projector_scale else None
        projector_prefix = "backbone.0.projector."

        def _maybe_remap_key(key: str) -> str:
            """
            兼容 projector_scale 从 ["P4"] 扩展到 ["P3","P4","P5"] 时的 stage 索引偏移：
            - 单尺度 checkpoint 的 stages.0 / stages_sampling.0 (P4) -> 当前 stages.{p4_index} / stages_sampling.{p4_index} (P4)
            """
            if key in model_state:
                return key
            if not key.startswith("backbone.0."):
                return key

            # projector 的 stage 映射需要优先处理：多尺度模型里 stages.0 通常是 P3，不能直接把 ckpt 的 P4(stage0) 载进去。
            if key.startswith("backbone.0.projector."):
                rel = key[len("backbone.0.projector.") :]
                # 若 ckpt 只有 1 个 stage 且模型包含 P4 且 P4 不在 index=0，则把 stage.0 映射到 stage.{p4_index}
                if ckpt_stage_count == 1 and p4_index not in (None, 0) and len(projector_scale) > 1:
                    rel = re.sub(r"^(stages(?:_sampling)?)\.0\.", rf"\1.{p4_index}.", rel, count=1)
                remapped = projector_prefix + rel
                if remapped in model_state:
                    return remapped
            return key

        for key, tensor in raw_state.items():
            if exclude and key.startswith(exclude):
                skipped.append(key)
                continue
            key_in_model = _maybe_remap_key(key)
            target = model_state.get(key_in_model)
            if target is None:
                skipped.append(key)
                continue
            if target.shape == tensor.shape:
                filtered[key_in_model] = tensor
                continue
            # 兼容 num_feature_levels 变化导致的 MS-DeformAttn 线性层输出维度变化：
            # sampling_offsets/attention_weights 的 out_dim 会随 n_levels 成倍变化，可通过沿 dim0 repeat 方式扩展。
            can_expand = ("sampling_offsets" in key_in_model) or ("attention_weights" in key_in_model)
            if can_expand and tensor.ndim == target.ndim:
                if tensor.ndim == 2 and tensor.shape[1] == target.shape[1] and target.shape[0] % tensor.shape[0] == 0:
                    rep = target.shape[0] // tensor.shape[0]
                    filtered[key_in_model] = tensor.repeat(rep, 1)[: target.shape[0]]
                    continue
                if tensor.ndim == 1 and target.shape[0] % tensor.shape[0] == 0:
                    rep = target.shape[0] // tensor.shape[0]
                    filtered[key_in_model] = tensor.repeat(rep)[: target.shape[0]]
                    continue
            skipped.append(key)

        missing, unexpected = model.load_state_dict(filtered, strict=False)
        logging.info(
            "预训练权重加载完成: 使用 %d 项, 跳过 %d 项, 缺失 %d 项, 额外 %d 项",
            len(filtered),
            len(skipped),
            len(missing),
            len(unexpected),
        )
        total = len(raw_state)
        if total > 0:
            coverage = len(filtered) / float(total)
            if coverage < 0.8:
                logging.warning(
                    "预训练加载覆盖率偏低(%.1f%%)。若 mAP 异常偏低，优先检查配置是否与 ckpt 对齐："
                    "model.group_detr / train.patch_size / model.patch_size / model.positional_encoding_size / model.projector_scale。",
                    coverage * 100.0,
                )

    def build_model(self):
        try:
            from engines.models.msifdetr.builder import build_model as build_msifdetr_model
        except ImportError as exc:
            raise ImportError("构建 MSIF-DETR 模型失败，请确认 transformers/peft 等依赖已安装。") from exc

        args = self._ensure_data_args()
        model_cfg = get_config(self.cfg, "model", {}) or {}
        train_cfg = get_config(self.cfg, "train", {}) or {}

        # 模型构建依赖的核心超参
        args.device = str(self.device)
        args.num_classes = int(get_config(train_cfg, "num_classes", get_config(model_cfg, "num_classes", 1)))
        args.aux_loss = bool(get_config(train_cfg, "aux_loss", getattr(args, "aux_loss", False)))
        args.dropout = float(get_config(train_cfg, "dropout", 0.0))
        args.drop_path = float(get_config(train_cfg, "drop_path", 0.0))
        args.dec_layers = int(get_config(model_cfg, "dec_layers", getattr(args, "dec_layers", 1)))
        args.dec_n_points = int(get_config(model_cfg, "dec_n_points", getattr(args, "dec_n_points", 4)))
        args.decoder_norm = get_config(model_cfg, "decoder_norm", getattr(args, "decoder_norm", "LN"))
        args.segmentation_head = bool(get_config(model_cfg, "segmentation_head", get_config(train_cfg, "segmentation_head", getattr(args, "segmentation_head", False))))
        args.num_select = int(
            get_config(
                model_cfg,
                "num_select",
                get_config(train_cfg, "num_select", getattr(args, "num_queries", 300)),
            )
        )
        args.load_dinov2_weights = bool(get_config(model_cfg, "load_dinov2_weights", getattr(args, "load_dinov2_weights", True)))
        args.encoder_only = bool(getattr(args, "encoder_only", False))
        args.backbone_only = bool(getattr(args, "backbone_only", False))
        # Decoder 侧融合（Query 级语义对齐）
        args.decoder_ms_fusion = bool(get_config(model_cfg, "decoder_ms_fusion", False))
        args.decoder_ms_fusion_alpha_init = float(get_config(model_cfg, "decoder_ms_fusion_alpha_init", 0.0))
        args.decoder_ms_fusion_alpha_scale = float(get_config(model_cfg, "decoder_ms_fusion_alpha_scale", 1.0))
        args.freeze_decoder_ms_fusion_alpha = bool(get_config(model_cfg, "freeze_decoder_ms_fusion_alpha", False))
        # Decoder 侧 MSI reference shift（补偿残余平移错位）
        args.decoder_ms_fusion_ref_shift = bool(get_config(model_cfg, "decoder_ms_fusion_ref_shift", False))
        args.decoder_ms_fusion_ref_shift_init = get_config(model_cfg, "decoder_ms_fusion_ref_shift_init", (0.0, 0.0))
        args.decoder_ms_fusion_ref_shift_scale = float(get_config(model_cfg, "decoder_ms_fusion_ref_shift_scale", 0.02))
        args.freeze_decoder_ms_fusion_ref_shift = bool(get_config(model_cfg, "freeze_decoder_ms_fusion_ref_shift", False))

        pretrain_path = None
        if not get_config(model_cfg, "force_no_pretrain", False):
            pretrain_path = self._resolve_pretrain_weights(model_cfg)
        args.pretrain_weights = str(pretrain_path) if pretrain_path else None

        self.model = build_msifdetr_model(args).to(self.device)
        if pretrain_path is not None:
            exclude = get_config(model_cfg, "pretrain_exclude_keys", None)
            self._load_pretrained(model=self.model, path=pretrain_path, exclude_keys=exclude)
        return self.model
