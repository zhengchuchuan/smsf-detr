import argparse
import logging
from pathlib import Path
from typing import Dict, Iterable

import torch

from engines.core.parse_config import get_config
from engines.trainer.base_trainer import BaseTrainer

class RfdetrTrainer(BaseTrainer):
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
        for key, tensor in raw_state.items():
            if exclude and key.startswith(exclude):
                skipped.append(key)
                continue
            target = model_state.get(key)
            if target is None:
                skipped.append(key)
                continue
            if target.shape == tensor.shape:
                filtered[key] = tensor
                continue
            # 兼容 num_feature_levels 变化导致的 MS-DeformAttn 线性层输出维度变化。
            can_expand = ("sampling_offsets" in key) or ("attention_weights" in key)
            if can_expand and tensor.ndim == target.ndim:
                if tensor.ndim == 2 and tensor.shape[1] == target.shape[1] and target.shape[0] % tensor.shape[0] == 0:
                    rep = target.shape[0] // tensor.shape[0]
                    filtered[key] = tensor.repeat(rep, 1)[: target.shape[0]]
                    continue
                if tensor.ndim == 1 and target.shape[0] % tensor.shape[0] == 0:
                    rep = target.shape[0] // tensor.shape[0]
                    filtered[key] = tensor.repeat(rep)[: target.shape[0]]
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

    def build_model(self):
        try:
            from engines.models.rfdetr.builder import build_model as build_rfdetr_model
        except ImportError as exc:
            raise ImportError("构建 RF-DETR 模型失败，请确认 transformers/peft 等依赖已安装。") from exc

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
        args.load_dinov2_weights = bool(get_config(model_cfg, "load_dinov2_weights", getattr(args, "load_dinov2_weights", True)))
        args.num_select = int(
            get_config(
                model_cfg,
                "num_select",
                get_config(train_cfg, "num_select", getattr(args, "num_queries", 300)),
            )
        )
        args.encoder_only = bool(getattr(args, "encoder_only", False))
        args.backbone_only = bool(getattr(args, "backbone_only", False))

        pretrain_path = None
        if not get_config(model_cfg, "force_no_pretrain", False):
            pretrain_path = self._resolve_pretrain_weights(model_cfg)
        args.pretrain_weights = str(pretrain_path) if pretrain_path else None

        self.model = build_rfdetr_model(args).to(self.device)
        if pretrain_path is not None:
            exclude = get_config(model_cfg, "pretrain_exclude_keys", None)
            self._load_pretrained(model=self.model, path=pretrain_path, exclude_keys=exclude)
        return self.model
