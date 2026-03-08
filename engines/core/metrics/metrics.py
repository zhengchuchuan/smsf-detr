import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Tuple

import torch
from torch import nn

from engines.core.parse_config import get_config
from engines.core.metrics.rfdetr_loss import (
    HungarianMatcher,
    PostProcess,
    SetCriterion,
)
# 约定：注册函数返回 (criterion: nn.Module, extras: dict|None)，extras 可挂载 matcher/postprocess 等辅助组件。
LOSS_REGISTRY: Dict[str, Callable[..., Tuple[nn.Module, Dict[str, Any]]]] = {}


def register_loss(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """装饰器：将损失构造器注册到全局表中。"""
    def decorator(builder: Callable[..., Any]) -> Callable[..., Any]:
        if name in LOSS_REGISTRY:
            logging.warning("重复注册损失构造器：%s，覆盖旧值。", name)
        LOSS_REGISTRY[name] = builder
        return builder
    return decorator


def build_loss(name: str, **kwargs) -> Tuple[nn.Module, Dict[str, Any]]:
    """根据注册名构建损失，默认返回 (criterion, extras)。"""
    if name not in LOSS_REGISTRY:
        raise KeyError(f"未找到名为 {name} 的损失构造器，请先注册。")
    builder = LOSS_REGISTRY[name]
    result = builder(**kwargs)
    if isinstance(result, tuple):
        if len(result) == 2:
            criterion, extras = result
            extras = extras or {}
            return criterion, extras
        if len(result) == 1:
            return result[0], {}
    return result, {}


@dataclass
class _LossItem:
    name: str
    weight: float
    module: nn.Module


class CompositeCriterion(nn.Module):
    """
    聚合多个子损失：先按配置归一化权重，再将各子损失求和，输出总损失与分项日志。
    - 子损失 forward 需返回 Tensor 或 {str: Tensor}。
    - 汇总时会为键自动加上前缀，避免不同子损失同名键冲突。
    """
    def __init__(self, losses: Iterable[_LossItem]):
        super().__init__()
        loss_list = list(losses)
        if not loss_list:
            raise ValueError("CompositeCriterion 初始化失败：losses 不能为空。")
        weights = torch.tensor([max(0.0, float(item.weight)) for item in loss_list], dtype=torch.float32)
        total = float(weights.sum().item())
        norm_weights = (weights / total) if total > 0 else torch.ones_like(weights) / len(loss_list)

        # 保存模块和元数据
        self.loss_items: List[_LossItem] = []
        self.loss_modules = nn.ModuleList()
        for item, norm_w in zip(loss_list, norm_weights):
            self.loss_items.append(_LossItem(name=item.name, weight=float(norm_w.item()), module=item.module))
            self.loss_modules.append(item.module)

        # 持久化权重，方便 forward 使用
        self.register_buffer("_norm_weights", norm_weights, persistent=False)

    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        loss_dict: Dict[str, torch.Tensor] = {}
        total_loss = torch.tensor(0.0, device=self._norm_weights.device)

        for idx, item in enumerate(self.loss_items):
            module = self.loss_modules[idx]
            raw = module(*args, **kwargs)
            if isinstance(raw, dict):
                current = raw
            else:
                current = {item.name: raw}

            # 统一 key，防止冲突
            merged = {}
            for k, v in current.items():
                key = k
                if not k.startswith(item.name):
                    key = f"{item.name}.{k}"
                merged[key] = v

            # 仅聚合 loss，允许前缀存在（如 set_criterion.loss_ce）
            scalar_sum = sum(
                v
                for k, v in merged.items()
                if torch.is_tensor(v) and k.split(".", 1)[-1].startswith("loss")
            )
            total_loss = total_loss + self._norm_weights[idx] * scalar_sum

            loss_dict.update(merged)

        loss_dict["loss_total"] = total_loss
        return loss_dict


@register_loss("set_criterion")
def build_set_criterion(cfg: Any = None, loss_cfg: Dict[str, Any] | None = None, **params: Any) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    复用 RF-DETR 的损失组合逻辑，按配置构建 matcher + SetCriterion + PostProcess。
    - 依赖 train.* 与 model.* 中的相关超参。
    - params 可用于覆写特定超参（例如测试自定义 loss 参数）。
    """
    train_cfg = get_config(cfg, "train", {}) if cfg is not None else {}
    model_cfg = get_config(cfg, "model", {}) if cfg is not None else {}

    # 优先 train 覆盖 model
    num_classes = get_config(train_cfg, "num_classes", get_config(model_cfg, "num_classes", 1))
    group_detr = get_config(model_cfg, "group_detr", get_config(train_cfg, "group_detr", 1))
    aux_loss = get_config(train_cfg, "aux_loss", False)
    two_stage = get_config(model_cfg, "two_stage", get_config(train_cfg, "two_stage", False))
    dec_layers = get_config(model_cfg, "dec_layers", 1)
    segmentation_head = get_config(train_cfg, "segmentation_head", get_config(model_cfg, "segmentation_head", False))

    matcher = HungarianMatcher(
        cost_class=get_config(train_cfg, "set_cost_class", 1.0),
        cost_bbox=get_config(train_cfg, "set_cost_bbox", 1.0),
        cost_giou=get_config(train_cfg, "set_cost_giou", 1.0),
        focal_alpha=get_config(train_cfg, "focal_alpha", 0.25),
        cost_mask_ce=get_config(train_cfg, "mask_ce_loss_coef", 1.0),
        cost_mask_dice=get_config(train_cfg, "mask_dice_loss_coef", 1.0),
        mask_point_sample_ratio=get_config(train_cfg, "mask_point_sample_ratio", 16),
    )

    weight_dict: Dict[str, float] = {
        'loss_ce': get_config(train_cfg, "cls_loss_coef", 1.0),
        'loss_bbox': get_config(train_cfg, "bbox_loss_coef", 5.0),
        'loss_giou': get_config(train_cfg, "giou_loss_coef", 2.0),
    }
    if segmentation_head:
        weight_dict['loss_mask_ce'] = get_config(train_cfg, "mask_ce_loss_coef", 1.0)
        weight_dict['loss_mask_dice'] = get_config(train_cfg, "mask_dice_loss_coef", 1.0)

    if aux_loss:
        aux_weight_dict = {}
        for i in range(max(0, dec_layers - 1)):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        if two_stage:
            aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    if segmentation_head:
        losses.append('masks')

    criterion = SetCriterion(
        num_classes=num_classes + 1,
        matcher=matcher,
        weight_dict=weight_dict,
        focal_alpha=get_config(train_cfg, "focal_alpha", 0.25),
        losses=losses,
        group_detr=group_detr,
        sum_group_losses=get_config(train_cfg, "sum_group_losses", False),
        use_varifocal_loss=get_config(train_cfg, "use_varifocal_loss", False),
        use_position_supervised_loss=get_config(train_cfg, "use_position_supervised_loss", False),
        ia_bce_loss=get_config(train_cfg, "ia_bce_loss", False),
        mask_point_sample_ratio=get_config(train_cfg, "mask_point_sample_ratio", 16),
    )

    extras = {
        "postprocess": PostProcess(num_select=get_config(model_cfg, "num_select", 300)),
        "weight_dict": weight_dict,
    }
    return criterion, extras
