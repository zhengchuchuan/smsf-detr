from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple, Union

import torch.nn as nn

from utils.misc import NestedTensor

try:  # pragma: no cover - core.data 可能在早期导入阶段不可用
    from engines.core.data.modal import ModalBatch
except Exception:  # pragma: no cover
    ModalBatch = None  # type: ignore

ModalInputs = Union[NestedTensor, Dict[str, NestedTensor]]


@dataclass(frozen=True)
class ModalInfo:
    """描述模型在不同模态下的期望输入。"""

    primary: str
    available: Tuple[str, ...]

    def is_supported_by(self, supported: Iterable[str]) -> bool:
        supported_set = {item.lower() for item in supported}
        return all(mod.lower() in supported_set for mod in self.available)


class BaseDetector(nn.Module):
    """所有检测模型共享的最小接口，统一模态能力声明与批量适配。"""

    supported_modalities: Tuple[str, ...] = ("rgb",)
    default_primary_modality: str = "rgb"

    def __init__(self) -> None:
        super().__init__()

    # ----- 模态处理 -----
    def prepare_modal_inputs(
        self,
        samples: ModalInputs,
        *,
        primary: Optional[str] = None,
    ) -> ModalInputs:
        """可由子类重写，以在前向前对多模态输入进行融合或筛选。"""
        return samples

    def forward_from_modal_batch(self, batch: "ModalBatch") -> Dict:
        if ModalBatch is None:
            raise RuntimeError("ModalBatch 未注册，无法处理多模态批数据。")
        samples, targets = batch.as_legacy_tuple()
        prepared = self.prepare_modal_inputs(
            samples,
            primary=batch.primary or self.default_primary_modality,
        )
        return self(prepared, targets)

    # ----- 能力声明 -----
    def modal_info(self, available: Iterable[str], *, primary: Optional[str] = None) -> ModalInfo:
        normalized = tuple(str(mod).lower() for mod in available)
        fallback_primary = primary or self.default_primary_modality
        if fallback_primary not in normalized and normalized:
            fallback_primary = normalized[0]
        return ModalInfo(primary=fallback_primary, available=normalized)

    def supports_modalities(self, modalities: Iterable[str]) -> bool:
        info = ModalInfo(primary=self.default_primary_modality, available=tuple(modalities))
        return info.is_supported_by(self.supported_modalities)

    def validate_modalities(self, modalities: Iterable[str], split: Optional[str] = None) -> None:
        if not modalities:
            return
        if self.supports_modalities(modalities):
            return
        split_hint = f"（split={split}）" if split else ""
        raise ValueError(
            f"{self.__class__.__name__} 不支持数据加载器提供的模态 {tuple(modalities)}{split_hint}；"
            f"模型仅声明支持 {self.supported_modalities}。"
        )


__all__ = ["BaseDetector", "ModalInfo"]
