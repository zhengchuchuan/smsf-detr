# ------------------------------------------------------------------------
# 模型注册表，用于在同一训练脚本内灵活切换不同论文的实现。
# ------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Type, Any


@dataclass(frozen=True)
class ModelEntry:
    build_model: Callable[[Any], Any]
    build_criterion_and_postprocessors: Callable[[Any], Any]
    post_process_cls: Type[Any]


_MODEL_REGISTRY: Dict[str, ModelEntry] = {}


def register_model(
    name: str,
    *,
    build_model_fn: Callable[[Any], Any],
    build_criterion_fn: Callable[[Any], Any],
    post_process_cls: Type[Any],
) -> None:
    """注册一个可供 main.py 调用的模型构建入口。"""
    if name in _MODEL_REGISTRY:
        raise ValueError(f"模型 {name} 已注册，不可重复注册。")
    _MODEL_REGISTRY[name] = ModelEntry(
        build_model=build_model_fn,
        build_criterion_and_postprocessors=build_criterion_fn,
        post_process_cls=post_process_cls,
    )


def get_model_entry(name: str) -> ModelEntry:
    """根据名称返回模型入口，未注册时给出可选列表便于排查。"""
    try:
        return _MODEL_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(_MODEL_REGISTRY)) or "<empty>"
        raise KeyError(f"未找到名为 {name} 的模型，当前可选：{available}") from exc


def list_models() -> Iterable[str]:
    """列出已注册模型名称，供 CLI choices 与帮助信息使用。"""
    return sorted(_MODEL_REGISTRY.keys())
