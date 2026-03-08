# ------------------------------------------------------------------------
# SMSFDETR 注册入口（阶段1: 与 RTMSFDETR 等价基线）。
# ------------------------------------------------------------------------

from ..registry import register_model
from .builder import build_model, build_criterion_and_postprocessors

MODEL_NAME = "smsfdetr"

# 阶段1保持与 RTMSFDETR 同行为：PostProcessor 仍由 vendored RT-DETRv4 提供。
register_model(
    MODEL_NAME,
    build_model_fn=build_model,
    build_criterion_fn=build_criterion_and_postprocessors,
    post_process_cls=object,  # type: ignore[arg-type]
)

__all__ = [
    "MODEL_NAME",
    "build_model",
    "build_criterion_and_postprocessors",
]
