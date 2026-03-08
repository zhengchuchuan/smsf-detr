# ------------------------------------------------------------------------
# RF-DETR 注册入口，供模型注册表自动发现。
# ------------------------------------------------------------------------

from ..registry import register_model
from .builder import build_model, build_criterion_and_postprocessors, PostProcess

MODEL_NAME = "rf_detr"

register_model(
    MODEL_NAME,
    build_model_fn=build_model,
    build_criterion_fn=build_criterion_and_postprocessors,
    post_process_cls=PostProcess,
)

__all__ = [
    "build_model",
    "build_criterion_and_postprocessors",
    "PostProcess",
    "MODEL_NAME",
]
