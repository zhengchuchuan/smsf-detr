# ------------------------------------------------------------------------
# MSCFT 模型注册入口（基于 third_party/mscft YOLO+CFT 迁移实现）。
# ------------------------------------------------------------------------

from ..registry import register_model
from .builder import build_model, build_criterion_and_postprocessors, PostProcessWrapper

MODEL_NAME = "mscft"

register_model(
    MODEL_NAME,
    build_model_fn=build_model,
    build_criterion_fn=build_criterion_and_postprocessors,
    post_process_cls=PostProcessWrapper,
)

__all__ = [
    "build_model",
    "build_criterion_and_postprocessors",
    "PostProcessWrapper",
    "MODEL_NAME",
]
