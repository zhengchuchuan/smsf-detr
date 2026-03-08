# ------------------------------------------------------------------------
# RTMSFDETR（RT-DETRv4 RGB baseline）注册入口。
# ------------------------------------------------------------------------

from ..registry import register_model
from .builder import build_model, build_criterion_and_postprocessors

MODEL_NAME = "rtmsfdetr"

# RT-DETRv4 的 PostProcessor 来自 vendored `engine`，这里不强制暴露具体类，保持注册表可用即可。
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
