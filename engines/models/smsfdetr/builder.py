"""
阶段1等价基线：
- SMSFDETR 先完全复用 RTMSFDETR 的模型/损失/后处理构建逻辑；
- 后续模块化增强（EEMSA/ACAF/P2DBF）在该入口逐步接入。
"""

from engines.models.rtmsfdetr.builder import build_criterion_and_postprocessors
from engines.models.rtmsfdetr.builder import build_model
from engines.models.rtmsfdetr.builder import build_model_and_processors

__all__ = [
    "build_model",
    "build_criterion_and_postprocessors",
    "build_model_and_processors",
]
