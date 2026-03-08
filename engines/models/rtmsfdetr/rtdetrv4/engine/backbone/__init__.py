"""
RT-DETRv4 vendored backbones.

注意：避免在 `engine.backbone` 顶层导入所有可选 backbone。

原因：
- 部分 backbone 依赖可选包（例如 `timm`），但 rtmsfdetr 的 `rtv4_hgnetv2_s` baseline 仅需要 HGNetv2。
- 具体 backbone 会在 `engines/models/rtmsfdetr/builder.py` 中按需导入并完成注册。
"""

from .common import FrozenBatchNorm2d, freeze_batch_norm2d, get_activation
from .eemsa import EEMSA
from .hgnetv2 import HGNetv2

__all__ = [
    "FrozenBatchNorm2d",
    "freeze_batch_norm2d",
    "get_activation",
    "EEMSA",
    "HGNetv2",
]
