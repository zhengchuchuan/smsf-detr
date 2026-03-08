"""
RT-DETRv4 vendored engine package.

注意：不要在此处做“全量 import”来触发注册。

原因：
- RT-DETRv4 原仓库的 `engine/__init__.py` 会默认导入 data/backbone 等子模块，可能引入可选依赖
  （例如 `faster_coco_eval`、`timm`），而本工程的 rtmsfdetr 训练/评估并不需要这些依赖。
- 本工程会在 `engines/models/rtmsfdetr/builder.py` 中按需导入：
  `engine.core` / `engine.optim` / `engine.rtv4` / `engine.backbone.hgnetv2`
  以完成注册并构建模型。
"""

from .core.workspace import create, register  # re-export

__all__ = ["create", "register"]
