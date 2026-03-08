"""模型组件包。
"""

"""
注意：这里不要无条件导入所有模型子包。

原因：
- 部分模型依赖可选第三方库（如 transformers 等），在只训练其它模型时也会被提前 import，
  导致整个工程无法启动。
- 绝大多数场景下，Hydra instantiate 会按需导入具体的 _target_，无需在包入口做“全量注册”。
"""

from importlib import import_module
import logging

_LOG = logging.getLogger(__name__)


def _optional_import(submodule: str) -> None:
    try:
        import_module(f"{__name__}.{submodule}")
    except ModuleNotFoundError as exc:
        # 仅忽略缺失的“外部依赖”；如果是本项目子模块缺失则应直接报错。
        missing = getattr(exc, "name", None) or ""
        if missing.startswith(__name__):
            raise
        _LOG.debug("Skip optional model package %s (missing dependency: %s)", submodule, missing)


for _name in ("rfdetr", "msifdetr", "mscft", "rtmsfdetr"):
    _optional_import(_name)
