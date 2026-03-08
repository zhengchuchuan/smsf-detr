from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf


DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def resolve_config_name(
    config: str | Path, config_dir: str | Path | None = None
) -> str:
    """
    将配置名称/路径解析为字符串标识（不含扩展名）。

    - 若配置文件位于 config_dir 之下，返回 Hydra 需要的相对路径（不含扩展名）。
    - 若配置文件不在 config_dir 之下，返回其绝对路径（不含扩展名）。
    """
    config_root = Path(config_dir or DEFAULT_CONFIG_DIR).resolve()
    return _resolve_config_name(config, config_root)


def _resolve_config_file(entry: str | Path, config_dir: Path) -> Path:
    config_dir = config_dir.resolve()
    entry_str = str(entry).strip()
    if not entry_str:
        raise ValueError("配置文件名不能为空。")

    candidate = Path(entry_str)
    if candidate.suffix not in {".yaml", ".yml"}:
        candidate = candidate.with_suffix(".yaml")

    # 绝对路径只需验证一次；否则依次在当前目录和配置根目录查找。
    if candidate.is_absolute():
        search_paths = [candidate]
    else:
        search_paths = [Path.cwd() / candidate, config_dir / candidate]

    for path in search_paths:
        if path.exists():
            return path.resolve()

    raise FileNotFoundError(f"无法找到配置文件：{candidate}")


def _resolve_config_name(entry: str | Path, config_dir: Path) -> str:
    """
    将用户输入的配置名称/路径转换为 Hydra 需要的相对路径（不含扩展名）。

    允许以下几种写法：
    - 绝对路径或相对路径，可带或不带 .yaml/.yml 扩展名；
    - 直接传入 Hydra 风格的相对路径（例如 task/rfdetr/coco_det_nano）。
    """
    config_dir = config_dir.resolve()
    resolved = _resolve_config_file(entry, config_dir)
    try:
        relative = resolved.relative_to(config_dir)
    except ValueError as exc:
        # 配置文件不在 config_dir 下时，返回绝对路径标识，供日志/输出目录命名使用。
        # 注意：该返回值不适用于 Hydra compose。
        return resolved.with_suffix("").as_posix()

    return relative.with_suffix("").as_posix()


def load_config(
    config: str | Path,
    *,
    config_dir: str | Path | None = None,
    overrides: Iterable[str] | None = None,
) -> DictConfig:
    """
    加载单个配置文件，并按需应用命令行覆盖项，返回 DictConfig。

    - 若配置文件位于 config_dir 之下，则使用 Hydra compose（支持 defaults/config group）。
    - 若配置文件位于 config_dir 之外，则直接 OmegaConf.load（适用于 outputs/ 下保存的完整 config.yaml）。
    """
    config_root = Path(config_dir or DEFAULT_CONFIG_DIR).resolve()
    resolved = _resolve_config_file(config, config_root)
    try:
        relative = resolved.relative_to(config_root)
    except ValueError:
        relative = None

    # 重新初始化 Hydra，避免脚本多次调用时复用旧状态。
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    if relative is None:
        cfg = OmegaConf.load(resolved)
        OmegaConf.set_struct(cfg, False)
    else:
        config_name = relative.with_suffix("").as_posix()
        with initialize_config_dir(
            config_dir=str(config_root), job_name="config_loader", version_base=None
        ):
            cfg = compose(config_name=config_name, return_hydra_config=False)
            OmegaConf.set_struct(cfg, False)  # 允许后续动态写入新键

    override_list = list(overrides or [])
    if override_list:
        override_cfg = OmegaConf.from_dotlist(override_list)
        cfg = OmegaConf.merge(cfg, override_cfg)
        OmegaConf.set_struct(cfg, False)

    return cfg

def get_config(node, key, default=None):
    if node is None:
        return default
    if hasattr(node, "get"):
        return node.get(key, default)
    return getattr(node, key, default)
