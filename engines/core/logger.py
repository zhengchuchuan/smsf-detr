from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Mapping

from omegaconf import DictConfig


def setup_runtime_logger(
    runtime_cfg: DictConfig | Mapping[str, Any],
    *,
    logger_name: str = "msifp-detr",
) -> logging.Logger:
    """
    根据 runtime 配置初始化日志系统:
    - 日志目录来自 runtime.output_root；
    - 文件名固定为 log.log；
    - 等级由 runtime.log_level 控制，默认为 INFO。
    同时输出到控制台和文件，重复调用会重置旧 handler。
    """

    def _lookup(key: str, default: Any) -> Any:
        if isinstance(runtime_cfg, DictConfig):
            return runtime_cfg.get(key, default)
        if isinstance(runtime_cfg, Mapping):
            return runtime_cfg.get(key, default)
        raise TypeError("runtime_cfg 必须是 DictConfig 或 Mapping")

    output_root = Path(_lookup("output_root", "outputs")).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    # DDP 多进程下避免多个进程同时写同一个文件导致日志互相打断/重复：
    # - rank0 写 log.log
    # - 其它 rank 写 log.rank{rank}.log（仍保留控制台输出）
    rank = 0
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            rank = int(dist.get_rank())
        else:
            rank = int(os.environ.get("RANK", "0"))
    except Exception:
        rank = int(os.environ.get("RANK", "0"))

    log_file = output_root / ("log.log" if rank == 0 else f"log.rank{rank}.log")

    level_name = str(_lookup("log_level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    # DDP 下默认只让 rank0 把 INFO/WARNING 打到控制台，避免重复刷屏；
    # 其它 rank 仍会写各自的日志文件，控制台仅保留 ERROR 以便快速发现异常。
    log_rank0_only = bool(_lookup("log_rank0_only", True))

    # 先清理并配置 root logger，确保全局可用。
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    stream_handler = logging.StreamHandler()
    if log_rank0_only and rank != 0:
        stream_handler.setLevel(logging.ERROR)
    else:
        stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = True  # 允许使用任何 logger 名称时共享 handler
    return logger
