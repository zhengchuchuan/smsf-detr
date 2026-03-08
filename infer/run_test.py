from __future__ import annotations

"""
基于 trainer.test 的推理入口（适配 RTMSFDETR 等 configs）。

示例：
  python infer/run_test.py --config outputs/<run_dir>/config.yaml
  python infer/run_test.py --config outputs/<run_dir>/config.yaml --checkpoint outputs/<run_dir>/checkpoint_best.pth
"""

import argparse
import logging
import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from engines.core.logger import setup_runtime_logger
from engines.core.parse_config import get_config, load_config
from utils.seed import fixed_random_seed


def _as_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def _load_config_any(
    config: str | Path, *, config_dir: str | Path, overrides: list[str] | None
) -> tuple[Any, Path | None]:
    """
    支持两种输入：
    1) `configs/` 目录内的 Hydra 配置（走 load_config + defaults 组合）
    2) `outputs/**/config.yaml` 这类“已保存的完整配置”（直接 OmegaConf.load）
    """
    config_path = _as_path(config).expanduser()
    if config_path.is_file():
        cfg = OmegaConf.load(str(config_path))
        OmegaConf.set_struct(cfg, False)
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
            OmegaConf.set_struct(cfg, False)
        return cfg, config_path
    cfg = load_config(config, config_dir=_as_path(config_dir), overrides=overrides)
    return cfg, None


def _resolve_checkpoint(checkpoint: str | None, config_path: Path | None) -> Path | None:
    if checkpoint:
        return _as_path(checkpoint).expanduser()
    if config_path is not None and config_path.is_file():
        for name in ("checkpoint_best.pth", "checkpoint.pth"):
            candidate = config_path.parent / name
            if candidate.is_file():
                return candidate
    return None


def _is_torchrun_distributed() -> bool:
    try:
        return int(os.environ.get("WORLD_SIZE", "1")) > 1
    except Exception:
        return False


def _maybe_init_distributed(cfg) -> tuple[int, int]:
    """
    Initialize torch.distributed if launched via torchrun.

    Returns:
      (rank, world_size)
    """
    runtime_cfg = get_config(cfg, "runtime", None) or {}
    if not _is_torchrun_distributed():
        return 0, 1

    device = str(get_config(runtime_cfg, "device", "cpu") or "cpu").lower()
    use_cuda = device != "cpu" and torch.cuda.is_available()
    default_backend = "nccl" if use_cuda else "gloo"
    backend = str(get_config(runtime_cfg, "dist_backend", default_backend) or default_backend)
    init_method = str(get_config(runtime_cfg, "dist_url", "env://") or "env://")
    timeout_s = int(get_config(runtime_cfg, "dist_timeout_s", 7200) or 7200)

    # Map local rank to a real CUDA device id, honoring runtime.device_ids when provided.
    device_ids = list(get_config(runtime_cfg, "device_ids", []) or [])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if use_cuda:
        if device_ids:
            if local_rank >= len(device_ids):
                raise ValueError(f"LOCAL_RANK={local_rank} out of range for runtime.device_ids={device_ids}")
            torch.cuda.set_device(int(device_ids[local_rank]))
        else:
            torch.cuda.set_device(local_rank)

    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available but WORLD_SIZE>1 was requested.")
    if not dist.is_initialized():
        # Passing `device_id` avoids NCCL guessing rank->GPU mapping (can hang on heterogeneous mapping)
        # and silences barrier() device-context warnings.
        device_id = int(torch.cuda.current_device()) if use_cuda else None
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            timeout=timedelta(seconds=timeout_s),
            device_id=device_id,
        )
        if use_cuda:
            dist.barrier(device_ids=[device_id])
        else:
            dist.barrier()

    return int(dist.get_rank()), int(dist.get_world_size())


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="基于 trainer.test 的推理入口")
    parser.add_argument(
        "--config",
        required=True,
        help="训练时使用的配置：支持 `configs/` 下 Hydra 配置 或 `outputs/**/config.yaml`。",
    )
    parser.add_argument("--config-dir", default="configs", help="Hydra 搜索配置的根目录（默认 configs）。")
    parser.add_argument(
        "--opts",
        nargs="*",
        default=None,
        help="可选覆盖项：KEY=VALUE（会在 load_config 时按顺序生效）。",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="评估用 checkpoint 路径（默认：若 config 来自 outputs/**/config.yaml，则自动使用同目录 checkpoint_best.pth/ checkpoint.pth）。",
    )
    parser.add_argument("--device", default="cuda", help="cpu/cuda/cuda:0 等（cuda 不可用会自动回退 CPU）。")
    parser.add_argument(
        "--output-root",
        default=None,
        help="覆盖 runtime.output_root（不传则沿用配置或自动回退到 config 同目录/outputs）。",
    )
    parser.add_argument("--no-ema", action="store_true", help="评估时禁用 EMA 权重。")
    parser.add_argument("--seed", type=int, default=None, help="覆盖 train.seed。")
    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg, config_path = _load_config_any(args.config, config_dir=args.config_dir, overrides=list(args.opts or []))
    with open_dict(cfg):
        cfg.mode = "test"
        if "runtime" not in cfg:
            cfg.runtime = {}
        cfg.runtime.device = str(args.device)
        if args.output_root:
            cfg.runtime.output_root = str(args.output_root)
        if args.no_ema:
            cfg.runtime.eval_use_ema = False
        if args.seed is not None:
            if "train" not in cfg:
                cfg.train = {}
            cfg.train.seed = int(args.seed)

    ckpt_path = _resolve_checkpoint(args.checkpoint, config_path)
    if ckpt_path is None or (not ckpt_path.is_file()):
        raise FileNotFoundError("未找到可用 checkpoint，请通过 --checkpoint 指定。")

    with open_dict(cfg):
        cfg.runtime.eval_ckpt = str(ckpt_path)
        if not getattr(cfg.runtime, "output_root", None):
            if config_path is not None and config_path.is_file():
                cfg.runtime.output_root = str(config_path.parent)
            else:
                cfg.runtime.output_root = "outputs"

    rank, world_size = _maybe_init_distributed(cfg)
    if world_size > 1:
        from utils.misc import setup_for_distributed

        setup_for_distributed(rank == 0)

    runtime_cfg = get_config(cfg, "runtime", None)
    if runtime_cfg is None:
        raise KeyError("配置中缺少 runtime 节点，无法初始化日志系统。")
    logger = setup_runtime_logger(runtime_cfg)
    logger.info("配置加载完成: %s", args.config)
    logger.info("评估权重: %s", ckpt_path)
    logger.info("输出目录: %s", get_config(runtime_cfg, "output_root", None))

    seed = int(get_config(getattr(cfg, "train", {}), "seed", 42))
    fixed_random_seed(seed)

    trainer_cfg = getattr(cfg, "trainer", None)
    if trainer_cfg is None:
        raise KeyError("配置中缺少 trainer 节点，无法实例化训练器。")
    trainer = instantiate(trainer_cfg, cfg)
    trainer.test()
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
