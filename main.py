import argparse
import hashlib
import os
import subprocess
import sys
from datetime import datetime
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import OmegaConf

from engines.core.logger import setup_runtime_logger
from engines.core.parse_config import get_config
from engines.core.parse_config import load_config
from engines.core.parse_config import resolve_config_name
from utils.seed import fixed_random_seed

_AUTORUN_ENV_FLAG = "MSIFP_DETR_DISABLE_AUTORUN"


def _truncate_utf8_bytes(value: str, max_bytes: int) -> str:
    if max_bytes <= 0:
        return ""
    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


def _compact_path_component(value: str, max_bytes: int = 120) -> str:
    text = str(value)
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text

    digest = hashlib.sha1(encoded).hexdigest()[:10]
    suffix = ""
    stem = text
    path = Path(text)
    if path.suffix and path.name != path.suffix:
        suffix = path.suffix
        stem = text[:-len(suffix)]

    reserved = len(digest) + 1 + len(suffix.encode("utf-8"))
    head_budget = max_bytes - reserved
    if head_budget < 16:
        suffix = ""
        reserved = len(digest) + 1
        head_budget = max_bytes - reserved

    head = _truncate_utf8_bytes(stem, head_budget)
    compact = f"{head}-{digest}{suffix}"
    if len(compact.encode("utf-8")) > max_bytes:
        overflow = len(compact.encode("utf-8")) - max_bytes
        head = _truncate_utf8_bytes(head, len(head.encode("utf-8")) - overflow)
        compact = f"{head}-{digest}{suffix}"
    return compact


def _build_output_dir(
    *,
    output_root: str,
    dataset_name: str,
    model_name: str,
    model_alias: str,
    config_name: str,
    timestamp: str,
) -> tuple[Path, dict[str, tuple[str, str]]]:
    compacted = {
        "dataset_name": (dataset_name, _compact_path_component(dataset_name, max_bytes=96)),
        "model_name": (model_name, _compact_path_component(model_name, max_bytes=96)),
        "model_alias": (model_alias, _compact_path_component(model_alias, max_bytes=120)),
        "run_name": (
            f"{timestamp}-{config_name}",
            _compact_path_component(f"{timestamp}-{config_name}", max_bytes=160),
        ),
    }
    output_dir = (
        Path(output_root).expanduser()
        / compacted["dataset_name"][1]
        / compacted["model_name"][1]
        / compacted["model_alias"][1]
        / compacted["run_name"][1]
    )
    renamed = {key: value for key, value in compacted.items() if value[0] != value[1]}
    return output_dir, renamed


def _is_torchrun_distributed() -> bool:
    try:
        return int(os.environ.get("WORLD_SIZE", "1")) > 1
    except Exception:
        return False


def _maybe_autolaunch_torchrun(*, cfg, argv: list[str]) -> None:
    """
    Auto-launch multi-process DDP when runtime.device_ids specifies multiple GPUs.

    Notes:
    - True multi-GPU training requires multiple processes (DDP). DataParallel is intentionally not used.
    - This auto-launch is skipped when already running under torchrun (WORLD_SIZE>1) or when disabled via env flag.
    """
    runtime_cfg = get_config(cfg, "runtime", None)
    if runtime_cfg is None:
        return
    device = str(get_config(runtime_cfg, "device", "cpu") or "cpu").lower()
    device_ids = list(get_config(runtime_cfg, "device_ids", []) or [])
    auto_ddp = bool(get_config(runtime_cfg, "auto_ddp", True))
    if not auto_ddp:
        return
    if _is_torchrun_distributed():
        return
    if os.environ.get(_AUTORUN_ENV_FLAG, "0") == "1":
        return
    # Only auto-launch DDP when using CUDA. (CPU multi-process is not supported by this auto-run.)
    if device == "cpu" or (not torch.cuda.is_available()):
        return
    if len(device_ids) <= 1:
        return

    nproc = len(device_ids)
    cmd = ["torchrun", "--standalone", "--nproc_per_node", str(nproc), str(Path(__file__).resolve()), *argv]
    env = dict(os.environ)
    env[_AUTORUN_ENV_FLAG] = "1"
    proc = subprocess.run(cmd, env=env)
    raise SystemExit(proc.returncode)


def _maybe_init_distributed(*, cfg) -> tuple[int, int]:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spectral Recovery Toolbox")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='train or test')
    parser.add_argument('--config', type=str,
                        default='configs/task/rfdetr/coco_det_nano.yaml',
                        help='配置名称或路径。',)
    parser.add_argument('--config-dir', type=str, default='configs', help='Hydra 搜索配置的根目录。',)
    parser.add_argument('--opts', nargs='*', default=None,
                        help='可选的 KEY=VALUE 覆盖项，如: train.epochs=666, 最终在 OmegaConf 中按顺序生效。',)
    args = parser.parse_args()
    config_root = Path(args.config_dir).resolve()
    # 加载yaml配置
    cfg = load_config(args.config, config_dir=config_root, overrides=args.opts)
    # 将 CLI 模式写回配置，供后续训练/评估分支使用
    cfg.mode = args.mode

    # Auto-launch torchrun for multi-GPU if requested via runtime.device_ids.
    _maybe_autolaunch_torchrun(cfg=cfg, argv=sys.argv[1:])

    # Init distributed (DDP) when launched via torchrun, before generating output dir.
    rank, world_size = _maybe_init_distributed(cfg=cfg)
    if world_size > 1:
        # Silence `print()` on non-rank0 processes to avoid duplicated stdout spam (COCOeval, progress, etc.).
        from utils.misc import setup_for_distributed

        setup_for_distributed(rank == 0)

    # 动态构建输出目录：root/dataset/model_variant/timestamp-config_name

    output_root = get_config(cfg.runtime, "output_root", "outputs")

    # data.dataset_file 表示“数据集类型/加载器实现”（如 coco_rgb / coco_rgb_msi），不适合作为输出目录的“数据集名称”。
    # 优先使用显式配置的 data.dataset_name；否则回退到 dataset_dir 的目录名；最后才用 dataset_file。
    dataset_name = get_config(cfg.data, "dataset_name", None)
    if not dataset_name:
        dataset_dir = get_config(cfg.data, "dataset_dir", None) or get_config(cfg.data, "coco_path", None)
        if dataset_dir:
            dataset_name = Path(str(dataset_dir)).name
    if not dataset_name:
        dataset_name = get_config(cfg.data, "dataset_file", "dataset")

    model_name = get_config(cfg.model, "model_name", "model")
    # model_alias 用于输出目录分组；若缺省则提供兜底，避免 Path 拼接 NoneType 报错。
    model_alias = get_config(cfg.model, "model_alias", None) or "default"
    config_name = Path(resolve_config_name(args.config, config_root)).name
    output_path_compactions: dict[str, tuple[str, str]] = {}
    if world_size > 1:
        # Ensure all ranks share the same run directory.
        if rank == 0:
            timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
            output_dir, output_path_compactions = _build_output_dir(
                output_root=output_root,
                dataset_name=dataset_name,
                model_name=model_name,
                model_alias=model_alias,
                config_name=config_name,
                timestamp=timestamp,
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            payload: list[str] = [str(output_dir)]
        else:
            payload = [""]
        dist.broadcast_object_list(payload, src=0)
        output_dir = Path(payload[0]).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        dist.barrier()
    else:
        timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
        output_dir, output_path_compactions = _build_output_dir(
            output_root=output_root,
            dataset_name=dataset_name,
            model_name=model_name,
            model_alias=model_alias,
            config_name=config_name,
            timestamp=timestamp,
        )
        output_dir.mkdir(parents=True, exist_ok=True)

    # 将生成的目录写回配置和 W&B 本地输出路径
    runtime_cfg = get_config(cfg, "runtime", None)
    if runtime_cfg is not None:
        runtime_cfg.output_root = str(output_dir)
    os.environ.setdefault("WANDB_DIR", str(output_dir / "wandb"))

    # 保存完整配置，便于复现实验
    config_file = output_dir / "config.yaml"
    if world_size <= 1 or rank == 0:
        OmegaConf.save(config=cfg, f=str(config_file))

    # 初始化日志系统
    if runtime_cfg is None:
        raise KeyError("配置中缺少 runtime 节点，无法初始化日志系统。")
    logger = setup_runtime_logger(runtime_cfg)
    logger.info("配置加载完成: %s", args.config)
    logger.info("日志系统初始化完成。")
    for key, (original, compacted) in output_path_compactions.items():
        logger.info("输出目录分量过长，已自动缩短 %s: %s -> %s", key, original, compacted)
    # 根据配置文件实例化训练器
    trainer_cfg = getattr(cfg, "trainer", None)
    if trainer_cfg is None:
        raise KeyError("配置中缺少 trainer 节点，无法实例化训练器。")
    trainer = instantiate(trainer_cfg, cfg)
    # 固定随机种子
    seed = get_config(cfg.train, "seed", 42)
    fixed_random_seed(seed)
    # 开始训练或测试
    if args.mode == 'train':
        trainer.train()
    elif args.mode == 'test':
        trainer.test()
    else:
        raise ValueError(f"不支持的模式: {args.mode}")
    
