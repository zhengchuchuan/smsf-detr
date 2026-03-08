from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable as TypingIterable

import torch
from omegaconf import OmegaConf
import re

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engines.core.parse_config import load_config
from engines.trainer.msifdetr_train import MsifdetrTrainer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查 MSIF-DETR 预训练权重加载覆盖率（CPU 离线）")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--config",
        type=str,
        help="Hydra 配置路径（相对 configs 根目录或其子路径），例如 configs/task/msifdetr/xxx.yaml",
    )
    group.add_argument(
        "--resolved-config",
        type=str,
        help="已落盘的完整 config.yaml（如 outputs/.../config.yaml），不经过 Hydra compose。",
    )
    parser.add_argument("--config-dir", type=str, default="configs", help="Hydra 配置根目录。")
    parser.add_argument(
        "--opts",
        nargs="*",
        default=None,
        help="可选 dotlist 覆盖项，如 model.group_detr=13 train.patch_size=16",
    )
    return parser.parse_args()

def _load_cfg(args: argparse.Namespace):
    if args.resolved_config:
        cfg = OmegaConf.load(args.resolved_config)
    else:
        cfg = load_config(args.config, config_dir=args.config_dir, overrides=args.opts)
        # load_config 已经应用 overrides，无需重复 merge
        return cfg
    # resolved-config 场景手动应用 overrides
    if args.opts:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(args.opts)))
    return cfg


def _count_pretrain_coverage(
    *,
    model: torch.nn.Module,
    ckpt_path: Path,
    exclude_keys: TypingIterable[str] | None = None,
) -> dict:
    exclude = tuple(exclude_keys or ())
    with torch.serialization.safe_globals([argparse.Namespace]):
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    raw_state = state.get("model", state.get("state_dict", state))
    model_state = model.state_dict()

    stage_pat = re.compile(r"^backbone\.0\.projector\.(stages|stages_sampling)\.(\d+)\.")
    ckpt_max_stage = -1
    for k in raw_state.keys():
        m = stage_pat.match(k)
        if m:
            ckpt_max_stage = max(ckpt_max_stage, int(m.group(2)))
    ckpt_stage_count = ckpt_max_stage + 1 if ckpt_max_stage >= 0 else 0

    projector_scale = []
    try:
        backbone0 = model.backbone[0]
        if hasattr(backbone0, "rgb_backbone"):
            projector_scale = list(getattr(backbone0.rgb_backbone, "projector_scale", None) or [])
        else:
            projector_scale = list(getattr(backbone0, "projector_scale", None) or [])
    except Exception:
        projector_scale = []
    p4_index = projector_scale.index("P4") if "P4" in projector_scale else None
    if any(k.startswith("backbone.0.rgb_backbone.projector.") for k in model_state.keys()):
        projector_prefix = "backbone.0.rgb_backbone.projector."
    else:
        projector_prefix = "backbone.0.projector."

    def _maybe_remap_key(key: str) -> str:
        if key in model_state:
            return key
        if not key.startswith("backbone.0."):
            return key

        # projector 的 stage 映射需优先处理（多尺度模型避免把 ckpt 的 P4(stage0) 错载到 P3(stage0)）
        if key.startswith("backbone.0.projector."):
            rel = key[len("backbone.0.projector.") :]
            if ckpt_stage_count == 1 and p4_index not in (None, 0) and len(projector_scale) > 1:
                rel = re.sub(r"^(stages(?:_sampling)?)\.0\.", rf"\1.{p4_index}.", rel, count=1)
            remapped = projector_prefix + rel
            if remapped in model_state:
                return remapped

        remapped = "backbone.0.rgb_backbone." + key[len("backbone.0.") :]
        if remapped in model_state:
            return remapped

        return key

    filtered: dict[str, torch.Tensor] = {}
    skipped: list[str] = []

    for key, tensor in raw_state.items():
        if exclude and key.startswith(exclude):
            skipped.append(key)
            continue
        key_in_model = _maybe_remap_key(key)
        target = model_state.get(key_in_model)
        if target is None:
            skipped.append(key)
            continue
        if target.shape == tensor.shape:
            filtered[key_in_model] = tensor
            continue
        # 兼容 num_feature_levels 变化导致的 MS-DeformAttn 线性层输出维度变化
        can_expand = ("sampling_offsets" in key_in_model) or ("attention_weights" in key_in_model)
        if can_expand and tensor.ndim == target.ndim:
            if tensor.ndim == 2 and tensor.shape[1] == target.shape[1] and target.shape[0] % tensor.shape[0] == 0:
                rep = target.shape[0] // tensor.shape[0]
                filtered[key_in_model] = tensor.repeat(rep, 1)[: target.shape[0]]
                continue
            if tensor.ndim == 1 and target.shape[0] % tensor.shape[0] == 0:
                rep = target.shape[0] // tensor.shape[0]
                filtered[key_in_model] = tensor.repeat(rep)[: target.shape[0]]
                continue
        skipped.append(key)

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    return {
        "ckpt_total": len(raw_state),
        "used": len(filtered),
        "skipped": len(skipped),
        "missing": len(missing),
        "unexpected": len(unexpected),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args()
    cfg = _load_cfg(args)

    # 强制 CPU，避免依赖 GPU 环境；同时避免 init_device 内部触发 CUDA 逻辑。
    if getattr(cfg, "runtime", None) is None:
        cfg.runtime = {}
    cfg.runtime.device = "cpu"
    cfg.runtime.device_ids = []
    cfg.runtime.world_size = 1

    trainer = MsifdetrTrainer(cfg)
    trainer.init_device("cpu")
    model = trainer.build_model()

    pretrain = getattr(cfg.model, "pretrain_weights", None) if getattr(cfg, "model", None) is not None else None
    if not pretrain:
        raise SystemExit("config 中未设置 model.pretrain_weights，无法检查覆盖率。")
    ckpt_path = Path(str(pretrain)).expanduser()
    if not ckpt_path.is_file():
        raise SystemExit(f"预训练文件不存在：{ckpt_path}")

    exclude = getattr(cfg.model, "pretrain_exclude_keys", None) if getattr(cfg, "model", None) is not None else None
    stats = _count_pretrain_coverage(model=model, ckpt_path=ckpt_path, exclude_keys=exclude)

    print("=== Pretrain Coverage ===")
    print(f"config={args.config or args.resolved_config}")
    print(f"ckpt={ckpt_path}")
    print(
        "ckpt_total={ckpt_total} used={used} skipped={skipped} missing={missing} unexpected={unexpected}".format(
            **stats
        )
    )


if __name__ == "__main__":
    main()
