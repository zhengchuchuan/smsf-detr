from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engines.core.parse_config import load_config
from engines.trainer.msifdetr_train import MsifdetrTrainer
from utils.misc import NestedTensor


def _load_cfg(args: argparse.Namespace):
    if args.resolved_config:
        cfg_path = Path(args.resolved_config).expanduser().resolve()
        if not cfg_path.is_file():
            raise FileNotFoundError(f"--resolved-config 不存在: {cfg_path}")
        cfg = OmegaConf.load(str(cfg_path))
        OmegaConf.set_struct(cfg, False)
        return cfg

    if not args.config:
        raise ValueError("需要传入 --config 或 --resolved-config")
    overrides = list(args.opts or [])
    cfg = load_config(args.config, config_dir=args.config_dir, overrides=overrides)
    return cfg


def main():
    parser = argparse.ArgumentParser(description="MSIF-DETR backbone forward smoke test")
    parser.add_argument("--config", type=str, default=None, help="Hydra config path under configs/")
    parser.add_argument("--config-dir", type=str, default="configs", help="Hydra config root dir")
    parser.add_argument("--resolved-config", type=str, default=None, help="Resolved config.yaml path (from outputs/)")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--opts", nargs="*", default=None, help="Hydra dotlist overrides")
    args = parser.parse_args()

    cfg = _load_cfg(args)
    cfg.runtime = cfg.get("runtime", {})
    cfg.runtime["device"] = "cpu"
    cfg.runtime["device_ids"] = []
    cfg.runtime["world_size"] = 1

    trainer = MsifdetrTrainer(cfg)
    trainer.device = "cpu"
    model = trainer.build_model()
    model.eval()

    img_size = int(args.img_size)
    batch = int(args.batch)

    rgb_ch = int(cfg.get("data", {}).get("rgb_input_channels", 3))
    ms_ch = int(cfg.get("data", {}).get("ms_input_channels", cfg.get("data", {}).get("ms_expected_channels", 7)))

    rgb = torch.randn(batch, rgb_ch, img_size, img_size)
    ms = torch.randn(batch, ms_ch, img_size, img_size)
    mask = torch.zeros(batch, img_size, img_size, dtype=torch.bool)
    samples = {"rgb": NestedTensor(rgb, mask), "ms": NestedTensor(ms, mask)}

    with torch.no_grad():
        feats, pos = model.backbone(samples)

    print(f"num_feats={len(feats)} num_pos={len(pos)}")
    for i, (f, p) in enumerate(zip(feats, pos)):
        print(f"[{i}] feat={tuple(f.tensors.shape)} mask={tuple(f.mask.shape) if f.mask is not None else None} pos={tuple(p.shape)}")


if __name__ == "__main__":
    main()
