#!/usr/bin/env python3
"""
读取 checkpoint 中的融合门控参数（fusion_alpha）。

用法：
  python tools/inspect_ckpt_fusion_alpha.py /path/to/checkpoint_best.pth
  python tools/inspect_ckpt_fusion_alpha.py outputs/.../checkpoint_best.pth --list
  python tools/inspect_ckpt_fusion_alpha.py outputs/.../checkpoint_best.pth --pattern fusion_alpha
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping


def _torch_load(path: Path) -> Any:
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "未找到 torch，无法读取 .pth checkpoint。请在训练环境中运行该脚本。"
        ) from exc

    # PyTorch 2.6+ 默认 weights_only=True；若 checkpoint 里包含 OmegaConf/Hydra 对象，
    # 需要在 safe_globals 中显式 allowlist 相关类型才能安全加载。
    safe_globals_ctx = getattr(torch.serialization, "safe_globals", None)
    if safe_globals_ctx is None:
        return torch.load(path, map_location="cpu")

    allowed = [argparse.Namespace]
    try:
        import typing

        allowed.append(typing.Any)
    except Exception:
        pass
    try:
        from omegaconf import DictConfig, ListConfig  # type: ignore

        allowed.extend([DictConfig, ListConfig])
    except Exception:
        pass
    try:
        from omegaconf.base import ContainerMetadata  # type: ignore

        allowed.append(ContainerMetadata)
    except Exception:
        pass

    try:
        from pathlib import PosixPath, WindowsPath

        allowed.extend([PosixPath, WindowsPath])
    except Exception:
        pass

    with safe_globals_ctx(allowed):
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            # 兼容旧版本 torch.load 不支持 weights_only
            return torch.load(path, map_location="cpu")


def _extract_state(obj: Any) -> Mapping[str, Any]:
    if isinstance(obj, Mapping):
        if "model" in obj and isinstance(obj["model"], Mapping):
            return obj["model"]
        if "state_dict" in obj and isinstance(obj["state_dict"], Mapping):
            return obj["state_dict"]
        # 直接就是 state_dict
        if obj and all(isinstance(k, str) for k in obj.keys()):
            return obj
    raise TypeError(f"无法从 checkpoint 解析 state_dict，类型为 {type(obj)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt", type=str, help="checkpoint 路径（.pth/.pt）")
    parser.add_argument(
        "--pattern",
        type=str,
        default="fusion_alpha",
        help="筛选的参数名子串（默认：fusion_alpha）",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出所有匹配的 key（不只打印数值）",
    )
    parser.add_argument(
        "--unsafe",
        action="store_true",
        help="使用 weights_only=False 加载（仅在你完全信任 checkpoint 来源时使用）。",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"未找到 checkpoint：{ckpt_path}")

    if args.unsafe:
        import torch

        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    else:
        raw = _torch_load(ckpt_path)
    state = _extract_state(raw)

    pattern = str(args.pattern)
    matches = sorted((k, v) for k, v in state.items() if pattern in k)
    if not matches:
        print(f"[miss] 未找到包含 '{pattern}' 的参数。")
        # 提供一些常见候选提示
        hints = [k for k in state.keys() if "alpha" in k or "fusion" in k]
        print(f"[hint] 发现 {len(hints)} 个包含 alpha/fusion 的 key，可用 --pattern 进一步筛选。")
        if hints[:30]:
            for k in sorted(hints)[:30]:
                print(" -", k)
        return 2

    print(f"[ok] checkpoint: {ckpt_path}")
    print(f"[ok] matched: {len(matches)} (pattern='{pattern}')")

    for k, v in matches:
        if args.list:
            shape = getattr(v, "shape", None)
            dtype = getattr(v, "dtype", None)
            print(f"{k} | shape={shape} dtype={dtype}")
            continue

        # 尝试打印标量或统计信息
        try:
            import torch

            if torch.is_tensor(v):
                if v.numel() == 1:
                    print(f"{k} = {float(v.detach().cpu().item()):.6f}")
                else:
                    vv = v.detach().float().cpu()
                    print(
                        f"{k} | shape={tuple(v.shape)} mean={vv.mean().item():.6f} std={vv.std().item():.6f} "
                        f"min={vv.min().item():.6f} max={vv.max().item():.6f}"
                    )
            else:
                print(f"{k} = {v}")
        except Exception:
            print(f"{k} = {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
