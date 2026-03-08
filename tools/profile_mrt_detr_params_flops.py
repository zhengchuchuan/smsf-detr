#!/usr/bin/env python
"""
Compute parameter count + (approx) FLOPs/MACs for MRT-DETR (third_party/MRT-DETR).

Notes
-----
- Parameter count can be computed from the instantiated model (exact).
- FLOPs depend on input shape and execution path; we estimate compute using profilers.
- This script defaults to disabling `pretrained: True` in the YAML to avoid downloads.

Examples
--------
python tools/profile_mrt_detr_params_flops.py \
  --config third_party/MRT-DETR/rtdetr_pytorch/configs/rtdetr/rtdetr_dual.yml \
  --img-size 640 640
"""

from __future__ import annotations

import argparse
import inspect
import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch


def _human_count(n: float) -> str:
    units = ["", "K", "M", "G", "T", "P"]
    v = float(n)
    for u in units:
        if abs(v) < 1000.0:
            return f"{v:.3f}{u}"
        v /= 1000.0
    return f"{v:.3f}E"


def _count_params(model: torch.nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _set_key_recursive(obj: Any, key: str, value: Any) -> None:
    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            if k == key:
                obj[k] = value
            else:
                _set_key_recursive(v, key, value)
    elif isinstance(obj, list):
        for v in obj:
            _set_key_recursive(v, key, value)


def _infer_required_positional_inputs(model: torch.nn.Module) -> int:
    sig = inspect.signature(model.forward)
    params = [p for p in sig.parameters.values() if p.name != "self"]
    required = [
        p
        for p in params
        if p.default is inspect._empty
        and p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    return len(required)


def _parse_int_list(s: str) -> List[int]:
    items = [x.strip() for x in s.split(",") if x.strip()]
    return [int(x) for x in items]


def _make_image_inputs(
    *,
    n_inputs: int,
    batch_size: int,
    channels: Sequence[int],
    img_size: Tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, ...]:
    h, w = img_size
    if len(channels) == 1:
        channels = list(channels) * n_inputs
    if len(channels) != n_inputs:
        raise ValueError(f"--channels expects 1 or {n_inputs} values, got {len(channels)}")

    return tuple(
        torch.randn(batch_size, int(ch), h, w, device=device, dtype=dtype) for ch in channels
    )


def _load_checkpoint_into_model(model: torch.nn.Module, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Common patterns seen in checkpoints.
    if isinstance(ckpt, dict):
        for key in ["model", "state_dict", "ema", "model_state", "net"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break

    if not isinstance(ckpt, dict):
        raise ValueError(f"Unsupported checkpoint format: {type(ckpt)}")

    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    if missing:
        print(f"[warn] missing keys: {len(missing)}")
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)}")


def _profile_thop(model: torch.nn.Module, inputs: Tuple[torch.Tensor, ...]) -> Tuple[float, float]:
    from thop import profile

    macs, params = profile(model, inputs=inputs, verbose=False)
    return float(macs), float(params)


def _profile_ptflops(
    model: torch.nn.Module, inputs: Tuple[torch.Tensor, ...], img_size: Tuple[int, int]
) -> Tuple[float, float]:
    """
    ptflops's public API is oriented around single-input models, but supports input_constructor.
    We return: (macs_or_flops, params) as floats.
    """

    from ptflops import get_model_complexity_info

    c0 = int(inputs[0].shape[1])
    h, w = img_size

    def _input_constructor(_: Any) -> Dict[str, Any]:
        # ptflops calls model(*inputs) when as_strings=False, so we provide positional args.
        return {"input": inputs}

    # When `as_strings=False`, ptflops returns raw numbers (not formatted strings).
    macs, params = get_model_complexity_info(
        model,
        (c0, h, w),
        as_strings=False,
        print_per_layer_stat=False,
        verbose=False,
        input_constructor=_input_constructor,
    )
    return float(macs), float(params)


def _build_mrt_detr_from_yaml(
    cfg_path: str,
    *,
    disable_pretrained: bool,
    eval_spatial_size: Optional[Tuple[int, int]],
    input_channels: Optional[Sequence[int]] = None,
) -> torch.nn.Module:
    repo = os.path.abspath("third_party/MRT-DETR/rtdetr_pytorch")
    if not os.path.isdir(repo):
        raise FileNotFoundError(f"Missing MRT-DETR repo at: {repo}")

    sys.path.insert(0, repo)

    # Import registers all @register modules into src.core.GLOBAL_CONFIG.
    import src  # noqa: F401
    from src.core import YAMLConfig

    cfg = YAMLConfig(cfg_path)
    if disable_pretrained:
        _set_key_recursive(cfg.yaml_cfg, "pretrained", False)
    # Many configs bake positional embeddings / anchors using `eval_spatial_size`.
    # For profiling, keep it consistent with the dummy input resolution.
    if eval_spatial_size is not None:
        h, w = eval_spatial_size
        _set_key_recursive(cfg.yaml_cfg, "eval_spatial_size", [int(h), int(w)])

    # MRT-DETR RTDETR uses two injected backbones: `backbone` and `backbone_ir`.
    # If the caller asks for different channel counts per input (e.g. RGB + MSI),
    # patch the injection configs to pass `in_channels` / `in_chans`.
    if input_channels is not None and len(input_channels) >= 1:
        model_name = cfg.yaml_cfg.get("model", None)
        if model_name == "RTDETR" and isinstance(cfg.yaml_cfg.get("RTDETR", None), dict):
            mcfg = cfg.yaml_cfg["RTDETR"]
            c0 = int(input_channels[0])
            c1 = int(input_channels[1]) if len(input_channels) >= 2 else int(input_channels[0])

            # Convert "backbone": "PResNet" to dict injection, so we can set in_channels per-branch.
            if "backbone" in mcfg:
                bb = mcfg["backbone"]
                if isinstance(bb, str):
                    mcfg["backbone"] = {"type": bb}
                if isinstance(mcfg["backbone"], dict):
                    mcfg["backbone"]["in_channels"] = c0
                    mcfg["backbone"]["in_chans"] = c0

            if "backbone_ir" in mcfg:
                bb = mcfg["backbone_ir"]
                if isinstance(bb, str):
                    mcfg["backbone_ir"] = {"type": bb}
                if isinstance(mcfg["backbone_ir"], dict):
                    mcfg["backbone_ir"]["in_channels"] = c1
                    mcfg["backbone_ir"]["in_chans"] = c1

    return cfg.model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        required=True,
        help="MRT-DETR yaml path, e.g. third_party/MRT-DETR/rtdetr_pytorch/configs/rtdetr/rtdetr_dual.yml",
    )
    ap.add_argument("--checkpoint", default=None, help="Optional checkpoint to load (does not change FLOPs).")
    ap.add_argument("--device", default="cpu", help="cpu | cuda:0 | ...")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument(
        "--img-size",
        type=int,
        nargs=2,
        default=[640, 640],
        metavar=("H", "W"),
        help="Input image size (H W).",
    )
    ap.add_argument(
        "--n-inputs",
        type=int,
        default=0,
        help="Number of required image inputs to forward(). 0 = auto infer (e.g. dual-input RTDETR => 2).",
    )
    ap.add_argument(
        "--channels",
        type=_parse_int_list,
        default=[3],
        help="Comma-separated channels per input; either 1 value (broadcast) or n_inputs values. "
        "Examples: '3' or '3,3'.",
    )
    ap.add_argument(
        "--method",
        default="thop",
        choices=["thop", "ptflops"],
        help="Profiler backend. thop reports MACs (multiply-accumulate).",
    )
    ap.add_argument(
        "--keep-pretrained",
        action="store_true",
        help="Do not override `pretrained: True` in the yaml. "
        "Warning: this may trigger downloads and fail in restricted-network environments.",
    )
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]

    # Pre-build channel list so we can patch config before model instantiation.
    # For dual-input RTDETR, a single value means "both inputs use the same channels".
    requested_channels = list(args.channels)
    if len(requested_channels) == 1:
        requested_channels = requested_channels * 2

    model = _build_mrt_detr_from_yaml(
        args.config,
        disable_pretrained=not args.keep_pretrained,
        eval_spatial_size=(int(args.img_size[0]), int(args.img_size[1])),
        input_channels=requested_channels,
    )
    if args.checkpoint:
        _load_checkpoint_into_model(model, args.checkpoint)

    model.to(device=device)
    model.eval()

    n_inputs = int(args.n_inputs) if int(args.n_inputs) > 0 else _infer_required_positional_inputs(model)
    if n_inputs <= 0:
        raise ValueError("Could not infer number of required forward() inputs; pass --n-inputs explicitly.")

    channels = list(args.channels)
    if len(channels) == 1 and n_inputs > 1:
        channels = channels * n_inputs

    inputs = _make_image_inputs(
        n_inputs=n_inputs,
        batch_size=int(args.batch_size),
        channels=channels,
        img_size=(int(args.img_size[0]), int(args.img_size[1])),
        device=device,
        dtype=dtype,
    )

    with torch.no_grad():
        # Quick sanity check that forward runs before profiling (gives clearer errors than profiler internals).
        try:
            _ = model(*inputs)
        except RuntimeError as e:
            msg = str(e)
            print("[error] model forward failed; cannot profile.")
            print(f"[error] {type(e).__name__}: {msg}")
            if "selected index k out of range" in msg:
                print(
                    "[hint] This often means `num_queries` is larger than the number of encoder locations "
                    "for your `--img-size`. Try a larger resolution (e.g. 640x640) or reduce `num_queries` "
                    "in the YAML."
                )
            raise

        if args.method == "thop":
            macs, _ = _profile_thop(model, inputs)
            macs_s = _human_count(macs)
            flops_s = _human_count(2.0 * macs)
            compute_line = f"MACs(thop): {macs_s}  |  approx FLOPs(2*MACs): {flops_s}"
        else:
            macs_or_flops, _ = _profile_ptflops(model, inputs, (int(args.img_size[0]), int(args.img_size[1])))
            compute_line = f"ptflops: { _human_count(macs_or_flops) } (see ptflops docs for exact definition)"

    total_p, trainable_p = _count_params(model)

    print(f"config: {args.config}")
    if args.checkpoint:
        print(f"checkpoint: {args.checkpoint}")
    print(f"device: {device} | dtype: {dtype} | batch: {args.batch_size} | img: {args.img_size[0]}x{args.img_size[1]}")
    print(f"inputs: n={n_inputs} | channels={channels}")
    print(f"params(total): {_human_count(total_p)} ({total_p})")
    print(f"params(trainable): {_human_count(trainable_p)} ({trainable_p})")
    print(compute_line)


if __name__ == "__main__":
    main()
