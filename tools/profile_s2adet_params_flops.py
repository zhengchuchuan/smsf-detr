#!/usr/bin/env python
"""
Profile S2ADet (third_party/S2ADet) parameters and MACs/FLOPs for dual-input models.

This repo's S2ADet dual-stream forward is implemented in `third_party/S2ADet/models/yolo_test.py`,
where the second stream is selected via `from: -4` in the YAML graph.

We build the model from YAML here (without modifying vendored code) and allow per-stream input channels,
e.g. RGB=3, MSI=7.

Example:
  python tools/profile_s2adet_params_flops.py \
    --cfg third_party/S2ADet/models/hsi/yolov5s_fusion_transformerx3_hsi.yaml \
    --img-size 640 640 \
    --channels 3,7
"""

from __future__ import annotations

import argparse
import os
import sys
from copy import deepcopy
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn


def _human(n: float) -> str:
    units = ["", "K", "M", "G", "T"]
    v = float(n)
    for u in units:
        if abs(v) < 1000.0:
            return f"{v:.3f}{u}"
        v /= 1000.0
    return f"{v:.3f}P"


def _count_params(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _parse_channels(s: str) -> List[int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError("--channels must be like '3,7'")
    return [int(parts[0]), int(parts[1])]


class DualStreamModel(nn.Module):
    """YOLOv5-style graph runner with a second input for layers with `from: -4`."""

    def __init__(self, model: nn.Sequential, save: List[int], yaml_dict: Dict[str, Any]) -> None:
        super().__init__()
        self.model = model
        self.save = save
        self.yaml = yaml_dict

        m = self.model[-1]
        self.stride = getattr(m, "stride", torch.tensor([32.0]))

    def forward(self, x: torch.Tensor, x2: torch.Tensor) -> Any:
        y: List[Any] = []
        for m in self.model:
            if getattr(m, "f", -1) != -1:
                if m.f != -4:
                    x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]

            if m.f == -4:
                x = m(x2)
            else:
                x = m(x)

            y.append(x if getattr(m, "i", -1) in self.save else None)
        return x


def _build_s2adet_dual_from_yaml(
    cfg_path: str, *, ch_rgb: int, ch_ir: int, nc_override: int | None = None
) -> DualStreamModel:
    import yaml

    # Ensure S2ADet's own `models/` and `utils/` are imported (avoid clash with top-level repo packages).
    s2_root = os.path.abspath("third_party/S2ADet")
    sys.path.insert(0, s2_root)

    # Provide eval() namespace for parse_model (matches YOLOv5 style).
    # NOTE: can't use `import *` inside a function, so we import modules and eval with an explicit namespace.
    import importlib

    common = importlib.import_module("models.common")
    experimental = importlib.import_module("models.experimental")
    yolo_mod = importlib.import_module("models.yolo")
    utils_autoanchor = importlib.import_module("utils.autoanchor")
    utils_general = importlib.import_module("utils.general")
    utils_torch_utils = importlib.import_module("utils.torch_utils")

    ns: Dict[str, Any] = {}
    ns.update(common.__dict__)
    ns.update(experimental.__dict__)
    ns.update(yolo_mod.__dict__)  # provides Detect and some other YOLOv5 modules
    ns.update({"torch": torch, "nn": nn})

    check_anchor_order = utils_autoanchor.check_anchor_order
    make_divisible = utils_general.make_divisible
    initialize_weights = utils_torch_utils.initialize_weights

    # Common module classes referenced in YAMLs.
    Conv = common.Conv
    GhostConv = getattr(common, "GhostConv", None)
    Bottleneck = common.Bottleneck
    GhostBottleneck = getattr(common, "GhostBottleneck", None)
    SPP = common.SPP
    SPPF = getattr(common, "SPPF", None)
    DWConv = getattr(common, "DWConv", None)
    MixConv2d = getattr(common, "MixConv2d", None)
    Focus = common.Focus
    CrossConv = getattr(common, "CrossConv", None)
    BottleneckCSP = getattr(common, "BottleneckCSP", None)
    C3 = common.C3
    C3TR = getattr(common, "C3TR", None)
    Concat = common.Concat
    Detect = getattr(yolo_mod, "Detect", None)
    Contract = getattr(common, "Contract", None)
    Expand = getattr(common, "Expand", None)
    Add = getattr(common, "Add", None)
    Add2 = getattr(common, "Add2", None)
    GPT = getattr(common, "GPT", None)

    with open(cfg_path, "r") as f:
        d = yaml.safe_load(f)

    if nc_override is not None and int(nc_override) != int(d.get("nc", nc_override)):
        d["nc"] = int(nc_override)

    anchors, nc, gd, gw = d["anchors"], d["nc"], d["depth_multiple"], d["width_multiple"]
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors
    no = na * (nc + 5)

    eval_locals: Dict[str, Any] = {"anchors": anchors, "nc": nc, "no": no}

    layers: List[nn.Module] = []
    save: List[int] = []
    ch: List[int] = [int(ch_rgb)]  # track output channels per layer index

    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):
        m = eval(m, ns) if isinstance(m, str) else m  # noqa: S307
        for j, a in enumerate(args):
            try:
                args[j] = eval(a, ns, eval_locals) if isinstance(a, str) else a  # noqa: S307
            except Exception:
                pass

        n = max(round(n * gd), 1) if n > 1 else n

        blocks = [Conv, Bottleneck, SPP, Focus, C3]
        for maybe in [GhostConv, GhostBottleneck, SPPF, DWConv, MixConv2d, CrossConv, BottleneckCSP, C3TR]:
            if maybe is not None:
                blocks.append(maybe)

        if m in blocks:
            if m is Focus:
                # Dual-stream: the second stream starts with `from: -4` in these YAMLs.
                c1 = int(ch_ir) if f == -4 else (int(ch_rgb) if i == 0 else ch[f])
                c2 = args[0]
                if c2 != no:
                    c2 = make_divisible(c2 * gw, 8)  # noqa: F405
                args = [c1, c2, *args[1:]]
            else:
                c1, c2 = ch[f], args[0]
                if c2 != no:
                    c2 = make_divisible(c2 * gw, 8)  # noqa: F405
                args = [c1, c2, *args[1:]]
                repeats = [C3]
                for maybe in [BottleneckCSP, C3TR]:
                    if maybe is not None:
                        repeats.append(maybe)
                if m in repeats:
                    args.insert(2, n)
                    n = 1

        elif m is nn.BatchNorm2d:
            c2 = ch[f]
            args = [c2]
        elif Add is not None and m is Add:
            c2 = ch[f[0]]
            args = [c2]
        elif Add2 is not None and m is Add2:
            c2 = ch[f[0]]
            args = [c2, args[1]]
        elif GPT is not None and m is GPT:
            c2 = ch[f[0]]
            args = [c2, args[1], args[2], args[3]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif Detect is not None and m is Detect:
            c2 = no
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif Contract is not None and m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif Expand is not None and m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)
        np_ = sum(p.numel() for p in m_.parameters())
        m_.i, m_.f, m_.np = i, f, np_  # type: ignore[attr-defined]
        layers.append(m_)

        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        if i == 0:
            ch = []
        ch.append(int(c2))

    model = nn.Sequential(*layers)
    initialize_weights(model)

    # Match vendored yolo_test behavior: set fixed strides for Detect, normalize anchors, then check order.
    m_last = model[-1]
    if m_last.__class__.__name__ == "Detect":
        m_last.stride = torch.tensor([8.0, 16.0, 32.0])
        m_last.anchors /= m_last.stride.view(-1, 1, 1)
        check_anchor_order(m_last)

    return DualStreamModel(model, sorted(save), d)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cfg",
        required=True,
        help="S2ADet model yaml, e.g. third_party/S2ADet/models/hsi/yolov5s_fusion_transformerx3_hsi.yaml",
    )
    ap.add_argument("--channels", type=_parse_channels, default="3,7", help="Input channels as 'rgb,ir', e.g. 3,7.")
    ap.add_argument("--img-size", type=int, nargs=2, default=[640, 640], metavar=("H", "W"))
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--no-flops", action="store_true", help="Only print params; skip thop profiling.")
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    ch_rgb, ch_ir = int(args.channels[0]), int(args.channels[1])
    h, w = int(args.img_size[0]), int(args.img_size[1])

    model = _build_s2adet_dual_from_yaml(args.cfg, ch_rgb=ch_rgb, ch_ir=ch_ir).to(device=device).eval()
    total, trainable = _count_params(model)

    print(f"cfg: {args.cfg}")
    print(f"device: {device} | dtype: {dtype} | batch: {args.batch_size} | img: {h}x{w}")
    print(f"channels: rgb={ch_rgb}, ir={ch_ir}")
    print(f"params(total): {_human(total)} ({total})")
    print(f"params(trainable): {_human(trainable)} ({trainable})")

    if args.no_flops:
        return

    try:
        from thop import profile
    except Exception as e:
        raise SystemExit(f"thop not available: {e}")

    x1 = torch.randn(int(args.batch_size), ch_rgb, h, w, device=device, dtype=dtype)
    x2 = torch.randn(int(args.batch_size), ch_ir, h, w, device=device, dtype=dtype)

    with torch.no_grad():
        # Sanity forward (gives clearer errors than profiler internals).
        _ = model(x1, x2)

        macs, _ = profile(model, inputs=(x1, x2), verbose=False)
        macs = float(macs)
        flops = 2.0 * macs

    print(f"MACs(thop): {_human(macs)} ({macs:.0f})")
    print(f"approx FLOPs(2*MACs): {_human(flops)} ({flops:.0f})")


if __name__ == "__main__":
    main()
