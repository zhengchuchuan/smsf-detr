from __future__ import annotations

from typing import Any

import torch


def _infer_batch_size(samples: Any) -> int | None:
    if samples is None:
        return None
    if isinstance(samples, dict):
        if not samples:
            return None
        return _infer_batch_size(next(iter(samples.values())))
    if hasattr(samples, "tensors") and torch.is_tensor(getattr(samples, "tensors")):
        tensors = getattr(samples, "tensors")
        if tensors.ndim >= 1:
            return int(tensors.shape[0])
        return None
    if torch.is_tensor(samples):
        if samples.ndim >= 1:
            return int(samples.shape[0])
        return None
    return None


def try_compute_gflops(model: torch.nn.Module, samples: Any, *, per_image: bool = True) -> float | None:
    """
    Try computing FLOPs (GFLOPs) for a forward pass.

    Notes:
    - FLOPs depend on the input spatial resolution; pass representative `samples`.
    - This returns GFLOPs (1e9 FLOPs). When `per_image=True`, divides by batch size.
    """
    batch_size = _infer_batch_size(samples)

    total_flops: int | None = None
    try:
        from torch.utils.flop_counter import FlopCounterMode

        with FlopCounterMode(display=False) as flop_counter:
            _ = model(samples)
        total_flops = int(flop_counter.get_total_flops())
    except Exception:
        total_flops = None

    if total_flops is None:
        try:
            from thop import profile as thop_profile  # type: ignore

            macs, _params = thop_profile(model, inputs=(samples,), verbose=False)
            total_flops = int(macs * 2)
        except Exception:
            total_flops = None

    if total_flops is None or total_flops <= 0:
        return None

    gflops = float(total_flops) / 1e9
    if per_image and batch_size and batch_size > 0:
        gflops /= float(batch_size)
    return gflops

