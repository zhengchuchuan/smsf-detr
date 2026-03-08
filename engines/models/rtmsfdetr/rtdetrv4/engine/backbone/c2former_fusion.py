from __future__ import annotations

"""
Deprecated module path.

The implementation was renamed/moved to:
- `adaptive_sampling_ms_fusion.py` (ASMF: Adaptive Sampling Multispectral Fusion)

This file is kept as a thin re-export layer to avoid breaking older imports/configs.
"""

# Re-export everything for backward compatibility.
from .adaptive_sampling_ms_fusion import *  # noqa: F403

