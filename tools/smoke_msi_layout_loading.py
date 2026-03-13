#!/usr/bin/env python3
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import tifffile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.multispectral_coco import _load_msi_as_tensor  # noqa: E402


def _assert_equal(name: str, actual: np.ndarray, expected: np.ndarray) -> None:
    if actual.shape != expected.shape:
        raise AssertionError(f"{name}: shape mismatch, actual={actual.shape}, expected={expected.shape}")
    if not np.array_equal(actual, expected):
        diff = float(np.mean(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))
        raise AssertionError(f"{name}: values mismatch, mean_abs_diff={diff}")
    print(f"[PASS] {name}: shape={actual.shape}")


def main() -> int:
    # Square CHW TIFF is the regression case from 3c1c9ce:
    # the old heuristic mistakenly treated [C, H, W] as CWH when H >= W,
    # which silently transposed MSI relative to RGB on 640x640 oil tiles.
    c = 7
    square_h, square_w = 8, 8
    rect_h, rect_w = 10, 12
    ref_square_chw = np.arange(c * square_h * square_w, dtype=np.uint16).reshape(c, square_h, square_w)
    ref_rect_chw = np.arange(c * rect_h * rect_w, dtype=np.uint16).reshape(c, rect_h, rect_w)
    ref_rect_cwh = np.transpose(ref_rect_chw, (0, 2, 1))

    with tempfile.TemporaryDirectory(prefix="smsf_msi_layout_") as tmp_dir:
        tmp = Path(tmp_dir)

        tif_path = tmp / "sample_chw.tif"
        tifffile.imwrite(str(tif_path), ref_square_chw)
        tif_tensor = _load_msi_as_tensor(tif_path, expected_channels=c, npy_layout="auto")
        _assert_equal("tif auto keeps CHW semantics", tif_tensor.numpy(), ref_square_chw.astype(np.float32))

        npy_chw_path = tmp / "sample_chw.npy"
        np.save(npy_chw_path, ref_rect_chw)
        npy_chw_tensor = _load_msi_as_tensor(npy_chw_path, expected_channels=c, npy_layout="auto")
        _assert_equal("npy auto supports CHW", npy_chw_tensor.numpy(), ref_rect_chw.astype(np.float32))

        npy_cwh_path = tmp / "sample_cwh.npy"
        np.save(npy_cwh_path, ref_rect_cwh)
        npy_cwh_tensor = _load_msi_as_tensor(npy_cwh_path, expected_channels=c, npy_layout="auto")
        _assert_equal("npy auto supports CWH", npy_cwh_tensor.numpy(), ref_rect_chw.astype(np.float32))

        npz_cwh_path = tmp / "sample_cwh.npz"
        np.savez(npz_cwh_path, bands=ref_rect_cwh)
        npz_cwh_tensor = _load_msi_as_tensor(npz_cwh_path, expected_channels=c, npy_layout="auto")
        _assert_equal("npz auto supports CWH", npz_cwh_tensor.numpy(), ref_rect_chw.astype(np.float32))

    print("All MSI layout checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
