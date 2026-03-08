#!/usr/bin/env python3
"""
Inspect a multispectral image file and report its true channel count.

Why this exists:
- Standard PNG only supports 1/2/3/4 samples per pixel (channels). It cannot store 16 channels directly.
- Some datasets store "16-channel multispectral" as a *single-channel mosaicked raw* image (e.g. a 4x4 MSFA pattern).
  In that case, you can unpack it into 16 bands by taking every N-th pixel.

Examples:
  python tools/inspect_msi_file.py path/to/0001.png
  python tools/inspect_msi_file.py path/to/0001.png --expect-ch 16 --mosaic 4 --save-vis out/bands --save-npz out/msi_0001.npz
  python tools/inspect_msi_file.py path/to/msi_dir --recursive --max-files 20 --expect-ch 16
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from PIL import Image

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None  # type: ignore

try:
    import tifffile  # type: ignore
except Exception:
    tifffile = None  # type: ignore


PNG_SIG = b"\x89PNG\r\n\x1a\n"


@dataclass(frozen=True)
class PngIhdr:
    width: int
    height: int
    bit_depth: int
    color_type: int
    compression: int
    filter_method: int
    interlace: int

    @property
    def channels(self) -> Optional[int]:
        # PNG color types:
        # 0: grayscale, 2: truecolor(RGB), 3: indexed-color, 4: grayscale+alpha, 6: truecolor+alpha(RGBA)
        return {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(int(self.color_type))


def _is_png(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return f.read(8) == PNG_SIG
    except OSError:
        return False


def _is_tiff(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            sig = f.read(4)
        return sig in {b"II*\x00", b"MM\x00*"}
    except OSError:
        return False


def _read_png_ihdr(path: Path) -> PngIhdr:
    with path.open("rb") as f:
        sig = f.read(8)
        if sig != PNG_SIG:
            raise ValueError(f"Not a PNG file: {path}")
        length = struct.unpack(">I", f.read(4))[0]
        ctype = f.read(4)
        if ctype != b"IHDR" or length != 13:
            raise ValueError(f"Invalid PNG IHDR: {path} (type={ctype!r} len={length})")
        data = f.read(length)
        width, height, bit_depth, color_type, compression, filter_method, interlace = struct.unpack(
            ">IIBBBBB", data
        )
        return PngIhdr(
            width=int(width),
            height=int(height),
            bit_depth=int(bit_depth),
            color_type=int(color_type),
            compression=int(compression),
            filter_method=int(filter_method),
            interlace=int(interlace),
        )


def _iter_png_chunks(path: Path) -> Iterable[tuple[str, int]]:
    with path.open("rb") as f:
        sig = f.read(8)
        if sig != PNG_SIG:
            return
        while True:
            raw_len = f.read(4)
            if not raw_len:
                return
            length = struct.unpack(">I", raw_len)[0]
            ctype = f.read(4)
            if len(ctype) != 4:
                return
            chunk_type = ctype.decode("ascii", errors="replace")
            # skip chunk data + crc
            f.seek(int(length) + 4, os.SEEK_CUR)
            yield chunk_type, int(length)
            if chunk_type == "IEND":
                return


def _safe_percentiles(x: np.ndarray, p_lo: float, p_hi: float) -> tuple[float, float]:
    if x.size == 0:
        return 0.0, 1.0
    flat = x.reshape(-1)
    lo = float(np.percentile(flat, p_lo))
    hi = float(np.percentile(flat, p_hi))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(flat))
        hi = float(np.max(flat))
        if hi <= lo:
            hi = lo + 1.0
    return lo, hi


def _to_uint8(x: np.ndarray, *, p_lo: float = 1.0, p_hi: float = 99.0) -> np.ndarray:
    x = np.asarray(x)
    lo, hi = _safe_percentiles(x, p_lo, p_hi)
    y = (x.astype(np.float32) - lo) / (hi - lo)
    y = np.clip(y, 0.0, 1.0)
    return (y * 255.0 + 0.5).astype(np.uint8)


def _unpack_mosaic(raw_hw: np.ndarray, mosaic: int) -> np.ndarray:
    if raw_hw.ndim != 2:
        raise ValueError(f"mosaic unpack expects a 2D array, got shape={raw_hw.shape}")
    h, w = (int(raw_hw.shape[0]), int(raw_hw.shape[1]))
    m = int(mosaic)
    if m <= 0:
        raise ValueError(f"mosaic must be positive, got: {mosaic}")
    if h % m != 0 or w % m != 0:
        raise ValueError(f"image size must be divisible by mosaic={m}: got HxW={h}x{w}")
    bands = [raw_hw[r::m, c::m] for r in range(m) for c in range(m)]
    return np.stack(bands, axis=0)  # [C, H/m, W/m]


def _fmt_stats(x: np.ndarray) -> str:
    x = np.asarray(x)
    if x.size == 0:
        return "empty"
    return (
        f"shape={tuple(int(s) for s in x.shape)} dtype={x.dtype} "
        f"min={float(np.min(x)):.6g} max={float(np.max(x)):.6g} "
        f"mean={float(np.mean(x)):.6g} std={float(np.std(x)):.6g}"
    )


def _read_via_pil(path: Path) -> tuple[np.ndarray, str]:
    with Image.open(path) as img:
        img.load()
        arr = np.array(img)
        mode = str(img.mode)
    return arr, mode


def _read_via_cv2(path: Path) -> Optional[np.ndarray]:
    if cv2 is None:
        return None
    try:
        arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        return arr
    except Exception:
        return None


def _read_via_tifffile(path: Path) -> Optional[np.ndarray]:
    if tifffile is None:
        return None
    try:
        return tifffile.imread(str(path))
    except Exception:
        return None


def _iter_input_files(root: Path, *, recursive: bool) -> list[Path]:
    if root.is_file():
        return [root]
    if not root.is_dir():
        raise FileNotFoundError(f"Path not found: {root}")

    suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff", ".npy", ".npz"}
    it = root.rglob("*") if recursive else root.glob("*")
    return sorted(p for p in it if p.is_file() and p.suffix.lower() in suffixes)


def analyze_file(
    path: Path,
    *,
    expect_ch: Optional[int],
    mosaic: Optional[int],
    save_vis: Optional[Path],
    save_npz: Optional[Path],
) -> None:
    size_bytes = path.stat().st_size
    print(f"\n=== {path} ===")
    print(f"file_size: {size_bytes} bytes")

    if _is_png(path):
        ihdr = _read_png_ihdr(path)
        print(
            "png_ihdr:",
            f"size={ihdr.width}x{ihdr.height}",
            f"bit_depth={ihdr.bit_depth}",
            f"color_type={ihdr.color_type}",
            f"channels={ihdr.channels}",
            f"interlace={ihdr.interlace}",
        )
        chunks = list(_iter_png_chunks(path))
        if chunks:
            top = ", ".join(f"{t}({n})" for t, n in chunks[:20])
            suffix = " ..." if len(chunks) > 20 else ""
            print(f"png_chunks: {top}{suffix}")

    if _is_tiff(path) and tifffile is not None:
        arr_tif = _read_via_tifffile(path)
        if arr_tif is not None:
            print("tifffile:", _fmt_stats(arr_tif))

    pil_arr, pil_mode = _read_via_pil(path)
    print("PIL:", f"mode={pil_mode}", _fmt_stats(pil_arr))

    cv2_arr = _read_via_cv2(path)
    if cv2_arr is not None:
        print("cv2:", _fmt_stats(cv2_arr))

    # Determine "stored channels" as what typical readers return.
    stored_ch = 1 if pil_arr.ndim == 2 else int(pil_arr.shape[2])
    if expect_ch is not None:
        exp = int(expect_ch)
        if stored_ch == exp:
            print(f"expect_ch={exp}: OK (stored channels match)")
        else:
            print(f"expect_ch={exp}: MISMATCH (stored_ch={stored_ch})")
            if stored_ch <= 4 and path.suffix.lower() == ".png" and exp > 4:
                print(
                    "note: standard PNG cannot store >4 channels; if this is MSFA mosaicked raw, "
                    "try --mosaic 4 (for 16 bands)."
                )

    if mosaic is not None:
        m = int(mosaic)
        if pil_arr.ndim == 3 and pil_arr.shape[2] == 1:
            raw_hw = pil_arr[..., 0]
        elif pil_arr.ndim == 2:
            raw_hw = pil_arr
        else:
            raise ValueError(
                f"--mosaic expects a single-channel image (2D or HWC with C=1), got shape={pil_arr.shape}"
            )
        bands = _unpack_mosaic(raw_hw, m)
        print(f"mosaic_unpack: mosaic={m} -> bands={bands.shape} (C,H,W)")
        for i in range(min(int(bands.shape[0]), 16)):
            print(f"  band[{i:02d}]:", _fmt_stats(bands[i]))

        if save_npz is not None:
            save_npz.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(str(save_npz), bands=bands)
            print(f"saved_npz: {save_npz}")

        if save_vis is not None:
            save_vis.mkdir(parents=True, exist_ok=True)
            grid_rows = m
            grid_cols = m
            band_h, band_w = int(bands.shape[1]), int(bands.shape[2])
            grid = np.zeros((grid_rows * band_h, grid_cols * band_w), dtype=np.uint8)
            for r in range(grid_rows):
                for c in range(grid_cols):
                    idx = r * grid_cols + c
                    tile = _to_uint8(bands[idx])
                    y0, y1 = r * band_h, (r + 1) * band_h
                    x0, x1 = c * band_w, (c + 1) * band_w
                    grid[y0:y1, x0:x1] = tile
                    Image.fromarray(tile).save(save_vis / f"band_{idx:02d}.png")
            Image.fromarray(grid).save(save_vis / "bands_grid.png")
            print(f"saved_vis: {save_vis} (band_*.png + bands_grid.png)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect multispectral image files (PNG/TIFF/etc) and channel layouts.")
    parser.add_argument("path", type=Path, help="An image file or a directory.")
    parser.add_argument("--recursive", action="store_true", help="If path is a directory, scan recursively.")
    parser.add_argument("--max-files", type=int, default=0, help="Limit number of scanned files (0 = no limit).")
    parser.add_argument("--expect-ch", type=int, default=None, help="Expected channel count (e.g., 16).")
    parser.add_argument(
        "--mosaic",
        type=int,
        default=None,
        help="Unpack a single-channel mosaicked raw image into mosaic*mosaic bands (e.g., 4 -> 16).",
    )
    parser.add_argument("--save-vis", type=Path, default=None, help="Output directory to save per-band PNGs + grid.")
    parser.add_argument("--save-npz", type=Path, default=None, help="Save unpacked bands to a compressed .npz.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    files = _iter_input_files(Path(args.path), recursive=bool(args.recursive))
    if not files:
        print(f"No supported image files found under: {args.path}")
        return 2

    limit = int(args.max_files)
    if limit > 0:
        files = files[:limit]

    for p in files:
        analyze_file(
            p,
            expect_ch=args.expect_ch,
            mosaic=args.mosaic,
            save_vis=args.save_vis,
            save_npz=args.save_npz,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

