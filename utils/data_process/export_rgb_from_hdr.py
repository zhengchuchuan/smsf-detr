#!/usr/bin/env python3

"""从对齐后的 HDR 立方体中分离导出 RGB 图像 + 7 通道光谱 TIF。"""

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
import tifffile

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.io.spectral_io import open_hdr_img


def iter_hdr_files(folder: Path) -> Iterable[Path]:
    return sorted(folder.glob("*.hdr"))


def to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    if np.issubdtype(image.dtype, np.integer):
        return np.clip(image, 0, 255).astype(np.uint8)
    finite = np.nan_to_num(image, nan=0.0, copy=False)
    max_val = finite.max()
    min_val = finite.min()
    if max_val <= min_val:
        return np.zeros_like(finite, dtype=np.uint8)
    scaled = (finite - min_val) / (max_val - min_val)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def export_rgb_and_spectral_from_hdr(
    input_hdr_dir: Path,
    rgb_output_dir: Optional[Path],
    spectral_tif_output_dir: Optional[Path],
    rgb_ext: str,
    overwrite: bool,
    spectral_channel_names: Optional[list[str]] = None,
) -> tuple[int, int]:
    if not input_hdr_dir.is_dir():
        raise FileNotFoundError(f"输入目录不存在: {input_hdr_dir}")

    if rgb_output_dir is not None:
        rgb_output_dir.mkdir(parents=True, exist_ok=True)
    if spectral_tif_output_dir is not None:
        spectral_tif_output_dir.mkdir(parents=True, exist_ok=True)

    hdr_files = list(iter_hdr_files(input_hdr_dir))
    if not hdr_files:
        print(f"未在目录中找到HDR文件: {input_hdr_dir}")
        return

    iterator: Iterable[Path] = hdr_files
    if tqdm is not None:
        iterator = tqdm(hdr_files, total=len(hdr_files), desc="Export", unit="img")

    rgb_written = 0
    tif_written = 0

    for hdr_path in iterator:
        img_cube = open_hdr_img(str(hdr_path))
        if img_cube is None:
            print(f"跳过无法读取的HDR: {hdr_path.name}")
            continue
        if img_cube.ndim != 3:
            print(f"跳过维度不正确的HDR({img_cube.shape}): {hdr_path.name}")
            continue

        h, w, c = img_cube.shape
        if c < 3:
            print(f"跳过通道数不足的HDR({img_cube.shape}): {hdr_path.name}")
            continue

        # RGB export always uses the first 3 channels as BGR.
        # - For 10ch cubes: first 3 are BGR placeholders.
        # - For 7ch cubes: first 3 are typically 450/550/650nm, which map naturally to B/G/R.
        bgr = img_cube[:, :, :3]

        spectral7: Optional[np.ndarray] = None
        if c >= 10:
            spectral7 = img_cube[:, :, 3:10]
        elif c >= 7:
            spectral7 = img_cube[:, :, -7:]

        # Export RGB (if present).
        # NOTE: The aligned HDR produced by this repo stores the first 3 channels as BGR.
        # Write it directly to avoid cvtColor on non-contiguous views (which can corrupt output).
        if rgb_output_dir is not None:
            bgr_u8 = np.ascontiguousarray(to_uint8(bgr))
            rgb_output_path = rgb_output_dir / f"{hdr_path.stem}.{rgb_ext}"
            if rgb_output_path.exists() and not overwrite:
                pass
            else:
                ok = cv2.imwrite(str(rgb_output_path), bgr_u8)
                if not ok:
                    raise RuntimeError(f"写入失败: {rgb_output_path}")
                rgb_written += 1

        # Export spectral 7ch TIF (HxWx7 -> planar 7xHxW).
        if spectral_tif_output_dir is None or spectral7 is None:
            continue

        tif_output_path = spectral_tif_output_dir / f"{hdr_path.stem}.tif"
        if tif_output_path.exists() and not overwrite:
            continue

        spectral7 = np.ascontiguousarray(spectral7)
        planar = np.ascontiguousarray(spectral7.transpose(2, 0, 1))
        metadata = {"axes": "SYX"}
        if spectral_channel_names is not None and len(spectral_channel_names) == 7:
            metadata["ChannelNames"] = spectral_channel_names
        tifffile.imwrite(
            str(tif_output_path),
            planar,
            dtype=planar.dtype,
            photometric="MINISBLACK",
            planarconfig="SEPARATE",
            metadata=metadata,
        )
        tif_written += 1

    return rgb_written, tif_written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="批量读取对齐后的 HDR（通常 10 通道）并导出 RGB 图像 + 7 通道光谱 TIF。"
    )
    parser.add_argument(
        "--input-hdr-dir",
        type=Path,
        default="/mnt/d/Project/master-graduation-project/data/oil/val/aligned_full_tif_20251231-2225_matchanything_affine_1440",
        help="包含 HDR 文件的目录（来自配准导出的 .hdr/.img）。",
    )
    parser.add_argument(
        "--output-rgb-dir",
        type=Path,
        default="/mnt/d/Project/master-graduation-project/master-graduation/data/oil/val/aligned_full_tif_20251231-2225_matchanything_affine_1440_crop",
        help="RGB 图像输出目录。",
    )
    parser.add_argument(
        "--output-spectral-tif-dir",
        type=Path,
        default="/mnt/d/Project/master-graduation-project/master-graduation/data/oil/val/aligned_full_tif_20251231-2225_matchanything_affine_1440_crop",
        help="7 通道光谱 TIF 输出目录。",
    )
    parser.add_argument(
        "--rgb-ext",
        default="png",
        help="RGB 导出图像后缀（默认: png）。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="若输出文件已存在则覆盖。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rgb_written, tif_written = export_rgb_and_spectral_from_hdr(
        input_hdr_dir=args.input_hdr_dir,
        rgb_output_dir=args.output_rgb_dir,
        spectral_tif_output_dir=args.output_spectral_tif_dir,
        rgb_ext=args.rgb_ext.lstrip("."),
        overwrite=args.overwrite,
        spectral_channel_names=["450nm", "550nm", "650nm", "720nm", "750nm", "800nm", "850nm"],
    )
    print(f"完成: rgb={rgb_written}, tif={tif_written}")


if __name__ == "__main__":
    main()
