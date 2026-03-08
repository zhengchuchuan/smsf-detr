#!/usr/bin/env python3
"""
批量将按 CHW（planar）存储的 TIF 图像转换为 HWC（contiguous）布局。

示例：
    python convert_chw_tif_to_hwc.py \
        --input /path/to/chw_tifs \
        --output /path/to/hwc_tifs
"""

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import tifffile
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert CHW-formatted TIF stacks into HWC layout."
    )
    parser.add_argument(
        "--input",
        # required=True,
        type=Path,
        default="/mnt/d/Project/master-graduation-project/master-graduation/data/oil/val/aligned/aligned_nir_tif_20251104-1707",
        help="包含 CHW TIF 文件的输入目录。",
    )
    parser.add_argument(
        "--output",
        # required=True,
        type=Path,
        default="/mnt/d/Project/master-graduation-project/master-graduation/data/oil/val/aligned/aligned_nir_tif_20251104-1707-tmp",
        help="转换后 HWC TIF 的输出目录。",
    )
    parser.add_argument(
        "--exts",
        nargs="*",
        default=[".tif", ".tiff"],
        help="需要处理的文件扩展名（默认: .tif .tiff）。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="如输出文件存在则覆盖。",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="始终执行 CHW->HWC 转置，不做形状检查。",
    )
    return parser.parse_args()


def detect_layout(array: np.ndarray) -> str:
    """基于尺寸粗略判断是 CHW 还是 HWC。"""
    if array.ndim < 3:
        raise ValueError(f"期望至少 3 维数据，得到 shape={array.shape}")

    first, second, third = array.shape[:3]
    if first <= 32 and second > 32 and third > 32:
        return "CHW"
    if third <= 32 and first > 32 and second > 32:
        return "HWC"
    return "UNKNOWN"


def transpose_to_hwc(array: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.transpose(array, (1, 2, 0)))


def load_axes_metadata(tif_path: Path) -> Tuple[str, dict]:
    axes = None
    metadata_dict = {}
    with tifffile.TiffFile(tif_path) as tif:
        description = tif.pages[0].tags.get("ImageDescription")
        if description is not None:
            try:
                metadata_dict = json.loads(description.value)
                axes = metadata_dict.get("axes")
            except Exception:
                metadata_dict = {}
    return axes, metadata_dict


def convert_single_file(
    src_path: Path,
    dst_path: Path,
    overwrite: bool,
    force: bool,
) -> None:
    if dst_path.exists() and not overwrite:
        return

    array = tifffile.imread(src_path)
    axes, src_meta = load_axes_metadata(src_path)

    need_transpose = force
    if not force:
        if axes == "SYX":
            need_transpose = True
        elif axes == "YXS":
            need_transpose = False
        else:
            layout = detect_layout(array)
            if layout == "CHW":
                need_transpose = True
            elif layout == "HWC":
                need_transpose = False
            else:
                raise ValueError(
                    f"无法根据形状判定布局: {src_path.name}, shape={array.shape}"
                )

    if need_transpose:
        array = transpose_to_hwc(array)

    metadata = {
        "axes": "YXS",
    }
    if "ChannelNames" in src_meta:
        metadata["ChannelNames"] = src_meta["ChannelNames"]

    tifffile.imwrite(
        str(dst_path),
        array,
        dtype=array.dtype,
        photometric="MINISBLACK",
        metadata=metadata,
    )


def main() -> None:
    args = parse_args()
    input_dir = args.input.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = [
        p
        for p in sorted(input_dir.iterdir())
        if p.suffix.lower() in {ext.lower() for ext in args.exts}
    ]

    if not candidates:
        print(f"在 {input_dir} 未找到指定扩展的 TIF 文件。")
        return

    for src in tqdm(candidates, desc="Converting TIF"):
        dst = output_dir / src.name
        try:
            convert_single_file(src, dst, args.overwrite, args.force)
        except Exception as exc:
            print(f"转换失败 {src.name}: {exc}")


if __name__ == "__main__":
    main()
