import argparse
import os
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import tifffile
from tqdm import tqdm
from shapely.geometry import Polygon, box


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


HDR_SPECTRAL7_CHANNEL_NAMES = [
    "450nm",
    "550nm",
    "650nm",
    "720nm",
    "750nm",
    "800nm",
    "850nm",
]


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


def _load_bgr_and_spectral_from_hdr(
    hdr_path: str,
    export_hdr_spectral: bool,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[list[str]]]:
    try:
        from utils.io.spectral_io import open_hdr_img
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "导入 HDR 读取器失败（utils.io.spectral_io.open_hdr_img）。"
        ) from exc

    cube = open_hdr_img(hdr_path)
    if cube is None:
        return None, None, None
    if cube.ndim != 3 or cube.shape[2] < 3:
        return None, None, None

    bgr_u8 = np.ascontiguousarray(to_uint8(cube[:, :, :3]))

    if not export_hdr_spectral:
        return bgr_u8, None, None

    spectral7: Optional[np.ndarray] = None
    channel_names: Optional[list[str]] = None
    if cube.shape[2] >= 10:
        spectral7 = cube[:, :, 3:10]
        channel_names = HDR_SPECTRAL7_CHANNEL_NAMES.copy()
    elif cube.shape[2] >= 7:
        spectral7 = cube[:, :, -7:]
        channel_names = HDR_SPECTRAL7_CHANNEL_NAMES.copy()

    if spectral7 is not None:
        spectral7 = np.ascontiguousarray(spectral7)
    return bgr_u8, spectral7, channel_names


def load_image_for_slicing(
    image_path: str,
    export_hdr_spectral: bool,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[list[str]]]:
    ext = os.path.splitext(image_path)[1].lower()
    if ext == ".hdr":
        return _load_bgr_and_spectral_from_hdr(
            hdr_path=image_path,
            export_hdr_spectral=export_hdr_spectral,
        )
    return cv2.imread(image_path), None, None


def _normalize_tif_axes(
    tif_array: np.ndarray,
    expected_height: int,
    expected_width: int,
) -> tuple[np.ndarray, bool]:
    """调整读取的 TIF 数据轴顺序，使其为 (H, W, C...) 形式。

    返回值：
        规范化后的数组，以及是否执行过轴重排的布尔标记。
    """
    if tif_array.ndim < 2:
        return tif_array, False

    # 若前两个轴已匹配目标尺寸，直接返回
    if (
        tif_array.shape[0] == expected_height
        and tif_array.ndim >= 2
        and tif_array.shape[1] == expected_width
    ):
        return tif_array, False

    axes = list(range(tif_array.ndim))
    height_axis = None
    width_axis = None

    for idx, size in enumerate(tif_array.shape):
        if height_axis is None and size == expected_height:
            height_axis = idx
            continue
        if width_axis is None and size == expected_width and idx != height_axis:
            width_axis = idx

    if height_axis is None or width_axis is None or height_axis == width_axis:
        return tif_array, False

    remaining_axes = [ax for ax in axes if ax not in (height_axis, width_axis)]
    target_order = [height_axis, width_axis, *remaining_axes]
    return np.transpose(tif_array, target_order), True


def _dedupe_polygon_points(points: list[list[float]]) -> list[list[float]]:
    """Remove consecutive duplicates and trailing closure point for LabelMe/X-AnyLabeling polygons."""
    if not points:
        return []

    cleaned: list[list[float]] = []
    for point in points:
        if not cleaned or point != cleaned[-1]:
            cleaned.append(point)

    while len(cleaned) >= 2 and cleaned[0] == cleaned[-1]:
        cleaned.pop()

    return cleaned


@dataclass
class ImageSliceStats:
    image_name: str
    source_shapes: int = 0
    tiles_total: int = 0
    tiles_saved: int = 0
    tiles_skipped_no_label: int = 0
    tiles_skipped_empty: int = 0
    shapes_saved: int = 0
    shapes_filtered_small_area: int = 0
    write_rgb_fail: int = 0
    write_tif_fail: int = 0
    write_json_fail: int = 0
    label_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class RunStats:
    images_total: int = 0
    images_processed: int = 0
    images_missing_json: int = 0
    images_failed: int = 0
    tiles_total: int = 0
    tiles_saved: int = 0
    tiles_skipped_no_label: int = 0
    tiles_skipped_empty: int = 0
    shapes_saved: int = 0
    shapes_filtered_small_area: int = 0
    write_rgb_fail: int = 0
    write_tif_fail: int = 0
    write_json_fail: int = 0
    label_counts: dict[str, int] = field(default_factory=dict)

    def add_image(self, stats: ImageSliceStats) -> None:
        self.images_processed += 1
        self.tiles_total += int(stats.tiles_total)
        self.tiles_saved += int(stats.tiles_saved)
        self.tiles_skipped_no_label += int(stats.tiles_skipped_no_label)
        self.tiles_skipped_empty += int(stats.tiles_skipped_empty)
        self.shapes_saved += int(stats.shapes_saved)
        self.shapes_filtered_small_area += int(stats.shapes_filtered_small_area)
        self.write_rgb_fail += int(stats.write_rgb_fail)
        self.write_tif_fail += int(stats.write_tif_fail)
        self.write_json_fail += int(stats.write_json_fail)
        for label, count in stats.label_counts.items():
            self.label_counts[label] = self.label_counts.get(label, 0) + int(count)


def slice_image_and_label(
    image_path,
    labelme_path,
    output_dir,
    tile_width,
    tile_height,
    step_x=None,
    step_y=None,
    save_only_labeled=False,
    tif_image_path=None,
    tif_output_dir=None,
    rgb_output_dir=None,
    label_output_dir=None,
    min_label_area=0.0,
    rgb_ext="jpg",
    export_hdr_spectral=True,
    show_tile_progress: bool = True,
):
    if step_x is None:
        step_x = tile_width
    if step_y is None:
        step_y = tile_height

    image, hdr_spectral, hdr_channel_names = load_image_for_slicing(
        image_path=image_path,
        export_hdr_spectral=export_hdr_spectral,
    )
    if image is None:
        print(f"无法读取图像: {image_path}")
        return
    img_height, img_width = image.shape[:2]

    tif_image = None
    tif_channel_names = None
    tif_output_dir = tif_output_dir or output_dir
    if tif_image_path:
        if not os.path.exists(tif_image_path):
            print(f"未找到对应的 TIF 文件: {tif_image_path}")
        else:
            try:
                with tifffile.TiffFile(tif_image_path) as tif:
                    tif_image = tif.asarray()
                    tif_metadata = {}
                    description = tif.pages[0].tags.get("ImageDescription")
                    if description is not None:
                        try:
                            tif_metadata = json.loads(description.value)
                        except Exception:
                            tif_metadata = {}
                    tif_channel_names = tif_metadata.get("ChannelNames")
            except Exception as exc:
                print(f"读取 TIF 文件失败 {tif_image_path}: {exc}")
                tif_image = None
            else:
                tif_image, _ = _normalize_tif_axes(tif_image, img_height, img_width)
                tif_height, tif_width = tif_image.shape[:2]
                if (tif_height, tif_width) != (img_height, img_width):
                    print(
                        f"TIF 图像尺寸与 RGB 不匹配: {tif_image_path} "
                        f"(TIF: {tif_width}x{tif_height}, RGB: {img_width}x{img_height})"
                    )
                    tif_image = None
                    tif_channel_names = None
    elif hdr_spectral is not None:
        tif_image = hdr_spectral
        tif_channel_names = hdr_channel_names

    base_output_dir = output_dir or os.path.dirname(image_path)
    rgb_dir = rgb_output_dir or base_output_dir
    label_dir = label_output_dir or base_output_dir
    spectral_dir = tif_output_dir if (tif_image is not None and tif_output_dir) else None
    if tif_image is not None and spectral_dir is None:
        spectral_dir = base_output_dir

    for path in {output_dir, rgb_dir, label_dir, spectral_dir}:
        if path and not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    if not os.path.exists(labelme_path):
        print(f"未找到对应的 JSON 文件: {labelme_path}")
        return

    with open(labelme_path, 'r', encoding='utf-8') as f:
        annotations = json.load(f)

    shapes = annotations.get("shapes", [])
    image_stats = ImageSliceStats(
        image_name=os.path.basename(image_path),
        source_shapes=len(shapes),
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    shape_polygons = []
    for shape in shapes:
        points = shape.get("points", [])
        polygon = Polygon(points)

        # 几何有效性处理[1](@ref)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)

        # 提取X-AnyLabeling特有字段[6,7](@ref)
        extra_fields = {
            "kie_linking": shape.get("kie_linking", []),
            "score": shape.get("score", None),
            "group_id": shape.get("group_id", None),
            "difficult": shape.get("difficult", False),
            "attributes": shape.get("attributes", {}),
            "description": shape.get("description", "")
        }

        shape_polygons.append({
            "label": shape.get("label", "unknown"),
            "polygon": polygon,
            "shape_type": shape.get("shape_type", "polygon"),
            "other_data": {k: v for k, v in shape.items() 
                          if k not in ["label", "points", "shape_type"]},
            **extra_fields  # 合并扩展字段
        })

    y_positions = list(range(0, img_height, step_y))
    x_positions = list(range(0, img_width, step_x))
    image_stats.tiles_total = len(y_positions) * len(x_positions)

    with tqdm(
        total=image_stats.tiles_total,
        desc=f"Slicing {Path(image_path).stem}",
        unit="tile",
        leave=False,
        disable=not show_tile_progress,
    ) as tile_pbar:
        tiles_seen = 0
        for row, top in enumerate(y_positions):
            for col, left in enumerate(x_positions):
                right = min(left + tile_width, img_width)
                bottom = min(top + tile_height, img_height)

                # 边缘补齐逻辑
                if right - left < tile_width:
                    left = img_width - tile_width
                if bottom - top < tile_height:
                    top = img_height - tile_height

                sub_image = image[top:bottom, left:right]
                if sub_image.size == 0:
                    image_stats.tiles_skipped_empty += 1
                    tiles_seen += 1
                    tile_pbar.update(1)
                    continue

                tile_box = box(left, top, right, bottom)

                sub_shapes = []
                for sp in shape_polygons:
                    poly_geometry = sp["polygon"]
                    if poly_geometry.is_empty:
                        continue

                    # 几何有效性二次验证[1](@ref)
                    if not poly_geometry.is_valid:
                        poly_geometry = poly_geometry.buffer(0)
                        if poly_geometry.is_empty or not poly_geometry.is_valid:
                            continue

                    if not poly_geometry.intersects(tile_box):
                        continue

                    try:
                        poly_intersection = poly_geometry.intersection(tile_box)
                    except Exception as e:
                        print(f"Intersection error: {e}")
                        continue

                    if poly_intersection.is_empty:
                        continue

                    # 处理多边形坐标转换[7](@ref)
                    if poly_intersection.geom_type == 'Polygon':
                        polygons_to_save = [poly_intersection]
                    elif poly_intersection.geom_type == 'MultiPolygon':
                        polygons_to_save = poly_intersection.geoms
                    else:
                        continue

                    for pg in polygons_to_save:
                        if pg.is_empty or not pg.is_valid:
                            continue
                        if min_label_area > 0:
                            poly_area = pg.area
                            if poly_area < min_label_area:
                                image_stats.shapes_filtered_small_area += 1
                                continue
                        exterior_coords = list(pg.exterior.coords)
                        # 保留4位小数精度[1](@ref)
                        sub_polygon_points = [
                            [round(x - left, 4), round(y - top, 4)]
                            for x, y in exterior_coords
                        ]
                        sub_polygon_points = _dedupe_polygon_points(sub_polygon_points)
                        if len(sub_polygon_points) < 3:
                            continue

                        # 构建X-AnyLabeling格式的shape[6,7](@ref)
                        sub_shape = {
                            "label": sp["label"],
                            "points": sub_polygon_points,
                            "shape_type": sp["shape_type"],
                            "kie_linking": sp["kie_linking"],
                            "score": sp["score"],
                            "group_id": sp["group_id"],
                            "difficult": sp["difficult"],
                            "attributes": sp["attributes"],
                            "description": sp["description"],
                            **sp["other_data"],
                        }

                        sub_shapes.append(sub_shape)

                if save_only_labeled and not sub_shapes:
                    image_stats.tiles_skipped_no_label += 1
                    tiles_seen += 1
                    tile_pbar.update(1)
                    continue

                image_stats.tiles_saved += 1
                image_stats.shapes_saved += len(sub_shapes)
                for shp in sub_shapes:
                    label = str(shp.get("label", "unknown"))
                    image_stats.label_counts[label] = image_stats.label_counts.get(label, 0) + 1

                # 保存切片图像
                tile_basename = f"{os.path.splitext(os.path.basename(image_path))[0]}_slice_{row}_{col}"
                tile_filename = f"{tile_basename}.{rgb_ext.lstrip('.')}"
                sub_image_path = os.path.join(rgb_dir, tile_filename)
                if not cv2.imwrite(sub_image_path, sub_image):
                    image_stats.write_rgb_fail += 1
                    print(f"写入 RGB 切片失败: {sub_image_path}")

                if tif_image is not None and spectral_dir:
                    sub_tif = tif_image[top:bottom, left:right, ...]
                    tif_filename = f"{tile_basename}.tif"
                    sub_tif_path = os.path.join(spectral_dir, tif_filename)
                    metadata = {}
                    tif_kwargs = {}

                    if sub_tif.ndim >= 3:
                        planar_sub_tif = np.ascontiguousarray(
                            np.transpose(sub_tif, (2, 0, 1))
                        )
                        tif_kwargs["planarconfig"] = "SEPARATE"
                        metadata["axes"] = "SYX"
                        if (
                            tif_channel_names
                            and len(tif_channel_names) == planar_sub_tif.shape[0]
                        ):
                            metadata["ChannelNames"] = tif_channel_names
                    else:
                        planar_sub_tif = np.ascontiguousarray(sub_tif)

                    try:
                        tifffile.imwrite(
                            sub_tif_path,
                            planar_sub_tif,
                            dtype=planar_sub_tif.dtype,
                            photometric="MINISBLACK",
                            metadata=metadata or None,
                            **tif_kwargs,
                        )
                    except Exception as exc:
                        image_stats.write_tif_fail += 1
                        print(f"写入 TIF 切片失败 {sub_tif_path}: {exc}")

                # 构建X-AnyLabeling格式的标注文件[7](@ref)
                sub_annotation = {
                    # "version": "2.5.4",  # 匹配X-AnyLabeling版本
                    "version": "3.3.0-beta.2",
                    "flags": annotations.get("flags", {}),
                    "shapes": sub_shapes,
                    "imagePath": tile_filename,
                    "imageData": None,
                    "imageHeight": sub_image.shape[0],
                    "imageWidth": sub_image.shape[1],
                    "description": annotations.get("description", ""),  # 全局描述
                }

                json_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_slice_{row}_{col}.json"
                sub_json_path = os.path.join(label_dir, json_filename)
                try:
                    with open(sub_json_path, 'w', encoding='utf-8') as jf:
                        json.dump(sub_annotation, jf, ensure_ascii=False, indent=2)
                except Exception as exc:
                    image_stats.write_json_fail += 1
                    print(f"写入 JSON 切片失败 {sub_json_path}: {exc}")

                tiles_seen += 1
                if show_tile_progress and (tiles_seen % 50 == 0 or tiles_seen == image_stats.tiles_total):
                    tile_pbar.set_postfix(
                        saved=image_stats.tiles_saved,
                        shapes=image_stats.shapes_saved,
                        skipped=image_stats.tiles_skipped_no_label,
                    )
                tile_pbar.update(1)

    return image_stats

def process_folder(
    image_folder_path,
    label_folder_path,
    output_dir,
    tile_width,
    tile_height,
    step_x=None,
    step_y=None,
    image_exts=None,
    tif_folder_path=None,
    tif_exts=None,
    save_only_labeled=False,
    min_label_area=0.0,
    rgb_ext="jpg",
    export_hdr_spectral=True,
    show_tile_progress: bool = True,
    print_stats: bool = True,
):
    """
    批量处理文件夹下的图像与对应的 LabelMe JSON 文件。
    参数：
        image_folder_path : 包含图像文件的文件夹路径
        label_folder_path : 包含 JSON 标签文件的文件夹路径
        output_dir        : 输出根目录（自动创建 rgb/labels_xanylabeling/spectral 子目录）
        tile_width        : 子图块的宽度
        tile_height       : 子图块的高度
        step_x            : 水平滑动步长（可选）
        step_y            : 垂直滑动步长（可选）
        image_exts        : 可识别的图像扩展名列表（如 [".jpg", ".png"]，可选）
        tif_folder_path   : 包含与 RGB 尺寸一致的 TIF 文件夹路径（可选）
        tif_exts          : 可识别的 TIF 扩展名列表（如 [".tif", ".tiff"]，可选）
        save_only_labeled : 仅保存有标签的图像（默认 False）
        min_label_area    : 切片内保留标签的最小面积阈值（像素单位，默认 0 为不过滤）
    """
    if image_exts is None:
        image_exts = [".jpg", ".png", ".jpeg", ".hdr"]
    if tif_exts is None:
        tif_exts = [".tif", ".tiff"]

    if not os.path.exists(image_folder_path):
        raise ValueError(f"图像文件夹不存在: {image_folder_path}")

    if tif_folder_path and not os.path.exists(tif_folder_path):
        raise ValueError(f"TIF 文件夹不存在: {tif_folder_path}")

    if not os.path.exists(label_folder_path):
        raise ValueError(f"标签文件夹不存在: {label_folder_path}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    rgb_output_dir = os.path.join(output_dir, "rgb")
    label_output_dir = os.path.join(output_dir, "xanylabeling")
    spectral_output_dir = os.path.join(output_dir, "spectral") if (tif_folder_path or export_hdr_spectral) else None
    os.makedirs(rgb_output_dir, exist_ok=True)
    os.makedirs(label_output_dir, exist_ok=True)
    if spectral_output_dir:
        os.makedirs(spectral_output_dir, exist_ok=True)

    # 遍历文件夹中的图像文件，按图像与其对应 JSON 进行处理
    run_stats = RunStats()

    image_files: list[str] = []
    for file_name in sorted(os.listdir(image_folder_path)):
        file_path = os.path.join(image_folder_path, file_name)
        if not os.path.isfile(file_path):
            continue
        _, ext = os.path.splitext(file_name.lower())
        if ext in image_exts:
            image_files.append(file_name)

    run_stats.images_total = len(image_files)

    with tqdm(image_files, desc="Processing images", unit="img") as pbar:
        for file_name in pbar:
            file_path = os.path.join(image_folder_path, file_name)

            json_path = os.path.join(
                label_folder_path, f"{os.path.splitext(file_name)[0]}.json"
            )
            if not os.path.exists(json_path):
                run_stats.images_missing_json += 1
                continue

            tif_path = None
            if tif_folder_path:
                base_name = os.path.splitext(file_name)[0]
                for tif_ext in tif_exts:
                    candidate = os.path.join(tif_folder_path, f"{base_name}{tif_ext}")
                    if os.path.exists(candidate):
                        tif_path = candidate
                        break
                if tif_path is None:
                    print(f"未找到匹配的 TIF 文件: {base_name}")

            try:
                image_stats = slice_image_and_label(
                    image_path=file_path,
                    labelme_path=json_path,
                    output_dir=output_dir,
                    tile_width=tile_width,
                    tile_height=tile_height,
                    step_x=step_x,
                    step_y=step_y,
                    save_only_labeled=save_only_labeled,
                    tif_image_path=tif_path,
                    min_label_area=min_label_area,
                    rgb_output_dir=rgb_output_dir,
                    label_output_dir=label_output_dir,
                    tif_output_dir=spectral_output_dir,
                    rgb_ext=rgb_ext,
                    export_hdr_spectral=export_hdr_spectral,
                    show_tile_progress=show_tile_progress,
                )
            except Exception as exc:
                run_stats.images_failed += 1
                print(f"处理失败: {file_name}: {exc}")
                continue

            if image_stats is None:
                run_stats.images_failed += 1
                continue

            run_stats.add_image(image_stats)
            pbar.set_postfix(
                saved_tiles=run_stats.tiles_saved,
                shapes=run_stats.shapes_saved,
                missing_json=run_stats.images_missing_json,
                failed=run_stats.images_failed,
            )

    if print_stats:
        print("\n=== Split Summary ===")
        print(
            "images: total={total}, processed={processed}, missing_json={missing}, failed={failed}".format(
                total=run_stats.images_total,
                processed=run_stats.images_processed,
                missing=run_stats.images_missing_json,
                failed=run_stats.images_failed,
            )
        )
        print(
            "tiles: total={total}, saved={saved}, skipped_no_label={skipped}, skipped_empty={empty}".format(
                total=run_stats.tiles_total,
                saved=run_stats.tiles_saved,
                skipped=run_stats.tiles_skipped_no_label,
                empty=run_stats.tiles_skipped_empty,
            )
        )
        print(
            "shapes: saved={saved}, filtered_small_area={filtered}".format(
                saved=run_stats.shapes_saved,
                filtered=run_stats.shapes_filtered_small_area,
            )
        )
        if run_stats.write_rgb_fail or run_stats.write_tif_fail or run_stats.write_json_fail:
            print(
                "write_fail: rgb={rgb}, tif={tif}, json={json}".format(
                    rgb=run_stats.write_rgb_fail,
                    tif=run_stats.write_tif_fail,
                    json=run_stats.write_json_fail,
                )
            )
        if run_stats.label_counts:
            print("label_counts:")
            for label, count in sorted(
                run_stats.label_counts.items(), key=lambda kv: (-kv[1], kv[0])
            ):
                print(f"  {label}: {count}")

    return run_stats

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="将 RGB/HDR 图像及 X-AnyLabeling/LabelMe 标注切分为固定大小瓦片，并同步裁剪对应的 TIF 栈或 HDR 内的光谱通道。"
    )
    parser.add_argument(
        "--image-folder",
        type=str,
        default=r"/mnt/d/Project/master-graduation-project/master-graduation/data/oil/train/feedback/aligned/aligned_rgb",
        help="RGB 图像目录路径。",
    )
    parser.add_argument(
        "--tif-folder",
        type=str,
        default="",
        help="对应的多通道 TIF 栈目录（可选）；为空则跳过 TIF 裁剪。",
    )
    parser.add_argument(
        "--label-folder",
        type=str,
        default=r"/mnt/d/Project/master-graduation-project/master-graduation/data/oil/train/feedback/aligned/labels_seg_5_XAnyLabeling",
        help="LabelMe/X-AnyLabeling JSON 目录路径。",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=r"/mnt/d/Project/master-graduation-project/master-graduation/data/oil/train/feedback/aligned/split",
        help="输出根目录（将自动创建 rgb/xanylabeling/spectral 子目录）。",
    )
    parser.add_argument(
        "--tile-width",
        type=int,
        default=640,
        help="瓦片宽度（像素）。",
    )
    parser.add_argument(
        "--tile-height",
        type=int,
        default=640,
        help="瓦片高度（像素）。",
    )
    parser.add_argument(
        "--step-x",
        type=int,
        default=None,
        help="水平步进（像素），默认等于瓦片宽度。",
    )
    parser.add_argument(
        "--step-y",
        type=int,
        default=None,
        help="垂直步进（像素），默认等于瓦片高度。",
    )
    parser.add_argument(
        "--image-exts",
        nargs="*",
        default=[".jpg", ".png", ".jpeg", ".hdr"],
        help="识别为图像的扩展名列表。",
    )
    parser.add_argument(
        "--tif-exts",
        nargs="*",
        default=[".tif", ".tiff"],
        help="TIF 文件扩展名列表。",
    )
    parser.add_argument(
        "--save-only-labeled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="仅输出包含标签的瓦片，默认开启，可通过 --no-save-only-labeled 关闭。",
    )
    parser.add_argument(
        "--rgb-ext",
        default="jpg",
        help="输出 RGB 切片图像扩展名（默认: jpg）。",
    )
    parser.add_argument(
        "--export-hdr-spectral",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="当输入为 HDR 且未提供 --tif-folder 时，是否从 HDR 额外导出 7 通道光谱 TIF 切片。",
    )
    parser.add_argument(
        "--show-tile-progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否显示单张图片的瓦片切分进度条（默认开启，可通过 --no-show-tile-progress 关闭）。",
    )
    parser.add_argument(
        "--print-stats",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否在结束后打印统计信息（默认开启，可通过 --no-print-stats 关闭）。",
    )
    parser.add_argument(
        "--min-label-area",
        type=float,
        default=9,
        help="保留标注的最小面积阈值（像素），默认为 0 表示不过滤。",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    process_folder(
        image_folder_path=args.image_folder,
        label_folder_path=args.label_folder,
        output_dir=args.output_dir,
        tile_width=args.tile_width,
        tile_height=args.tile_height,
        step_x=args.step_x,
        step_y=args.step_y,
        image_exts=args.image_exts,
        tif_folder_path=args.tif_folder or None,
        tif_exts=args.tif_exts,
        save_only_labeled=args.save_only_labeled,
        min_label_area=args.min_label_area,
        rgb_ext=args.rgb_ext,
        export_hdr_spectral=args.export_hdr_spectral,
        show_tile_progress=args.show_tile_progress,
        print_stats=args.print_stats,
    )


if __name__ == "__main__":
    main()
"""
python utils/data_process/split_image_and_anylabeling_labels.py\
 --image-folder /mnt/d/Project/master-graduation-project/data/oil/oil_dataset_20260109_简单目标/aligned_full_tif_20260110-1126_s1-matchanything_s2-none_cropped \
--label-folder /mnt/d/Project/master-graduation-project/data/oil/oil_dataset_20260109_简单目标/aligned_full_tif_20260110-1126_s1-matchanything_s2-none_cropped \
--output-dir /mnt/d/Project/master-graduation-project/data/oil/oil_dataset_20260109_简单目标/clip_test \
--tile-width 640 \
--tile-height 640 \
--save-only-labeled


python utils/data_process/split_image_and_anylabeling_labels.py\
 --image-folder /mnt/d/Project/master-graduation-project/data/oil/val/aligned_full_tif_20251231-2225_matchanything_affine_1440_cropped \
--label-folder /mnt/d/Project/master-graduation-project/data/oil/val/aligned_full_tif_20251231-2225_matchanything_affine_1440_cropped \
--output-dir /mnt/d/Project/master-graduation-project/data/oil/val/clip \
--tile-width 640 \
--tile-height 640 \
--save-only-labeled
"""
