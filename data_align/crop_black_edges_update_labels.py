import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import spectral as spy  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Missing dependency 'spectral'. Please install it to read/write ENVI HDR."
    ) from exc

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.io.spectral_io import save_hdr_img, channel as SPECTRAL_CHANNELS

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CropRect:
    top: int
    bottom: int
    left: int
    right: int

    @property
    def height(self) -> int:
        return int(self.bottom - self.top)

    @property
    def width(self) -> int:
        return int(self.right - self.left)


def ensure_byte_order_header(hdr_path: Path) -> None:
    try:
        text = hdr_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return
    if "byte order" in text:
        return
    hdr_path.write_text(text.rstrip() + "\nbyte order = 0\n", encoding="utf-8")


def load_envi_cube(hdr_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    if not hdr_path.exists():
        raise FileNotFoundError(hdr_path)

    try:
        img = spy.open_image(str(hdr_path))
    except Exception:
        ensure_byte_order_header(hdr_path)
        img = spy.open_image(str(hdr_path))

    arr = np.asarray(img[:, :, :])
    meta: Dict[str, Any] = dict(img.metadata) if getattr(img, "metadata", None) else {}
    return arr, meta


def parse_wavelengths(meta: Dict[str, Any], channels: int) -> List[float]:
    wl = meta.get("wavelength")
    if isinstance(wl, (list, tuple)) and len(wl) == channels:
        out: List[float] = []
        for v in wl:
            try:
                out.append(float(v))
            except Exception:
                out.append(float(len(out) + 1))
        return out
    return [float(i + 1) for i in range(channels)]


def parse_band_names(meta: Dict[str, Any], wavelengths: List[float]) -> List[str]:
    bn = meta.get("band names")
    if isinstance(bn, (list, tuple)) and len(bn) == len(wavelengths):
        return [str(x) for x in bn]
    return [f"{w:g}nm" for w in wavelengths]


def compute_black_mask(cube: np.ndarray, black_thr: float) -> np.ndarray:
    if cube.ndim != 3:
        raise ValueError(f"Expected HxWxC cube, got {cube.shape}")
    raise RuntimeError(
        "compute_black_mask(cube, ...) is deprecated. Use compute_band_black_mask(band, ...) instead."
    )


def compute_band_black_mask(band: np.ndarray, black_thr: float) -> np.ndarray:
    if band.ndim != 2:
        raise ValueError(f"Expected HxW band, got {band.shape}")
    if band.dtype.kind in {"u", "i"}:
        thr = int(black_thr)
        return band <= thr
    safe = np.nan_to_num(band, nan=0.0, posinf=0.0, neginf=0.0)
    return np.abs(safe) <= float(black_thr)


def border_connected_black(black_mask: np.ndarray) -> np.ndarray:
    if black_mask.ndim != 2:
        raise ValueError("black_mask must be 2D")
    h, w = black_mask.shape
    img = black_mask.astype(np.uint8, copy=True)
    filled_val = 2
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

    def seed_points() -> Iterable[Tuple[int, int]]:
        for x in range(w):
            yield (x, 0)
            if h > 1:
                yield (x, h - 1)
        for y in range(1, h - 1):
            yield (0, y)
            if w > 1:
                yield (w - 1, y)

    import cv2  # local import to keep module light if unused

    for x, y in seed_points():
        if img[y, x] != 1:
            continue
        cv2.floodFill(img, flood_mask, seedPoint=(int(x), int(y)), newVal=filled_val)

    return img == filled_val


def largest_true_rectangle(mask: np.ndarray) -> Optional[CropRect]:
    mask_int = mask.astype(np.uint8)
    height, width = mask_int.shape
    heights = np.zeros(width, dtype=np.int32)
    best_area = 0
    best_rect: Optional[Tuple[int, int, int, int]] = None

    for y in range(height):
        heights = (heights + 1) * mask_int[y]

        stack: list[Tuple[int, int]] = []
        for x in range(width + 1):
            current_height = heights[x] if x < width else 0
            start = x

            while stack and current_height < stack[-1][1]:
                idx, h_prev = stack.pop()
                width_rect = x - idx
                area = h_prev * width_rect
                if area > best_area and h_prev > 0 and width_rect > 0:
                    left = idx
                    right = x
                    top = y - h_prev + 1
                    bottom = y + 1
                    best_area = area
                    best_rect = (top, bottom, left, right)
                start = idx

            stack.append((start, current_height))

    if best_rect is None:
        return None
    top, bottom, left, right = best_rect
    return CropRect(top=int(top), bottom=int(bottom), left=int(left), right=int(right))


def compute_border_black_union(cube: np.ndarray, black_thr: float) -> np.ndarray:
    """Union of per-band border-connected 'black/fill' pixels.

    This avoids treating legitimate interior zeros as invalid, while ensuring that if any band
    has warp-induced black borders, those pixels are considered invalid for the final crop.
    """
    if cube.ndim != 3:
        raise ValueError(f"Expected HxWxC cube, got {cube.shape}")
    h, w, c = cube.shape
    union = np.zeros((h, w), dtype=bool)
    for ch in range(c):
        band = cube[:, :, ch]
        black = compute_band_black_mask(band, black_thr=black_thr)
        union |= border_connected_black(black)
    return union


def erode_mask(valid_mask: np.ndarray, erode_px: int) -> np.ndarray:
    if erode_px <= 0:
        return valid_mask
    import cv2  # local import

    k = int(erode_px) * 2 + 1
    kernel = np.ones((k, k), dtype=np.uint8)
    src = valid_mask.astype(np.uint8) * 255
    eroded = cv2.erode(src, kernel, iterations=1)
    return eroded > 0


def crop_cube(cube: np.ndarray, rect: CropRect) -> np.ndarray:
    return cube[rect.top : rect.bottom, rect.left : rect.right, :]


def compute_allzero_border_crop_rect(cube: np.ndarray) -> CropRect:
    """Crop only border rows/cols that are fully zero in *any* channel.

    A border row is removable if there exists a channel where that whole row is 0.
    A border column is removable if there exists a channel where that whole column is 0.
    This intentionally avoids threshold-based cropping so that dim borders are kept.
    """
    if cube.ndim != 3:
        raise ValueError(f"Expected HxWxC cube, got {cube.shape}")

    h, w, _ = cube.shape
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid cube shape: {cube.shape}")

    def count_leading_true(mask_1d: np.ndarray) -> int:
        if mask_1d.size == 0:
            return 0
        if bool(np.all(mask_1d)):
            return int(mask_1d.size)
        return int(np.flatnonzero(~mask_1d)[0])

    def count_trailing_true(mask_1d: np.ndarray) -> int:
        if mask_1d.size == 0:
            return 0
        if bool(np.all(mask_1d)):
            return int(mask_1d.size)
        return int(np.flatnonzero(~mask_1d[::-1])[0])

    top, bottom, left, right = 0, h, 0, w
    while True:
        prev = (top, bottom, left, right)

        view = cube[top:bottom, left:right, :]
        if view.size == 0:
            return CropRect(top=0, bottom=h, left=0, right=w)

        row_all_zero_per_channel = np.all(view == 0, axis=1)  # (H, C)
        row_removable = np.any(row_all_zero_per_channel, axis=1)  # (H,)
        top += count_leading_true(row_removable)
        bottom -= count_trailing_true(row_removable)
        if bottom <= top:
            return CropRect(top=0, bottom=h, left=0, right=w)

        view = cube[top:bottom, left:right, :]
        col_all_zero_per_channel = np.all(view == 0, axis=0)  # (W, C)
        col_removable = np.any(col_all_zero_per_channel, axis=1)  # (W,)
        left += count_leading_true(col_removable)
        right -= count_trailing_true(col_removable)
        if right <= left:
            return CropRect(top=0, bottom=h, left=0, right=w)

        if (top, bottom, left, right) == prev:
            break

    return CropRect(top=int(top), bottom=int(bottom), left=int(left), right=int(right))


def to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    if np.issubdtype(image.dtype, np.integer):
        # For uint16 RGB that was previously expanded from uint8 via bit replication
        # (v16 = (v8<<8)|v8), invert it with >>8 to avoid near-black rendering or saturation.
        if image.dtype == np.uint16:
            flat = image.reshape(-1)
            if flat.size > 1_000_000:
                flat = flat[:: max(flat.size // 1_000_000, 1)]
            if np.all((flat & 0x00FF) == (flat >> 8)):
                return (image >> 8).astype(np.uint8, copy=False)

        finite = np.nan_to_num(image, nan=0, copy=False)
        max_val = float(np.max(finite))
        min_val = float(np.min(finite))
        if max_val <= min_val:
            return np.zeros_like(finite, dtype=np.uint8)
        # Use percentiles to avoid outliers dominating the visualization.
        lo = float(np.percentile(finite, 1))
        hi = float(np.percentile(finite, 99))
        if hi <= lo:
            lo, hi = min_val, max_val
            if hi <= lo:
                return np.zeros_like(finite, dtype=np.uint8)
        scaled = (finite - lo) / (hi - lo)
        return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)
    finite = np.nan_to_num(image, nan=0.0, copy=False)
    max_val = float(np.max(finite))
    min_val = float(np.min(finite))
    if max_val <= min_val:
        return np.zeros_like(finite, dtype=np.uint8)
    scaled = (finite - min_val) / (max_val - min_val)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def save_bgr_png(bgr: np.ndarray, output_path: Path, *, overwrite: bool) -> bool:
    if output_path.exists() and not overwrite:
        return False
    import cv2  # local import

    bgr_u8 = np.ascontiguousarray(to_uint8(bgr))
    ok = cv2.imwrite(str(output_path), bgr_u8)
    if not ok:
        raise RuntimeError(f"Failed to write image: {output_path}")
    return True


def save_spectral7_tif(
    cube: np.ndarray,
    output_path: Path,
    *,
    overwrite: bool,
    channel_names: Optional[list[str]] = None,
) -> bool:
    if output_path.exists() and not overwrite:
        return False

    if cube.ndim != 3:
        raise ValueError(f"Expected HxWxC cube, got {cube.shape}")
    _, _, c = cube.shape
    if c >= 10:
        spectral7 = cube[:, :, 3:10]
    elif c >= 7:
        spectral7 = cube[:, :, -7:]
    else:
        return False

    import tifffile  # local import

    spectral7 = np.ascontiguousarray(spectral7)
    planar = np.ascontiguousarray(spectral7.transpose(2, 0, 1))
    metadata: Dict[str, Any] = {"axes": "SYX"}
    if channel_names is not None and len(channel_names) == planar.shape[0]:
        metadata["ChannelNames"] = channel_names
    tifffile.imwrite(
        str(output_path),
        planar,
        dtype=planar.dtype,
        photometric="MINISBLACK",
        planarconfig="SEPARATE",
        metadata=metadata,
    )
    return True


def polygon_area(points: List[Tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    x = np.asarray([p[0] for p in points], dtype=np.float64)
    y = np.asarray([p[1] for p in points], dtype=np.float64)
    return float(0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def clip_polygon_to_rect(
    points: List[Tuple[float, float]],
    *,
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
) -> List[Tuple[float, float]]:
    def clip_edge(
        poly: List[Tuple[float, float]],
        inside_fn,
        intersect_fn,
    ) -> List[Tuple[float, float]]:
        if not poly:
            return []
        out: List[Tuple[float, float]] = []
        prev = poly[-1]
        prev_inside = inside_fn(prev)
        for cur in poly:
            cur_inside = inside_fn(cur)
            if cur_inside:
                if not prev_inside:
                    out.append(intersect_fn(prev, cur))
                out.append(cur)
            elif prev_inside:
                out.append(intersect_fn(prev, cur))
            prev = cur
            prev_inside = cur_inside
        return out

    def intersect(p1: Tuple[float, float], p2: Tuple[float, float], x=None, y=None):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        if x is not None:
            if dx == 0:
                return (x, y1)
            t = (x - x1) / dx
            return (x, y1 + t * dy)
        if y is not None:
            if dy == 0:
                return (x1, y)
            t = (y - y1) / dy
            return (x1 + t * dx, y)
        raise ValueError("intersect requires x or y")

    poly = points
    poly = clip_edge(
        poly,
        inside_fn=lambda p: p[0] >= x_min,
        intersect_fn=lambda p1, p2: intersect(p1, p2, x=x_min),
    )
    poly = clip_edge(
        poly,
        inside_fn=lambda p: p[0] <= x_max,
        intersect_fn=lambda p1, p2: intersect(p1, p2, x=x_max),
    )
    poly = clip_edge(
        poly,
        inside_fn=lambda p: p[1] >= y_min,
        intersect_fn=lambda p1, p2: intersect(p1, p2, y=y_min),
    )
    poly = clip_edge(
        poly,
        inside_fn=lambda p: p[1] <= y_max,
        intersect_fn=lambda p1, p2: intersect(p1, p2, y=y_max),
    )
    return poly


def clamp(v: float, lo: float, hi: float) -> float:
    return float(min(max(v, lo), hi))


def update_labelme_json_for_crop(
    label_obj: Dict[str, Any],
    *,
    rect: CropRect,
    new_width: int,
    new_height: int,
    output_image_basename: str,
    set_image_path: str,
    drop_image_data: bool,
    min_polygon_area: float,
) -> Tuple[Dict[str, Any], int]:
    removed = 0
    label_obj = dict(label_obj)

    shapes = label_obj.get("shapes", [])
    if not isinstance(shapes, list):
        shapes = []

    updated_shapes: List[Dict[str, Any]] = []
    for shape in shapes:
        if not isinstance(shape, dict):
            continue
        points = shape.get("points")
        if not isinstance(points, list) or not points:
            removed += 1
            continue

        shifted: List[Tuple[float, float]] = []
        for pt in points:
            if (
                not isinstance(pt, (list, tuple))
                or len(pt) != 2
                or pt[0] is None
                or pt[1] is None
            ):
                shifted = []
                break
            shifted.append((float(pt[0]) - rect.left, float(pt[1]) - rect.top))

        if not shifted:
            removed += 1
            continue

        shape_type = str(shape.get("shape_type") or "polygon").lower()
        if shape_type == "rectangle" and len(shifted) >= 2:
            xs = [p[0] for p in shifted]
            ys = [p[1] for p in shifted]
            x0 = clamp(min(xs), 0.0, float(new_width))
            y0 = clamp(min(ys), 0.0, float(new_height))
            x1 = clamp(max(xs), 0.0, float(new_width))
            y1 = clamp(max(ys), 0.0, float(new_height))
            if x1 <= x0 or y1 <= y0:
                removed += 1
                continue
            shape = dict(shape)
            if len(points) == 2:
                shape["points"] = [[x0, y0], [x1, y1]]
            else:
                shape["points"] = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
            updated_shapes.append(shape)
            continue

        if shape_type == "polygon" and len(shifted) >= 3:
            clipped = clip_polygon_to_rect(
                shifted,
                x_min=0.0,
                y_min=0.0,
                x_max=float(new_width),
                y_max=float(new_height),
            )
            if len(clipped) < 3 or polygon_area(clipped) < float(min_polygon_area):
                removed += 1
                continue
            shape = dict(shape)
            shape["points"] = [[x, y] for x, y in clipped]
            updated_shapes.append(shape)
            continue

        xs = [p[0] for p in shifted]
        ys = [p[1] for p in shifted]
        if max(xs) < 0 or max(ys) < 0 or min(xs) > new_width or min(ys) > new_height:
            removed += 1
            continue
        clamped_pts = [
            [clamp(x, 0.0, float(new_width)), clamp(y, 0.0, float(new_height))]
            for x, y in shifted
        ]
        shape = dict(shape)
        shape["points"] = clamped_pts
        updated_shapes.append(shape)

    label_obj["shapes"] = updated_shapes
    label_obj["imageWidth"] = int(new_width)
    label_obj["imageHeight"] = int(new_height)
    if drop_image_data:
        # X-AnyLabeling/Labelme expects JSON null when image data is not embedded.
        label_obj["imageData"] = None

    if set_image_path == "keep":
        pass
    elif set_image_path == "basename":
        label_obj["imagePath"] = output_image_basename
    else:
        raise ValueError(f"Unknown set_image_path: {set_image_path}")

    return label_obj, removed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch crop border rows/cols that are fully zero in ENVI .hdr cubes and update X-AnyLabeling/Labelme JSON.\n"
            "Outputs are cropped .hdr (+.img) and updated label JSONs.\n"
            "Also exports cropped RGB images and 7-ch spectral TIF (defaults can be overridden)."
        )
    )
    parser.add_argument("--input-hdr-dir", type=Path, required=True, help="Input aligned .hdr directory")
    parser.add_argument("--input-label-dir", type=Path, required=True, help="Input label JSON directory")
    parser.add_argument(
        "--output-hdr-dir",
        "--output_hdr_dir",
        dest="output_hdr_dir",
        type=Path,
        default=None,
        help="Output directory for cropped .hdr/.img (default: sibling of input-hdr-dir).",
    )
    parser.add_argument(
        "--output-label-dir",
        "--output_label_dir",
        dest="output_label_dir",
        type=Path,
        default=None,
        help="Output updated label JSON directory (default: same as --output-hdr-dir).",
    )
    parser.add_argument(
        "--output-rgb-dir",
        "--output_rgb_dir",
        dest="output_rgb_dir",
        type=Path,
        default=None,
        help="Output cropped RGB images directory (default: same as --output-hdr-dir).",
    )
    parser.add_argument(
        "--output-spectral-tif-dir",
        "--output_spectral_tif_dir",
        dest="output_spectral_tif_dir",
        type=Path,
        default=None,
        help="If set, export cropped 7-channel spectral TIFs to this directory (default: disabled).",
    )
    parser.add_argument(
        "--export-spectral-tif",
        action="store_true",
        help="Export 7-channel spectral TIFs into --output-hdr-dir (default: disabled).",
    )
    parser.add_argument(
        "--rgb-ext",
        default="png",
        help="RGB export extension (default: png).",
    )
    parser.add_argument(
        "--black-thr",
        type=float,
        default=0.0,
        help="Deprecated/ignored: previously used for threshold-based crop (kept for compatibility).",
    )
    parser.add_argument(
        "--erode-valid-px",
        type=int,
        default=0,
        help="Deprecated/ignored: previously used for rotation/edge artifact removal (kept for compatibility).",
    )
    parser.add_argument("--min-polygon-area", type=float, default=1.0, help="Remove polygons with area < this value (default: 1.0)")
    parser.add_argument(
        "--set-image-path",
        choices=["keep", "basename"],
        default="basename",
        help="How to set JSON imagePath (default: basename -> <stem>.<rgb_ext> when exporting RGB, else <stem>.hdr).",
    )
    parser.add_argument(
        "--keep-missing-label",
        action="store_true",
        help="If a .hdr has no matching label JSON, still crop and save the cube.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bar")
    parser.add_argument(
        "--keep-image-data",
        action="store_true",
        help="Keep JSON imageData as-is (default: drop it because image size changes)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    input_hdr_dir: Path = args.input_hdr_dir
    input_label_dir: Path = args.input_label_dir

    if not input_hdr_dir.is_dir():
        raise FileNotFoundError(f"Missing input hdr dir: {input_hdr_dir}")
    if not input_label_dir.is_dir():
        raise FileNotFoundError(f"Missing input label dir: {input_label_dir}")

    sibling_root = input_hdr_dir.parent

    output_hdr_dir: Path = (
        args.output_hdr_dir
        if args.output_hdr_dir is not None
        else sibling_root / f"{input_hdr_dir.name}_cropped"
    )
    output_label_dir: Path = args.output_label_dir if args.output_label_dir is not None else output_hdr_dir
    output_rgb_dir: Path = args.output_rgb_dir if args.output_rgb_dir is not None else output_hdr_dir
    output_spectral_tif_dir: Optional[Path]
    if args.output_spectral_tif_dir is not None:
        output_spectral_tif_dir = args.output_spectral_tif_dir
    elif bool(args.export_spectral_tif):
        output_spectral_tif_dir = output_hdr_dir
    else:
        output_spectral_tif_dir = None

    output_hdr_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)
    output_rgb_dir.mkdir(parents=True, exist_ok=True)
    if output_spectral_tif_dir is not None:
        output_spectral_tif_dir.mkdir(parents=True, exist_ok=True)

    hdr_files = sorted([p for p in input_hdr_dir.glob("*.hdr") if p.is_file()])
    if not hdr_files:
        LOGGER.warning("No .hdr files found in %s", input_hdr_dir)
        return

    label_map = {p.stem: p for p in input_label_dir.glob("*.json") if p.is_file()}

    iterator: Iterable[Path] = hdr_files
    if tqdm is not None and not args.no_progress:
        iterator = tqdm(hdr_files, total=len(hdr_files), desc="Cropping", unit="img")

    total_removed = 0
    processed = 0
    skipped = 0
    exported_rgb = 0
    exported_tif = 0

    if float(args.black_thr) != 0.0 or int(args.erode_valid_px) != 0:
        LOGGER.warning(
            "Ignoring deprecated options: --black-thr=%.6g --erode-valid-px=%d (crop trims border rows/cols that are fully zero in any channel).",
            float(args.black_thr),
            int(args.erode_valid_px),
        )

    for hdr_path in iterator:
        stem = hdr_path.stem
        label_path = label_map.get(stem)

        if label_path is None and not args.keep_missing_label:
            skipped += 1
            continue

        out_hdr_path = output_hdr_dir / f"{stem}.hdr"
        out_label_path = output_label_dir / f"{stem}.json"
        if (
            not args.overwrite
            and out_hdr_path.exists()
            and (label_path is None or out_label_path.exists())
        ):
            skipped += 1
            continue

        try:
            cube, meta = load_envi_cube(hdr_path)
            rect = compute_allzero_border_crop_rect(cube)
            cropped = crop_cube(cube, rect)
        except Exception as exc:
            LOGGER.exception("Failed to crop %s: %s", hdr_path.name, exc)
            skipped += 1
            continue

        wavelengths = parse_wavelengths(meta, channels=int(cropped.shape[2]))
        band_names = parse_band_names(meta, wavelengths)
        save_hdr_img(
            cropped,
            str(out_hdr_path),
            band_list=wavelengths,
            band_names=band_names,
        )

        try:
            rgb_path = output_rgb_dir / f"{stem}.{str(args.rgb_ext).lstrip('.')}"
            if save_bgr_png(cropped[:, :, :3], rgb_path, overwrite=bool(args.overwrite)):
                exported_rgb += 1
        except Exception as exc:
            LOGGER.exception("Failed to export RGB for %s: %s", hdr_path.name, exc)

        if output_spectral_tif_dir is not None:
            try:
                tif_path = output_spectral_tif_dir / f"{stem}.tif"
                if save_spectral7_tif(
                    cropped,
                    tif_path,
                    overwrite=bool(args.overwrite),
                    channel_names=[f"{nm}nm" for nm in SPECTRAL_CHANNELS],
                ):
                    exported_tif += 1
            except Exception as exc:
                LOGGER.exception(
                    "Failed to export spectral TIF for %s: %s", hdr_path.name, exc
                )

        if label_path is None:
            processed += 1
            continue

        try:
            label_obj = json.loads(label_path.read_text(encoding="utf-8"))
        except Exception as exc:
            LOGGER.exception("Failed to read label %s: %s", label_path.name, exc)
            skipped += 1
            continue

        updated, removed = update_labelme_json_for_crop(
            label_obj,
            rect=rect,
            new_width=int(cropped.shape[1]),
            new_height=int(cropped.shape[0]),
            output_image_basename=f"{stem}.{str(args.rgb_ext).lstrip('.')}",
            set_image_path=str(args.set_image_path),
            drop_image_data=not bool(args.keep_image_data),
            min_polygon_area=float(args.min_polygon_area),
        )

        out_label_path.write_text(
            json.dumps(updated, ensure_ascii=False, indent=4) + "\n", encoding="utf-8"
        )
        total_removed += removed
        processed += 1

    LOGGER.info(
        "Done. processed=%d skipped=%d removed_shapes=%d output_hdr=%s output_labels=%s output_rgb=%s output_spectral_tif=%s exported_rgb=%d exported_tif=%d",
        processed,
        skipped,
        total_removed,
        output_hdr_dir,
        output_label_dir,
        output_rgb_dir,
        output_spectral_tif_dir,
        exported_rgb,
        exported_tif,
    )


if __name__ == "__main__":
    main()

"""
python data_align/crop_black_edges_update_labels.py \
--input-hdr-dir /mnt/d/Project/master-graduation-project/data/oil/oil_dataset_20260109_简单目标/aligned_full_tif_20260110-1126_s1-matchanything_s2-none \
--input-label-dir /mnt/d/Project/master-graduation-project/data/oil/oil_dataset_20260109_简单目标/xanylabeling \
--overwrite

python data_align/crop_black_edges_update_labels.py \
--input-hdr-dir /mnt/d/Project/master-graduation-project/data/oil/val/aligned_full_tif_20251231-2225_matchanything_affine_1440 \
--input-label-dir /mnt/d/Project/master-graduation-project/data/oil/val/labels_seg_5_XAnyLabeling \
--overwrite
"""
