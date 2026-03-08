
import argparse
import logging
import re
from datetime import datetime
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys

import cv2
import numpy as np
import tifffile

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None


"""
  两阶段配准（粗配准 + MatchAnything 微调）的示例命令行：
  - 第一步：使用 data/config/oil/my-conf 的平移参数做粗配准（等价于 data_align.py 的逻辑）
  - 第二步：使用 MatchAnything 做仿射/相似变换微调

--matchanything-estimator affine # 仿射变换
--matchanything-estimator similarity # 相似变换 更稳定

python data_align/data_align_matchanything_two_stage.py \
    --rgb-dir /mnt/d/Project/master-graduation-project/data/oil/oil_dataset_20260109_简单目标/images \
    --spectral-dir /mnt/d/Project/master-graduation-project/data/oil/oil_dataset_20260109_简单目标/msi \
    --config-dir data/config/oil/my-conf \
    --alignment-reference jpg \
    --secondary-topology chain \
    --chain-visible-rgb-ref channels \
    --stage1-method matchanything \
    --matchanything-model matchanything_roma \
    --matchanything-estimator similarity \
    --matchanything-device cuda:0 \
    --matchanything-imgresize 1440 
    


python data_align/data_align_matchanything_two_stage.py \
    --rgb-dir /mnt/d/Project/master-graduation-project/data/oil/oil_dataset_20260109_简单目标/images \
    --spectral-dir /mnt/d/Project/master-graduation-project/data/oil/oil_dataset_20260109_简单目标/msi \
    --config-dir data/config/oil/my-conf \
    --alignment-reference jpg \
    --stage1-method matchanything \
    --matchanything-model matchanything_roma \
    --matchanything-estimator similarity \
    --matchanything-device cuda:0 \
    --matchanything-imgresize 1440 


  MatchAnything 可视化结果：
  相邻波段配准
  python data_align/data_align_matchanything_two_stage.py \
    --rgb-dir tmp/register \
    --spectral-dir tmp/register \
    --config-dir data/config/oil/my-conf \
    --alignment-reference jpg \
    --secondary-topology chain \
    --chain-visible-rgb-ref channels \
    --stage1-method matchanything \
    --matchanything-model matchanything_roma \
    --matchanything-estimator similarity \
    --matchanything-device cuda:0 \
    --matchanything-imgresize 1440 \
    --output-full tmp/register/aligned_full_tif \
    --output-nir tmp/register/aligned_nir_tif \
    --overwrite \
    --save-match-vis \
    --save-pseudo-vis \
    --match-vis-use-raw
jpg为基准,无配置文件前置配准
  python data_align/data_align_matchanything_two_stage.py \
    --rgb-dir tmp/register \
    --spectral-dir tmp/register \
    --config-dir data/config/oil/my-conf \
    --alignment-reference jpg \
    --no-config-translation \
    --secondary-topology single \
    --stage1-method matchanything \
    --matchanything-model matchanything_roma \
    --matchanything-estimator similarity \
    --matchanything-device cuda:0 \
    --matchanything-imgresize 1440 \
    --output-full tmp/register/aligned_full_tif \
    --output-nir tmp/register/aligned_nir_tif \
    --overwrite \
    --save-match-vis \
    --save-pseudo-vis \
    --match-vis-use-raw

"""

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.io.spectral_io import (
    open_hdr_img,
    save_hdr_img,
    channel as SPECTRAL_CHANNELS,
)
from data_align.matchanything import (
    MatchAnythingSettings,
    estimate_matchanything_affine_transform_and_matches,
)

LOGGER = logging.getLogger(__name__)

DATE_PATTERN = re.compile(r"_(\d{8})_")
DATE_CUTOFF = datetime.strptime("20240513", "%Y%m%d")
BEFORE_CONFIG_NAME = "align_before_513.conf"
AFTER_CONFIG_NAME = "align_after_513.conf"

CHANNEL_ORDER: Tuple[object, ...] = ("jpg", *SPECTRAL_CHANNELS)

# ENVI wavelength list for the first 3 RGB channels.
# Note: This will overlap with spectral bands (450/550/650nm) in the same cube; ENVI allows it,
# but if you need unique wavelengths for downstream tools, consider using (447, 448, 449).
RGB_PLACEHOLDER_BANDS: Tuple[int, int, int] = (150, 250, 350)
DEFAULT_CHAIN_ANCHOR_BANDS: Tuple[int, int, int] = (450, 550, 650)
PSEUDO_BGR_WAVELENGTHS: Tuple[int, int, int] = (450, 550, 650)


def parse_int_list_csv(value: str) -> Tuple[int, ...]:
    """Parse comma-separated integers (e.g. '450,550,650')."""
    if value is None:
        return tuple()
    value = value.strip()
    if not value:
        return tuple()
    items: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            items.append(int(part))
        except ValueError as exc:  # pragma: no cover
            raise argparse.ArgumentTypeError(
                f"Invalid integer '{part}' in '{value}'"
            ) from exc
    return tuple(items)


def parse_alignment_reference(value: str) -> object:
    lowered = value.lower()
    if lowered == "jpg":
        return "jpg"
    try:
        numeric = int(value)
    except ValueError as exc:
        valid = ", ".join(str(band) for band in SPECTRAL_CHANNELS)
        raise argparse.ArgumentTypeError(
            f"Invalid alignment reference '{value}'. Use 'jpg' or one of: {valid}."
        ) from exc
    if numeric not in SPECTRAL_CHANNELS:
        valid = ", ".join(str(band) for band in SPECTRAL_CHANNELS)
        raise argparse.ArgumentTypeError(
            f"Unsupported alignment reference '{numeric}'. Use 'jpg' or one of: {valid}."
        )
    return numeric


def load_alignment_config(
    config_path: Path, reference: object
) -> Dict[object, Tuple[int, int]]:
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing alignment config: {config_path}")

    offsets: Dict[object, Tuple[int, int]] = {}
    parsed_values: List[Tuple[int, int]] = []

    with config_path.open("r", encoding="utf-8") as cfg:
        # 兼容历史错误：早期生成器误把换行写成字面量 “/n”，导致整份文件变成单行。
        content = cfg.read().replace("/n", "\n")
        for line in content.splitlines():
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",") if p.strip()]
            if len(parts) < 2:
                continue
            dx, dy = (int(float(parts[0])), int(float(parts[1])))
            parsed_values.append((dx, dy))

    expected = len(CHANNEL_ORDER)
    if len(parsed_values) < expected:
        raise ValueError(
            f"Alignment config {config_path} expected {expected} entries, "
            f"found {len(parsed_values)}."
        )

    for key, offset in zip(CHANNEL_ORDER, parsed_values[:expected]):
        offsets[key] = offset

    if "jpg" not in offsets:
        raise ValueError("Alignment config missing JPG reference offset.")

    base_offset = offsets.get(reference)
    if base_offset is None:
        raise ValueError(
            f"Alignment config missing reference offset for {reference}."
        )

    base_dx, base_dy = base_offset
    normalized_offsets: Dict[object, Tuple[int, int]] = {}
    for key, (dx, dy) in offsets.items():
        normalized_offsets[key] = (int(base_dx - dx), int(base_dy - dy))

    return normalized_offsets


def extract_capture_date(filename: str) -> datetime:
    match = DATE_PATTERN.search(filename)
    if not match:
        raise ValueError(f"Unable to extract 8-digit date from '{filename}'")
    return datetime.strptime(match.group(1), "%Y%m%d")


def select_offsets(
    filename: str,
    before: Dict[object, Tuple[int, int]],
    after: Dict[object, Tuple[int, int]],
    cutoff: datetime = DATE_CUTOFF,
) -> Dict[object, Tuple[int, int]]:
    capture_date = extract_capture_date(filename)
    # 与 cal_offset.py 保持一致：cutoff 当日及之后归为 “后”
    return before if capture_date < cutoff else after


def apply_pixel_offset(
    image: np.ndarray,
    offset: Tuple[int, int],
    fill_value: float = 0,
    preserve_full: bool = False,
    interpolation: int = cv2.INTER_LINEAR,
) -> Tuple[np.ndarray, Tuple[int, int]] | np.ndarray:
    dx, dy = (int(offset[0]), int(offset[1]))
    height, width = image.shape[:2]
    image = np.ascontiguousarray(image)

    if not preserve_full:
        transform = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
        border = (
            (float(fill_value),) * image.shape[2]
            if image.ndim == 3
            else float(fill_value)
        )
        shifted = cv2.warpAffine(
            image,
            transform,
            (width, height),
            flags=interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border,
        )
        return shifted

    pad_x = max(-dx, 0)
    pad_y = max(-dy, 0)
    new_width = width + abs(dx)
    new_height = height + abs(dy)
    transform = np.array(
        [[1, 0, dx + pad_x], [0, 1, dy + pad_y]], dtype=np.float32
    )
    border = (
        (float(fill_value),) * image.shape[2]
        if image.ndim == 3
        else float(fill_value)
    )
    shifted = cv2.warpAffine(
        image,
        transform,
        (new_width, new_height),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border,
    )
    # crop_offset 表示“由于负向平移而引入的左/上 padding”，用于将 preserve_full 的结果裁回原尺寸。
    # 以前这里错误地用了 (max(dy,0), max(dx,0))，会在 crop_to_shape 时把平移效果抵消掉。
    crop_offset = (pad_y, pad_x)
    return shifted, crop_offset


def normalize_to_uint8(band: np.ndarray, clip_percentile: float) -> np.ndarray:
    valid = band[np.isfinite(band)]
    if valid.size == 0:
        return np.zeros_like(band, dtype=np.uint8)

    lo = np.percentile(valid, clip_percentile)
    hi = np.percentile(valid, 100 - clip_percentile)

    if hi <= lo:
        lo = valid.min()
        hi = valid.max()
        if hi <= lo:
            return np.zeros_like(band, dtype=np.uint8)

    scaled = np.clip((band - lo) / (hi - lo), 0, 1)
    return (scaled * 255).astype(np.uint8)


def crop_to_shape(
    image: np.ndarray, offset: Tuple[int, int], height: int, width: int
) -> np.ndarray:
    y, x = offset
    y_end = min(y + height, image.shape[0])
    x_end = min(x + width, image.shape[1])
    slices = (slice(y, y_end), slice(x, x_end))
    if image.ndim == 3:
        slices = (*slices, slice(None))
    cropped = image[slices]
    if cropped.shape[0] != height or cropped.shape[1] != width:
        pad_h = height - cropped.shape[0]
        pad_w = width - cropped.shape[1]
        padding = (
            (0, max(pad_h, 0)),
            (0, max(pad_w, 0)),
        )
        if image.ndim == 3:
            padding = (*padding, (0, 0))
        cropped = np.pad(cropped, padding, mode="constant", constant_values=0)
    return cropped


def largest_true_rectangle(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
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

    return best_rect


def crop_stack_to_mask(
    rgb: np.ndarray,
    spectral_stack: np.ndarray,
    mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rect = largest_true_rectangle(mask)
    if rect is None:
        raise RuntimeError("Unable to find zero-free rectangle for mask.")

    top, bottom, left, right = rect
    rgb_crop = rgb[top:bottom, left:right, :]
    spectral_crop = spectral_stack[top:bottom, left:right, :]
    mask_crop = mask[top:bottom, left:right]

    if mask_crop.size == 0 or not mask_crop.all():
        raise RuntimeError("Identified rectangle still contains invalid pixels.")

    return rgb_crop, spectral_crop, mask_crop


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    raise ValueError("Unsupported image shape for grayscale conversion.")


def _resize_keep_aspect_vis(
    image: np.ndarray, *, max_dim: int
) -> Tuple[np.ndarray, float]:
    """Resize for visualization only (keep aspect), return (resized, scale)."""
    if max_dim <= 0:
        return image, 1.0
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_dim:
        return image, 1.0
    scale = float(max_dim) / float(longest)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def _as_u8_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim != 2:
        raise ValueError("Expected 2D grayscale image.")
    if image.dtype == np.uint8:
        return image
    # Robust fallback: scale finite values to [0, 255].
    img_f = image.astype(np.float32, copy=False)
    finite = img_f[np.isfinite(img_f)]
    if finite.size == 0:
        return np.zeros_like(image, dtype=np.uint8)
    lo = float(np.percentile(finite, 1.0))
    hi = float(np.percentile(finite, 99.0))
    if hi <= lo:
        lo = float(finite.min())
        hi = float(finite.max())
        if hi <= lo:
            return np.zeros_like(image, dtype=np.uint8)
    scaled = np.clip((img_f - lo) / (hi - lo), 0, 1)
    return (scaled * 255.0).astype(np.uint8)


def draw_match_vis(
    reference: np.ndarray,
    target: np.ndarray,
    ref_pts: np.ndarray,
    tgt_pts: np.ndarray,
    *,
    inliers: Optional[np.ndarray] = None,
    confidences: Optional[np.ndarray] = None,
    max_matches: int = 200,
    layout: str = "lr",
    max_dim: int = 1600,
    inliers_only: bool = False,
    title: Optional[str] = None,
) -> np.ndarray:
    """Create a side-by-side match visualization (like LoFTR/ROMA demos).

    - `layout="lr"`: reference on the left, target on the right.
    - `layout="tb"`: reference on the top, target on the bottom.
    """
    if layout not in {"lr", "tb"}:
        raise ValueError(f"Unsupported layout: {layout} (expected 'lr' or 'tb').")
    if ref_pts.ndim != 2 or ref_pts.shape[1] != 2:
        raise ValueError(f"ref_pts must be Nx2, got {ref_pts.shape}")
    if tgt_pts.ndim != 2 or tgt_pts.shape[1] != 2:
        raise ValueError(f"tgt_pts must be Nx2, got {tgt_pts.shape}")
    if ref_pts.shape[0] != tgt_pts.shape[0]:
        raise ValueError(
            f"ref_pts and tgt_pts must have same length, got {ref_pts.shape[0]} vs {tgt_pts.shape[0]}"
        )

    ref_u8 = _as_u8_gray(reference)
    tgt_u8 = _as_u8_gray(target)

    ref_u8, s_ref = _resize_keep_aspect_vis(ref_u8, max_dim=max_dim)
    tgt_u8, s_tgt = _resize_keep_aspect_vis(tgt_u8, max_dim=max_dim)
    ref_pts_s = ref_pts.astype(np.float32, copy=False) * float(s_ref)
    tgt_pts_s = tgt_pts.astype(np.float32, copy=False) * float(s_tgt)

    ref_bgr = cv2.cvtColor(ref_u8, cv2.COLOR_GRAY2BGR)
    tgt_bgr = cv2.cvtColor(tgt_u8, cv2.COLOR_GRAY2BGR)
    h1, w1 = ref_bgr.shape[:2]
    h2, w2 = tgt_bgr.shape[:2]

    inlier_mask: Optional[np.ndarray] = None
    if inliers is not None:
        inlier_mask = np.asarray(inliers).astype(bool).reshape(-1)
        if inlier_mask.shape[0] != ref_pts_s.shape[0]:
            raise ValueError(
                f"inliers must have length {ref_pts_s.shape[0]}, got {inlier_mask.shape[0]}"
            )

    idx = np.arange(ref_pts_s.shape[0])
    if inliers_only and inlier_mask is not None:
        idx = idx[inlier_mask]

    if idx.size == 0:
        # No matches to draw; still return the concatenated image.
        if layout == "tb":
            canvas = np.zeros((h1 + h2, max(w1, w2), 3), dtype=np.uint8)
            canvas[:h1, :w1] = ref_bgr
            canvas[h1 : h1 + h2, :w2] = tgt_bgr
        else:
            canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
            canvas[:h1, :w1] = ref_bgr
            canvas[:h2, w1 : w1 + w2] = tgt_bgr
        if title:
            cv2.putText(
                canvas,
                title,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        return canvas

    # Choose which matches to draw.
    if confidences is not None:
        conf = np.asarray(confidences).reshape(-1)
        if conf.shape[0] != ref_pts_s.shape[0]:
            raise ValueError(
                f"confidences must have length {ref_pts_s.shape[0]}, got {conf.shape[0]}"
            )
        order = np.argsort(-conf[idx])
        idx = idx[order]

    if max_matches > 0 and idx.size > max_matches:
        idx = idx[: int(max_matches)]

    if layout == "tb":
        canvas = np.zeros((h1 + h2, max(w1, w2), 3), dtype=np.uint8)
        canvas[:h1, :w1] = ref_bgr
        canvas[h1 : h1 + h2, :w2] = tgt_bgr
        cv2.line(canvas, (0, h1), (canvas.shape[1] - 1, h1), (0, 0, 255), 2)
        for i in idx.tolist():
            p1 = ref_pts_s[i]
            p2 = tgt_pts_s[i]
            x1, y1 = int(round(float(p1[0]))), int(round(float(p1[1])))
            x2, y2 = int(round(float(p2[0]))), int(round(float(p2[1]) + h1))
            if inlier_mask is not None and not bool(inlier_mask[i]):
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)
            cv2.line(canvas, (x1, y1), (x2, y2), color, 1)
            cv2.circle(canvas, (x1, y1), 2, (255, 0, 0), -1)
            cv2.circle(canvas, (x2, y2), 2, (0, 0, 255), -1)
    else:
        canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        canvas[:h1, :w1] = ref_bgr
        canvas[:h2, w1 : w1 + w2] = tgt_bgr
        cv2.line(canvas, (w1, 0), (w1, canvas.shape[0] - 1), (0, 0, 255), 2)
        for i in idx.tolist():
            p1 = ref_pts_s[i]
            p2 = tgt_pts_s[i]
            x1, y1 = int(round(float(p1[0]))), int(round(float(p1[1])))
            x2, y2 = int(round(float(p2[0]) + w1)), int(round(float(p2[1])))
            if inlier_mask is not None and not bool(inlier_mask[i]):
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)
            cv2.line(canvas, (x1, y1), (x2, y2), color, 1)
            cv2.circle(canvas, (x1, y1), 2, (255, 0, 0), -1)
            cv2.circle(canvas, (x2, y2), 2, (0, 0, 255), -1)

    if title:
        cv2.putText(
            canvas,
            title,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return canvas


def _make_pseudo_bgr(
    band_b: np.ndarray,
    band_g: np.ndarray,
    band_r: np.ndarray,
    *,
    clip_percentile: float,
) -> np.ndarray:
    b = normalize_to_uint8(band_b, clip_percentile)
    g = normalize_to_uint8(band_g, clip_percentile)
    r = normalize_to_uint8(band_r, clip_percentile)
    return np.stack([b, g, r], axis=2)


def _draw_pseudo_vis(
    pre_bgr: np.ndarray,
    post_bgr: np.ndarray,
    *,
    layout: str,
    max_dim: int,
) -> np.ndarray:
    if layout not in {"lr", "tb"}:
        raise ValueError(f"Unsupported layout: {layout} (expected 'lr' or 'tb').")
    pre_vis, _ = _resize_keep_aspect_vis(pre_bgr, max_dim=max_dim)
    post_vis, _ = _resize_keep_aspect_vis(post_bgr, max_dim=max_dim)
    h1, w1 = pre_vis.shape[:2]
    h2, w2 = post_vis.shape[:2]
    if layout == "tb":
        canvas = np.zeros((h1 + h2, max(w1, w2), 3), dtype=np.uint8)
        canvas[:h1, :w1] = pre_vis
        canvas[h1 : h1 + h2, :w2] = post_vis
    else:
        canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        canvas[:h1, :w1] = pre_vis
        canvas[:h2, w1 : w1 + w2] = post_vis
    return canvas


def _crop_to_rect(image: np.ndarray, rect: Tuple[int, int, int, int]) -> np.ndarray:
    top, bottom, left, right = rect
    if image.ndim == 2:
        return image[top:bottom, left:right]
    return image[top:bottom, left:right, :]


def _map_preserved_to_raw(
    pts: np.ndarray, shift_xy: Tuple[float, float], shape: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """Map preserved coords to raw coords via translation, return (pts_raw, mask_in_bounds)."""
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"pts must be Nx2, got {pts.shape}")
    shift_x, shift_y = float(shift_xy[0]), float(shift_xy[1])
    pts_raw = pts.astype(np.float32, copy=False).copy()
    pts_raw[:, 0] -= shift_x
    pts_raw[:, 1] -= shift_y
    h, w = int(shape[0]), int(shape[1])
    in_bounds = (
        (pts_raw[:, 0] >= 0)
        & (pts_raw[:, 0] <= float(w - 1))
        & (pts_raw[:, 1] >= 0)
        & (pts_raw[:, 1] <= float(h - 1))
    )
    return pts_raw, in_bounds


def estimate_affine_transform(
    reference: np.ndarray, target: np.ndarray
) -> Optional[np.ndarray]:
    orb = cv2.ORB_create(nfeatures=2000)
    kp_ref, des_ref = orb.detectAndCompute(reference, None)
    kp_tgt, des_tgt = orb.detectAndCompute(target, None)

    if (
        des_ref is None
        or des_tgt is None
        or len(kp_ref) < 10
        or len(kp_tgt) < 10
    ):
        return None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des_ref, des_tgt)
    if len(matches) < 8:
        return None

    matches = sorted(matches, key=lambda m: m.distance)[:200]
    ref_pts = np.float32([kp_ref[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    tgt_pts = np.float32([kp_tgt[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    matrix, inliers = cv2.estimateAffinePartial2D(
        tgt_pts,
        ref_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=2000,
        confidence=0.99,
    )
    if matrix is None or inliers is None or np.count_nonzero(inliers) < 8:
        return None

    scale_x = float(np.hypot(matrix[0, 0], matrix[0, 1]))
    scale_y = float(np.hypot(matrix[1, 0], matrix[1, 1]))
    if not (0.7 <= scale_x <= 1.3 and 0.7 <= scale_y <= 1.3):
        LOGGER.debug(
            "Rejecting affine transform due to scale anomaly "
            "(scale_x=%.3f, scale_y=%.3f)",
            scale_x,
            scale_y,
        )
        return None

    return matrix


def _center_crop_match(
    reference: np.ndarray, target: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int], Tuple[int, int]]:
    """Center-crop two 2D images to the same (min_h, min_w) size.

    Returns (ref_c, tgt_c, (rx0, ry0), (tx0, ty0)) where x0/y0 are crop origins.
    """
    href, wref = reference.shape[:2]
    htgt, wtgt = target.shape[:2]
    h = min(href, htgt)
    w = min(wref, wtgt)
    ry0 = (href - h) // 2
    rx0 = (wref - w) // 2
    ty0 = (htgt - h) // 2
    tx0 = (wtgt - w) // 2
    ref_c = reference[ry0 : ry0 + h, rx0 : rx0 + w]
    tgt_c = target[ty0 : ty0 + h, tx0 : tx0 + w]
    return ref_c, tgt_c, (rx0, ry0), (tx0, ty0)


def estimate_ecc_transform(
    reference: np.ndarray,
    target: np.ndarray,
    *,
    motion: str,
    reference_mask: Optional[np.ndarray],
    target_mask: Optional[np.ndarray],
    iters: int,
    eps: float,
) -> Optional[np.ndarray]:
    """Estimate a warp mapping `target -> reference` using ECC.

    Notes:
    - cv2.findTransformECC requires same-sized inputs; we center-crop to common size.
    - cv2.findTransformECC returns a warp that maps template->input (reference->target);
      we lift it back to full coordinates and invert it to get `target -> reference`.
    """
    if reference.ndim != 2 or target.ndim != 2:
        raise ValueError("ECC expects 2D grayscale inputs.")

    ref_c, tgt_c, (rx0, ry0), (tx0, ty0) = _center_crop_match(reference, target)
    ref_f = ref_c.astype(np.float32, copy=False)
    tgt_f = tgt_c.astype(np.float32, copy=False)

    mask_c: Optional[np.ndarray] = None
    if reference_mask is not None and target_mask is not None:
        refm_c = reference_mask[ry0 : ry0 + ref_c.shape[0], rx0 : rx0 + ref_c.shape[1]]
        tgtm_c = target_mask[ty0 : ty0 + tgt_c.shape[0], tx0 : tx0 + tgt_c.shape[1]]
        mask_c = (refm_c.astype(bool) & tgtm_c.astype(bool)).astype(np.uint8) * 255

    if motion == "translation":
        motion_type = cv2.MOTION_TRANSLATION
        warp = np.eye(2, 3, dtype=np.float32)
    elif motion == "affine":
        motion_type = cv2.MOTION_AFFINE
        warp = np.eye(2, 3, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported ECC motion: {motion}")

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, int(iters), float(eps))
    try:
        _, warp_out = cv2.findTransformECC(
            ref_f,
            tgt_f,
            warp,
            motion_type,
            criteria,
            inputMask=mask_c,
        )
    except cv2.error:
        return None

    if motion == "affine":
        scale_x = float(np.hypot(warp_out[0, 0], warp_out[0, 1]))
        scale_y = float(np.hypot(warp_out[1, 0], warp_out[1, 1]))
        if not (0.7 <= scale_x <= 1.3 and 0.7 <= scale_y <= 1.3):
            LOGGER.debug(
                "Rejecting ECC affine due to scale anomaly (scale_x=%.3f, scale_y=%.3f)",
                scale_x,
                scale_y,
            )
            return None

    # Lift to full coords: W_full(ref->tgt) = T_tgt_inv @ W_crop @ T_ref
    warp3 = np.eye(3, dtype=np.float32)
    warp3[:2, :] = warp_out
    T_ref = np.array([[1, 0, -rx0], [0, 1, -ry0], [0, 0, 1]], dtype=np.float32)
    T_tgt_inv = np.array([[1, 0, tx0], [0, 1, ty0], [0, 0, 1]], dtype=np.float32)
    warp_ref_to_tgt = T_tgt_inv @ warp3 @ T_ref

    try:
        warp_tgt_to_ref = np.linalg.inv(warp_ref_to_tgt).astype(np.float32)
    except np.linalg.LinAlgError:
        return None

    return warp_tgt_to_ref[:2, :]


def _lift_aligned_warp_to_preserved(
    matrix_aligned_tgt_to_ref: np.ndarray,
    *,
    ref_crop_offset: Tuple[int, int],
    tgt_crop_offset: Tuple[int, int],
) -> np.ndarray:
    """Lift a warp estimated on aligned (cropped) images back to preserved-image coordinates.

    aligned = crop_to_shape(preserved, crop_offset, H, W)
    so: x_aligned = x_preserved - crop_offset
    If M maps target_aligned -> ref_aligned, then:
      x_ref_preserved = T(ref_off) * M * T(-tgt_off) * x_tgt_preserved
    """
    ry, rx = ref_crop_offset
    ty, tx = tgt_crop_offset
    T_ref = np.array([[1, 0, rx], [0, 1, ry], [0, 0, 1]], dtype=np.float32)
    T_tgt_inv = np.array([[1, 0, -tx], [0, 1, -ty], [0, 0, 1]], dtype=np.float32)

    m3 = np.eye(3, dtype=np.float32)
    m3[:2, :] = matrix_aligned_tgt_to_ref.astype(np.float32, copy=False)
    lifted = T_ref @ m3 @ T_tgt_inv
    return lifted[:2, :]


def warp_affine(
    image: np.ndarray,
    matrix: Optional[np.ndarray],
    output_shape: Tuple[int, int],
    fill_value: int = 0,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    if matrix is None:
        matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

    border = (
        (float(fill_value),) * image.shape[2]
        if image.ndim == 3
        else float(fill_value)
    )
    return cv2.warpAffine(
        image,
        matrix,
        (output_shape[1], output_shape[0]),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border,
    )


def process_sample(
    image_path: Path,
    spectral_path: Path,
    offsets: Dict[object, Tuple[int, int]],
    alignment_reference: object,
    secondary_topology: str,
    chain_anchor_bands: Tuple[int, ...],
    chain_visible_rgb_ref: str,
    clip_percentile: float,
    enable_secondary_alignment: bool,
    stage1_method: str,
    stage2_method: str,
    matchanything: Optional[MatchAnythingSettings],
    ecc_motion: str,
    ecc_iters: int,
    ecc_eps: float,
    crop_black_edges: bool,
    *,
    match_vis_dir: Optional[Path] = None,
    match_vis_max_matches: int = 200,
    match_vis_layout: str = "lr",
    match_vis_max_dim: int = 1600,
    match_vis_inliers_only: bool = False,
    match_vis_use_raw: bool = False,
    match_vis_prefix: Optional[str] = None,
    pseudo_vis_dir: Optional[Path] = None,
    pseudo_vis_layout: str = "lr",
    pseudo_vis_max_dim: int = 1600,
) -> Tuple[np.ndarray, np.ndarray, Tuple[str, ...]]:
    bgr_image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if bgr_image is None:
        raise RuntimeError(f"Failed to load RGB image: {image_path}")

    if bgr_image.ndim == 2:
        bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_GRAY2BGR)
    elif bgr_image.shape[2] == 4:
        bgr_image = bgr_image[:, :, :3]
    elif bgr_image.shape[2] != 3:
        raise ValueError(
            f"Unsupported channel count {bgr_image.shape[2]} in {image_path}"
        )

    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB).astype(np.uint8)
    rgb_height, rgb_width = rgb.shape[:2]
    rgb_raw_gray = to_grayscale(rgb)
    rgb_preserved, rgb_crop_offset = apply_pixel_offset(
        rgb, offsets["jpg"], fill_value=0, preserve_full=True
    )
    rgb_aligned = crop_to_shape(rgb_preserved, rgb_crop_offset, rgb_height, rgb_width)
    rgb_gray_preserved = to_grayscale(rgb_preserved)
    rgb_mask = np.ones((rgb_height, rgb_width), dtype=np.uint8)
    rgb_mask_preserved, rgb_mask_offset = apply_pixel_offset(
        rgb_mask,
        offsets["jpg"],
        fill_value=0,
        preserve_full=True,
        interpolation=cv2.INTER_NEAREST,
    )
    rgb_mask_aligned = crop_to_shape(
        rgb_mask_preserved, rgb_mask_offset, rgb_height, rgb_width
    ).astype(bool)

    hdr_cube = open_hdr_img(str(spectral_path))
    if hdr_cube is None:
        raise RuntimeError(f"Failed to open HDR cube: {spectral_path}")

    if hdr_cube.ndim == 2:
        hdr_cube = hdr_cube[..., np.newaxis]

    spectral_height, spectral_width = hdr_cube.shape[:2]
    # Keep two views:
    # - *_raw: original dtype values used for warping + saving (preserve bit depth).
    # - *_u8: uint8-normalized views used only for matching/alignment algorithms.
    spectral_preserved_raw: Dict[int, np.ndarray] = {}
    spectral_preserved_u8: Dict[int, np.ndarray] = {}
    spectral_raw_u8: Dict[int, np.ndarray] = {}
    spectral_crop_offsets: Dict[int, Tuple[int, int]] = {}
    spectral_shifts: Dict[int, Tuple[float, float]] = {}
    aligned_bands_raw: Dict[int, np.ndarray] = {}
    aligned_bands_u8: Dict[int, np.ndarray] = {}
    spectral_masks_preserved: Dict[int, np.ndarray] = {}
    spectral_mask_offsets: Dict[int, Tuple[int, int]] = {}
    aligned_masks: Dict[int, np.ndarray] = {}

    for idx, band_nm in enumerate(SPECTRAL_CHANNELS):
        band = hdr_cube[:, :, idx]
        spectral_raw_u8[band_nm] = normalize_to_uint8(band, clip_percentile)
        preserved_raw, crop_offset = apply_pixel_offset(
            band, offsets[band_nm], fill_value=0, preserve_full=True
        )
        dx, dy = (int(offsets[band_nm][0]), int(offsets[band_nm][1]))
        pad_y, pad_x = crop_offset
        spectral_shifts[band_nm] = (float(dx + pad_x), float(dy + pad_y))
        preserved_u8 = normalize_to_uint8(preserved_raw, clip_percentile)
        spectral_preserved_raw[band_nm] = preserved_raw
        spectral_preserved_u8[band_nm] = preserved_u8
        spectral_crop_offsets[band_nm] = crop_offset
        aligned_bands_raw[band_nm] = crop_to_shape(
            preserved_raw, crop_offset, spectral_height, spectral_width
        )
        aligned_bands_u8[band_nm] = crop_to_shape(
            preserved_u8, crop_offset, spectral_height, spectral_width
        )
        mask = np.ones((spectral_height, spectral_width), dtype=np.uint8)
        mask_preserved, mask_offset = apply_pixel_offset(
            mask,
            offsets[band_nm],
            fill_value=0,
            preserve_full=True,
            interpolation=cv2.INTER_NEAREST,
        )
        spectral_masks_preserved[band_nm] = mask_preserved
        spectral_mask_offsets[band_nm] = mask_offset
        aligned_masks[band_nm] = crop_to_shape(
            mask_preserved, mask_offset, spectral_height, spectral_width
        ).astype(bool)

    pseudo_pre_bands: Optional[Dict[int, np.ndarray]] = None
    if pseudo_vis_dir is not None:
        missing = [band for band in PSEUDO_BGR_WAVELENGTHS if band not in aligned_bands_raw]
        if missing:
            LOGGER.warning(
                "Skipping pseudo-vis for %s (missing bands: %s).",
                image_path.name,
                ",".join(str(band) for band in missing),
            )
        else:
            pseudo_pre_bands = {
                band: aligned_bands_raw[band].copy() for band in PSEUDO_BGR_WAVELENGTHS
            }

    reference_key = alignment_reference
    if reference_key == "jpg":
        reference_preserved = rgb_preserved
        reference_crop_offset = rgb_crop_offset
        reference_gray = rgb_gray_preserved
    else:
        reference_preserved = spectral_preserved_raw[int(reference_key)]
        reference_crop_offset = spectral_crop_offsets[int(reference_key)]
        reference_gray = spectral_preserved_u8[int(reference_key)]

    if enable_secondary_alignment and (stage1_method != "none" or stage2_method != "none"):
        rgb_channel_map = {450: 2, 550: 1, 650: 0}  # wavelength -> RGB channel index (R=0,G=1,B=2)

        def rgb_ref_gray_preserved(band_hint: Optional[int]) -> np.ndarray:
            if (
                chain_visible_rgb_ref == "channels"
                and band_hint is not None
                and int(band_hint) in rgb_channel_map
            ):
                return np.ascontiguousarray(
                    rgb_preserved[:, :, rgb_channel_map[int(band_hint)]]
                )
            return rgb_gray_preserved

        def rgb_ref_gray_aligned(band_hint: Optional[int]) -> np.ndarray:
            if (
                chain_visible_rgb_ref == "channels"
                and band_hint is not None
                and int(band_hint) in rgb_channel_map
            ):
                return np.ascontiguousarray(
                    rgb_aligned[:, :, rgb_channel_map[int(band_hint)]]
                )
            return to_grayscale(rgb_aligned)

        def get_preserved_image(key: object) -> np.ndarray:
            return rgb_preserved if key == "jpg" else spectral_preserved_raw[int(key)]

        def get_gray_preserved(key: object, *, rgb_band_hint: Optional[int] = None) -> np.ndarray:
            return rgb_ref_gray_preserved(rgb_band_hint) if key == "jpg" else spectral_preserved_u8[int(key)]

        def get_gray_aligned(key: object, *, rgb_band_hint: Optional[int] = None) -> np.ndarray:
            return rgb_ref_gray_aligned(rgb_band_hint) if key == "jpg" else aligned_bands_u8[int(key)]

        def get_mask_preserved(key: object) -> np.ndarray:
            return rgb_mask_preserved if key == "jpg" else spectral_masks_preserved[int(key)]

        def get_mask_aligned(key: object) -> np.ndarray:
            return rgb_mask_aligned if key == "jpg" else aligned_masks[int(key)]

        def get_crop_offset(key: object) -> Tuple[int, int]:
            return rgb_crop_offset if key == "jpg" else spectral_crop_offsets[int(key)]

        jobs: list[tuple[object, object, Optional[int]]] = []
        if secondary_topology == "single":
            if reference_key != "jpg":
                jobs.append(("jpg", reference_key, None))
            for band_nm in SPECTRAL_CHANNELS:
                if band_nm == reference_key:
                    continue
                jobs.append((band_nm, reference_key, None))
        elif secondary_topology == "chain":
            if reference_key != "jpg":
                raise ValueError(
                    "secondary_topology=chain requires --alignment-reference jpg "
                    "to keep outputs anchored to RGB/labels."
                )
            anchor_set = {
                int(band) for band in chain_anchor_bands if int(band) in SPECTRAL_CHANNELS
            }
            spectral_order = list(SPECTRAL_CHANNELS)
            for idx, band_nm in enumerate(spectral_order):
                if idx == 0 or band_nm in anchor_set:
                    jobs.append((band_nm, "jpg", int(band_nm)))
                else:
                    jobs.append((band_nm, spectral_order[idx - 1], None))
        else:
            raise ValueError(f"Unknown secondary_topology: {secondary_topology}")

        refined_keys: set[object] = set()
        rgb_shift = (
            float(int(offsets["jpg"][0]) + rgb_crop_offset[1]),
            float(int(offsets["jpg"][1]) + rgb_crop_offset[0]),
        )

        def get_raw_gray(key: object) -> np.ndarray:
            return rgb_raw_gray if key == "jpg" else spectral_raw_u8[int(key)]

        def get_shift(key: object) -> Tuple[float, float]:
            return rgb_shift if key == "jpg" else spectral_shifts[int(key)]

        for tgt_key, ref_key_job, rgb_band_hint in jobs:
            ref_preserved = get_preserved_image(ref_key_job)
            ref_height, ref_width = ref_preserved.shape[:2]
            ref_crop_off = get_crop_offset(ref_key_job)
            ref_gray = get_gray_preserved(ref_key_job, rgb_band_hint=rgb_band_hint)
            tgt_gray = get_gray_preserved(tgt_key)
            ref_mask_preserved = get_mask_preserved(ref_key_job)
            tgt_mask_preserved = get_mask_preserved(tgt_key)

            def apply_refine_current(matrix: np.ndarray) -> None:
                nonlocal rgb_preserved, rgb_aligned, rgb_gray_preserved, rgb_crop_offset, rgb_mask_preserved, rgb_mask_aligned

                # IMPORTANT: for multi-stage alignment we must warp the *current* images.
                if tgt_key == "jpg":
                    cur_img = rgb_preserved
                    cur_mask = rgb_mask_preserved
                else:
                    cur_img = spectral_preserved_raw[int(tgt_key)]
                    cur_mask = spectral_masks_preserved[int(tgt_key)]

                refined_raw = warp_affine(
                    cur_img,
                    matrix,
                    (ref_height, ref_width),
                    fill_value=0,
                )
                refined_mask = warp_affine(
                    cur_mask,
                    matrix,
                    (ref_height, ref_width),
                    fill_value=0,
                    interpolation=cv2.INTER_NEAREST,
                )
                refined_mask_u8 = (refined_mask > 0).astype(np.uint8)
                refined_u8 = (
                    refined_raw
                    if refined_raw.dtype == np.uint8
                    else normalize_to_uint8(refined_raw, clip_percentile)
                )

                if tgt_key == "jpg":
                    rgb_preserved = refined_raw
                    rgb_aligned = crop_to_shape(
                        refined_raw, ref_crop_off, rgb_height, rgb_width
                    )
                    rgb_gray_preserved = to_grayscale(refined_raw)
                    rgb_crop_offset = ref_crop_off
                    rgb_mask_preserved = refined_mask_u8
                    rgb_mask_aligned = crop_to_shape(
                        refined_mask_u8, ref_crop_off, rgb_height, rgb_width
                    ).astype(bool)
                else:
                    tgt_band = int(tgt_key)
                    spectral_preserved_raw[tgt_band] = refined_raw
                    spectral_preserved_u8[tgt_band] = refined_u8
                    spectral_masks_preserved[tgt_band] = refined_mask_u8
                    spectral_crop_offsets[tgt_band] = ref_crop_off
                    aligned_bands_raw[tgt_band] = crop_to_shape(
                        refined_raw,
                        spectral_crop_offsets[tgt_band],
                        spectral_height,
                        spectral_width,
                    )
                    aligned_bands_u8[tgt_band] = crop_to_shape(
                        refined_u8,
                        spectral_crop_offsets[tgt_band],
                        spectral_height,
                        spectral_width,
                    )
                    aligned_masks[tgt_band] = crop_to_shape(
                        refined_mask_u8,
                        spectral_crop_offsets[tgt_band],
                        spectral_height,
                        spectral_width,
                    ).astype(bool)
                refined_keys.add(tgt_key)

            # Stage 1: coarse
            if stage1_method == "orb":
                m1 = estimate_affine_transform(ref_gray, tgt_gray)
                if m1 is not None:
                    apply_refine_current(m1)
                else:
                    LOGGER.debug("Stage1 ORB failed for %s (ref=%s)", tgt_key, ref_key_job)
            elif stage1_method == "matchanything":
                if matchanything is None:
                    raise ValueError(
                        "stage1_method=matchanything requires MatchAnything settings."
                    )
                result = estimate_matchanything_affine_transform_and_matches(
                    reference=ref_gray,
                    target=tgt_gray,
                    ref_mask=ref_mask_preserved,
                    tgt_mask=tgt_mask_preserved,
                    settings=matchanything,
                )
                if result is not None:
                    if match_vis_dir is not None:
                        match_vis_dir.mkdir(parents=True, exist_ok=True)
                        prefix = match_vis_prefix or image_path.stem
                        vis_name = (
                            f"{prefix}_ref-{ref_key_job}_tgt-{tgt_key}_s1-matchanything.jpg"
                        )
                        vis_path = match_vis_dir / vis_name
                        vis_ref = ref_gray
                        vis_tgt = tgt_gray
                        vis_ref_pts = result.mkpts_reference
                        vis_tgt_pts = result.mkpts_target
                        vis_inliers = result.inliers
                        vis_conf = result.confidence

                        if (
                            match_vis_use_raw
                            and ref_key_job not in refined_keys
                            and tgt_key not in refined_keys
                        ):
                            raw_ref = get_raw_gray(ref_key_job)
                            raw_tgt = get_raw_gray(tgt_key)
                            ref_shift = get_shift(ref_key_job)
                            tgt_shift = get_shift(tgt_key)
                            ref_pts_raw, ref_ok = _map_preserved_to_raw(
                                vis_ref_pts, ref_shift, raw_ref.shape[:2]
                            )
                            tgt_pts_raw, tgt_ok = _map_preserved_to_raw(
                                vis_tgt_pts, tgt_shift, raw_tgt.shape[:2]
                            )
                            keep = ref_ok & tgt_ok
                            if np.any(keep):
                                vis_ref = raw_ref
                                vis_tgt = raw_tgt
                                vis_ref_pts = ref_pts_raw[keep]
                                vis_tgt_pts = tgt_pts_raw[keep]
                                vis_inliers = vis_inliers[keep]
                                vis_conf = vis_conf[keep]
                            else:
                                LOGGER.debug(
                                    "Match vis raw mapping dropped all points (ref=%s, tgt=%s).",
                                    ref_key_job,
                                    tgt_key,
                                )
                        vis_img = draw_match_vis(
                            reference=vis_ref,
                            target=vis_tgt,
                            ref_pts=vis_ref_pts,
                            tgt_pts=vis_tgt_pts,
                            inliers=vis_inliers,
                            confidences=vis_conf,
                            max_matches=match_vis_max_matches,
                            layout=match_vis_layout,
                            max_dim=match_vis_max_dim,
                            inliers_only=match_vis_inliers_only,
                            title="Matches",
                        )
                        cv2.imwrite(
                            str(vis_path),
                            vis_img,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 90],
                        )
                    apply_refine_current(result.matrix)
                else:
                    LOGGER.debug(
                        "Stage1 MatchAnything failed for %s (ref=%s)", tgt_key, ref_key_job
                    )
            elif stage1_method != "none":
                raise ValueError(f"Unknown stage1_method: {stage1_method}")

            # Stage 2: fine
            if stage2_method == "ecc":
                ref_gray_aligned = get_gray_aligned(ref_key_job, rgb_band_hint=rgb_band_hint)
                ref_mask_aligned = get_mask_aligned(ref_key_job).astype(np.uint8) * 255
                ref_crop_off = get_crop_offset(ref_key_job)

                tgt_gray_aligned = get_gray_aligned(tgt_key)
                tgt_mask_aligned = get_mask_aligned(tgt_key).astype(np.uint8) * 255
                tgt_crop_off = get_crop_offset(tgt_key)

                m2_aligned = estimate_ecc_transform(
                    ref_gray_aligned,
                    tgt_gray_aligned,
                    motion=ecc_motion,
                    reference_mask=ref_mask_aligned,
                    target_mask=tgt_mask_aligned,
                    iters=ecc_iters,
                    eps=ecc_eps,
                )
                if m2_aligned is not None:
                    m2 = _lift_aligned_warp_to_preserved(
                        m2_aligned,
                        ref_crop_offset=ref_crop_off,
                        tgt_crop_offset=tgt_crop_off,
                    )
                    apply_refine_current(m2)
                else:
                    LOGGER.debug("Stage2 ECC failed for %s (ref=%s)", tgt_key, ref_key_job)
            elif stage2_method != "none":
                raise ValueError(f"Unknown stage2_method: {stage2_method}")

    spectral_stack = np.stack(
        [aligned_bands_raw[band_nm] for band_nm in SPECTRAL_CHANNELS], axis=2
    )

    spectral_valid_mask: Optional[np.ndarray] = None
    for band_nm in SPECTRAL_CHANNELS:
        mask = aligned_masks[band_nm]
        spectral_valid_mask = (
            mask.astype(bool)
            if spectral_valid_mask is None
            else spectral_valid_mask & mask.astype(bool)
        )
    if spectral_valid_mask is None:
        spectral_valid_mask = np.ones_like(rgb_mask_aligned, dtype=bool)

    full_valid_mask = spectral_valid_mask & rgb_mask_aligned

    if crop_black_edges:
        crop_rect = largest_true_rectangle(full_valid_mask)
        if crop_rect is None:
            raise RuntimeError("Unable to find zero-free rectangle for mask.")
        rgb_aligned, spectral_stack, full_valid_mask = crop_stack_to_mask(
            rgb_aligned, spectral_stack, full_valid_mask
        )
        spectral_valid_mask = full_valid_mask.copy()
    else:
        crop_rect = None

    nir_stack = spectral_stack.copy()
    bgr_aligned = rgb_aligned[..., ::-1]
    if bgr_aligned.dtype != spectral_stack.dtype:
        # When saving to uint16 stacks, plain astype(uint16) keeps values in 0..255 which
        # many viewers render as near-black. Expand 8-bit to 16-bit by bit replication:
        # v16 = (v8 << 8) | v8 == v8 * 257, so 255 maps to 65535 (full range).
        if bgr_aligned.dtype == np.uint8 and spectral_stack.dtype == np.uint16:
            bgr_u16 = bgr_aligned.astype(np.uint16, copy=False)
            bgr_aligned = (bgr_u16 << 8) | bgr_u16
        else:
            bgr_aligned = bgr_aligned.astype(spectral_stack.dtype, copy=False)
    full_stack = np.concatenate([bgr_aligned, spectral_stack], axis=2)

    if pseudo_vis_dir is not None and pseudo_pre_bands is not None:
        pseudo_vis_dir.mkdir(parents=True, exist_ok=True)
        pre_bgr = _make_pseudo_bgr(
            pseudo_pre_bands[PSEUDO_BGR_WAVELENGTHS[0]],
            pseudo_pre_bands[PSEUDO_BGR_WAVELENGTHS[1]],
            pseudo_pre_bands[PSEUDO_BGR_WAVELENGTHS[2]],
            clip_percentile=clip_percentile,
        )
        post_bgr = _make_pseudo_bgr(
            aligned_bands_raw[PSEUDO_BGR_WAVELENGTHS[0]],
            aligned_bands_raw[PSEUDO_BGR_WAVELENGTHS[1]],
            aligned_bands_raw[PSEUDO_BGR_WAVELENGTHS[2]],
            clip_percentile=clip_percentile,
        )
        if crop_rect is not None:
            pre_bgr = _crop_to_rect(pre_bgr, crop_rect)
            post_bgr = _crop_to_rect(post_bgr, crop_rect)
        pseudo_vis = _draw_pseudo_vis(
            pre_bgr,
            post_bgr,
            layout=pseudo_vis_layout,
            max_dim=pseudo_vis_max_dim,
        )
        prefix = match_vis_prefix or image_path.stem
        pseudo_name = f"{prefix}_pseudo_bgr.jpg"
        cv2.imwrite(
            str(pseudo_vis_dir / pseudo_name),
            pseudo_vis,
            [int(cv2.IMWRITE_JPEG_QUALITY), 90],
        )

    channel_labels = (
        "B",
        "G",
        "R",
        *[f"{band_nm}nm" for band_nm in SPECTRAL_CHANNELS],
    )
    return full_stack, nir_stack, channel_labels


def align_dataset(
    images_dir: Path,
    spectral_dir: Path,
    output_full: Path,
    output_nir: Path,
    full_format: str,
    nir_format: str,
    alignment_reference: object,
    secondary_topology: str,
    chain_anchor_bands: Tuple[int, ...],
    chain_visible_rgb_ref: str,
    offsets_before: Optional[Dict[object, Tuple[int, int]]],
    offsets_after: Optional[Dict[object, Tuple[int, int]]],
    secondary_alignment: bool,
    stage1_method: str,
    stage2_method: str,
    matchanything: Optional[MatchAnythingSettings],
    ecc_motion: str,
    ecc_iters: int,
    ecc_eps: float,
    crop_black_edges: bool,
    clip_percentile: float,
    match_vis_dir: Optional[Path],
    match_vis_max_matches: int,
    match_vis_layout: str,
    match_vis_max_dim: int,
    match_vis_inliers_only: bool,
    match_vis_use_raw: bool,
    pseudo_vis_dir: Optional[Path],
    pseudo_vis_layout: str,
    pseudo_vis_max_dim: int,
    overwrite: bool,
    show_progress: bool,
) -> None:
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing images directory: {images_dir}")
    if not spectral_dir.is_dir():
        raise FileNotFoundError(f"Missing spectral directory: {spectral_dir}")

    output_full.mkdir(parents=True, exist_ok=True)
    output_nir.mkdir(parents=True, exist_ok=True)
    if match_vis_dir is not None:
        match_vis_dir.mkdir(parents=True, exist_ok=True)
    if pseudo_vis_dir is not None:
        pseudo_vis_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        [
            p
            for p in images_dir.glob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
    )

    if not image_files:
        LOGGER.warning("No RGB images found in %s", images_dir)
        return

    iterator = image_files
    if tqdm is not None and show_progress:
        iterator = tqdm(image_files, total=len(image_files), desc="Aligning", unit="img")

    identity_offsets: Dict[object, Tuple[int, int]] = {
        key: (0, 0) for key in CHANNEL_ORDER
    }

    for image_path in iterator:
        stem = image_path.stem
        spectral_path = spectral_dir / f"{stem}.hdr"
        if not spectral_path.exists():
            LOGGER.warning("Skipping %s (missing spectral cube)", image_path.name)
            continue

        try:
            offsets = identity_offsets
            if offsets_before is not None and offsets_after is not None:
                offsets = select_offsets(
                    image_path.name, offsets_before, offsets_after
                )
            full_stack, nir_stack, channel_labels = process_sample(
                image_path=image_path,
                spectral_path=spectral_path,
                offsets=offsets,
                alignment_reference=alignment_reference,
                secondary_topology=secondary_topology,
                chain_anchor_bands=chain_anchor_bands,
                chain_visible_rgb_ref=chain_visible_rgb_ref,
                clip_percentile=clip_percentile,
                enable_secondary_alignment=secondary_alignment,
                stage1_method=stage1_method,
                stage2_method=stage2_method,
                matchanything=matchanything,
                ecc_motion=ecc_motion,
                ecc_iters=ecc_iters,
                ecc_eps=ecc_eps,
                crop_black_edges=crop_black_edges,
                match_vis_dir=match_vis_dir,
                match_vis_max_matches=match_vis_max_matches,
                match_vis_layout=match_vis_layout,
                match_vis_max_dim=match_vis_max_dim,
                match_vis_inliers_only=match_vis_inliers_only,
                match_vis_use_raw=match_vis_use_raw,
                match_vis_prefix=stem,
                pseudo_vis_dir=pseudo_vis_dir,
                pseudo_vis_layout=pseudo_vis_layout,
                pseudo_vis_max_dim=pseudo_vis_max_dim,
            )
        except Exception as exc:
            LOGGER.exception("Failed to process %s: %s", image_path.name, exc)
            continue

        full_extension = ".hdr" if full_format == "hdr" else ".tif"
        full_output = output_full / f"{stem}{full_extension}"

        if not overwrite and full_output.exists():
            LOGGER.info("Skipping existing %s", full_output.name)
        else:
            if full_format == "hdr":
                save_hdr_img(
                    full_stack,
                    str(full_output),
                    [*RGB_PLACEHOLDER_BANDS, *SPECTRAL_CHANNELS],
                    band_names=list(channel_labels),
                )
            else:
                planar_full = np.ascontiguousarray(full_stack.transpose(2, 0, 1))
                tifffile.imwrite(
                    str(full_output),
                    planar_full,
                    dtype=planar_full.dtype,
                    photometric="MINISBLACK",
                    planarconfig="SEPARATE",
                    metadata={
                        "axes": "SYX",
                        "ChannelNames": list(channel_labels),
                    },
                )
            LOGGER.info("Saved %s", full_output)

        nir_extension = ".hdr" if nir_format == "hdr" else ".tif"
        nir_output = output_nir / f"{stem}{nir_extension}"

        if not overwrite and nir_output.exists():
            LOGGER.info("Skipping existing %s", nir_output.name)
        else:
            if nir_format == "hdr":
                save_hdr_img(
                    nir_stack,
                    str(nir_output),
                    list(SPECTRAL_CHANNELS),
                    band_names=[f"{nm}nm" for nm in SPECTRAL_CHANNELS],
                )
            else:
                planar_nir = np.ascontiguousarray(nir_stack.transpose(2, 0, 1))
                tifffile.imwrite(
                    str(nir_output),
                    planar_nir,
                    dtype=planar_nir.dtype,
                    photometric="MINISBLACK",
                    planarconfig="SEPARATE",
                    metadata={
                        "axes": "SYX",
                        "ChannelNames": [f"{nm}nm" for nm in SPECTRAL_CHANNELS],
                    },
                )
            LOGGER.info("Saved %s", nir_output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-align RGB and spectral HDR data into TIF stacks (preserve original spectral dtype/bit depth)."
    )
    parser.add_argument(
        "--rgb-dir",
        type=Path,
        required=True,
        help="Directory containing RGB images (.jpg/.jpeg/.png).",
    )
    parser.add_argument(
        "--spectral-dir",
        type=Path,
        required=True,
        help="Directory containing spectral cubes (.hdr).",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=PROJECT_ROOT / "data/config/oil/my-conf",
        help="Directory containing alignment config files "
        f"'{BEFORE_CONFIG_NAME}' and '{AFTER_CONFIG_NAME}'.",
    )
    parser.add_argument(
        "--no-config-translation",
        action="store_true",
        help="Disable coarse config-based translation alignment (useful for non-oil datasets).",
    )
    parser.add_argument(
        "--output-full",
        type=Path,
        default=None,
        help="Output folder for RGB + 7-band stacks "
        "(default: sibling parent of --rgb-dir/--spectral-dir / 'aligned_full_tif').",
    )
    parser.add_argument(
        "--output-nir",
        type=Path,
        default=None,
        help="Output folder for 7-band spectral stacks "
        "(default: sibling parent of --rgb-dir/--spectral-dir / 'aligned_nir_tif').",
    )
    parser.add_argument(
        "--full-format",
        choices=["tif", "hdr"],
        default="hdr",
        help="File format for full (RGB+spectral) outputs.",
    )
    parser.add_argument(
        "--nir-format",
        choices=["tif", "hdr"],
        default="tif",
        help="File format for spectral-only outputs.",
    )
    parser.add_argument(
        "--alignment-reference",
        type=parse_alignment_reference,
        default='jpg',
        help="Channel used as the zero-offset alignment reference ('jpg' or wavelength).",
    )
    parser.add_argument(
        "--enable-secondary-align",
        action="store_true",
        help="Enable secondary alignment stages after config translation.",
    )
    parser.add_argument(
        "--stage1-method",
        choices=["none", "orb", "matchanything"],
        default="matchanything",
        help="Secondary stage1 method (coarse): none/orb/matchanything.",
    )
    parser.add_argument(
        "--stage2-method",
        choices=["none", "ecc"],
        default="none",
        help="Secondary stage2 method (fine): none/ecc.",
    )
    parser.add_argument(
        "--secondary-topology",
        choices=["single", "chain"],
        default="single",
        help=(
            "How to choose the reference for secondary alignment. "
            "'single' aligns every non-reference channel directly to --alignment-reference. "
            "'chain' anchors visible bands (default: 450/550/650) to JPG and then aligns each "
            "subsequent wavelength to the previous band (e.g. 720->650, 750->720, ...)."
        ),
    )
    parser.add_argument(
        "--chain-anchor-bands",
        type=parse_int_list_csv,
        default=DEFAULT_CHAIN_ANCHOR_BANDS,
        help=(
            "Comma-separated wavelengths that align directly to JPG when "
            "--secondary-topology chain (default: 450,550,650)."
        ),
    )
    parser.add_argument(
        "--chain-visible-rgb-ref",
        choices=["gray", "channels"],
        default="gray",
        help=(
            "When --secondary-topology chain uses JPG as reference for a visible band, "
            "use JPG grayscale ('gray') or the corresponding RGB channel ('channels': "
            "450->B, 550->G, 650->R)."
        ),
    )
    parser.add_argument(
        "--matchanything-repo",
        type=Path,
        default=PROJECT_ROOT / "third_party/MatchAnything/imcui/third_party/MatchAnything",
        help="!!! MatchAnything repo root (contains configs/, src/, weights/).",
    )
    parser.add_argument(
        "--matchanything-model",
        choices=["matchanything_eloftr", "matchanything_roma"],
        default="matchanything_roma",
        help="Which MatchAnything backbone to use.",
    )
    parser.add_argument(
        "--matchanything-estimator",
        choices=["similarity", "affine"],
        default="similarity",
        help="RANSAC model used to fit transform from matches (similarity=estimateAffinePartial2D, affine=estimateAffine2D).",
    )
    parser.add_argument(
        "--matchanything-config",
        type=Path,
        default=None,
        help="Override MatchAnything config path (defaults to configs/models/{eloftr,roma}_model.py).",
    )
    parser.add_argument(
        "--matchanything-ckpt",
        type=Path,
        default=None,
        help="Override MatchAnything checkpoint path (defaults to weights/matchanything_{eloftr,roma}.ckpt).",
    )
    parser.add_argument(
        "--matchanything-device",
        type=str,
        default="cuda",
        help="Torch device for MatchAnything (e.g. cuda, cuda:0, cpu).",
    )
    parser.add_argument(
        "--matchanything-imgresize",
        type=int,
        default=832,
        help="Resize longer edge before matching (<=0 disables).",
    )
    parser.add_argument(
        "--matchanything-min-confidence",
        type=float,
        default=0.1,
        help="Minimum match confidence to keep.",
    )
    parser.add_argument(
        "--matchanything-max-matches",
        type=int,
        default=5000,
        help="Max matches used for RANSAC (<=0 keeps all).",
    )
    parser.add_argument(
        "--matchanything-ransac-thr",
        type=float,
        default=3.0,
        help="RANSAC reprojection threshold (pixels).",
    )
    parser.add_argument(
        "--matchanything-min-inliers",
        type=int,
        default=8,
        help="Minimum inlier count to accept affine.",
    )
    parser.add_argument(
        "--matchanything-min-scale",
        type=float,
        default=0.7,
        help="Reject affine if scale is below this.",
    )
    parser.add_argument(
        "--matchanything-max-scale",
        type=float,
        default=1.3,
        help="Reject affine if scale is above this.",
    )
    parser.add_argument(
        "--ecc-motion",
        choices=["translation", "affine"],
        default="translation",
        help="ECC motion model for stage2 (when --stage2-method ecc).",
    )
    parser.add_argument(
        "--ecc-iters",
        type=int,
        default=200,
        help="ECC max iterations (when --stage2-method ecc).",
    )
    parser.add_argument(
        "--ecc-eps",
        type=float,
        default=1e-6,
        help="ECC convergence epsilon (when --stage2-method ecc).",
    )
    parser.add_argument(
        "--crop-black-edges",
        action="store_true",
        help="Trim zero-fill borders introduced during alignment.",
    )
    parser.add_argument(
        "--clip-percentile",
        type=float,
        default=1.0,
        help="Percentile for low/high clipping during uint8 normalization.",
    )
    parser.add_argument(
        "--save-match-vis",
        action="store_true",
        help="Save MatchAnything feature-match visualizations (reference vs target) with match lines.",
    )
    parser.add_argument(
        "--match-vis-dir",
        type=Path,
        default=None,
        help="Output directory for match visualizations (default: <output_full>/match_vis).",
    )
    parser.add_argument(
        "--match-vis-max-matches",
        type=int,
        default=200,
        help="Maximum number of matches to draw per visualization image.",
    )
    parser.add_argument(
        "--match-vis-layout",
        choices=["lr", "tb"],
        default="lr",
        help="Visualization layout: lr=left-right, tb=top-bottom.",
    )
    parser.add_argument(
        "--match-vis-max-dim",
        type=int,
        default=1600,
        help="Resize the longest edge for visualization (<=0 disables resizing).",
    )
    parser.add_argument(
        "--match-vis-inliers-only",
        action="store_true",
        help="Only draw RANSAC inlier matches (helps reduce clutter).",
    )
    parser.add_argument(
        "--match-vis-use-raw",
        action="store_true",
        help=(
            "Use pre-alignment raw images as the visualization base (maps matches back to raw "
            "coords before config translation). Only accurate before a band has been refined."
        ),
    )
    parser.add_argument(
        "--save-pseudo-vis",
        action="store_true",
        help="Save pseudo-color comparisons (450/550/650nm as BGR) before/after alignment.",
    )
    parser.add_argument(
        "--pseudo-vis-dir",
        type=Path,
        default=None,
        help="Output directory for pseudo-color visualizations (default: <output_full>/pseudo_vis).",
    )
    parser.add_argument(
        "--pseudo-vis-layout",
        choices=["lr", "tb"],
        default=None,
        help="Pseudo-color layout (default: use --match-vis-layout).",
    )
    parser.add_argument(
        "--pseudo-vis-max-dim",
        type=int,
        default=None,
        help="Resize the longest edge for pseudo-color visualization (default: --match-vis-max-dim).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs if present.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar (tqdm).",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="Output directory suffix (default: auto, e.g. 20251229-1157).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    images_dir = args.rgb_dir
    spectral_dir = args.spectral_dir

    if images_dir.parent == spectral_dir.parent:
        default_root = images_dir.parent
    else:
        default_root = Path(
            os.path.commonpath([images_dir.resolve(), spectral_dir.resolve()])
        )
        if default_root == Path(default_root.root):
            default_root = Path.cwd()

    # Default two-stage behavior (config translation + MatchAnything refine):
    # - If user passes --enable-secondary-align but explicitly disables both stage methods,
    #   fall back to MatchAnything for stage1.
    stage1_method = args.stage1_method
    stage2_method = args.stage2_method
    if (
        args.enable_secondary_align
        and stage1_method == "none"
        and stage2_method == "none"
    ):
        stage1_method = "matchanything"

    enable_secondary = args.enable_secondary_align or stage1_method != "none" or stage2_method != "none"

    if (
        enable_secondary
        and args.secondary_topology == "chain"
        and args.alignment_reference != "jpg"
    ):
        raise ValueError(
            "--secondary-topology chain requires --alignment-reference jpg "
            "to keep outputs anchored to RGB/labels."
        )

    offsets_before: Optional[Dict[object, Tuple[int, int]]] = None
    offsets_after: Optional[Dict[object, Tuple[int, int]]] = None
    if not args.no_config_translation:
        config_dir = args.config_dir
        offsets_before = load_alignment_config(
            config_dir / BEFORE_CONFIG_NAME, args.alignment_reference
        )
        offsets_after = load_alignment_config(
            config_dir / AFTER_CONFIG_NAME, args.alignment_reference
        )

    matchanything: Optional[MatchAnythingSettings] = None
    if stage1_method == "matchanything":
        repo_dir = args.matchanything_repo
        method = args.matchanything_model
        config_path = args.matchanything_config
        if config_path is None:
            config_path = repo_dir / (
                "configs/models/eloftr_model.py"
                if method == "matchanything_eloftr"
                else "configs/models/roma_model.py"
            )
        ckpt_path = args.matchanything_ckpt
        if ckpt_path is None:
            ckpt_path = repo_dir / (
                "weights/matchanything_eloftr.ckpt"
                if method == "matchanything_eloftr"
                else "weights/matchanything_roma.ckpt"
            )

        if not repo_dir.is_dir():
            raise FileNotFoundError(f"MatchAnything repo not found: {repo_dir}")
        if not config_path.is_file():
            raise FileNotFoundError(f"MatchAnything config not found: {config_path}")
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"MatchAnything ckpt not found: {ckpt_path}")

        matchanything = MatchAnythingSettings(
            repo_dir=repo_dir,
            config_path=config_path,
            ckpt_path=ckpt_path,
            method=str(method),
            estimator=str(args.matchanything_estimator),
            device=str(args.matchanything_device),
            imgresize=int(args.matchanything_imgresize),
            min_confidence=float(args.matchanything_min_confidence),
            max_matches=int(args.matchanything_max_matches),
            ransac_reproj_threshold=float(args.matchanything_ransac_thr),
            min_inliers=int(args.matchanything_min_inliers),
            min_scale=float(args.matchanything_min_scale),
            max_scale=float(args.matchanything_max_scale),
        )

    run_tag = args.run_tag or datetime.now().strftime("%Y%m%d-%H%M")
    scheme_tag = f"s1-{stage1_method}_s2-{stage2_method}"
    if args.secondary_topology != "single":
        scheme_tag = f"{scheme_tag}_topo-{args.secondary_topology}"

    output_full_base = args.output_full or (default_root / "aligned_full_tif")
    output_nir_base = args.output_nir or (default_root / "aligned_nir_tif")

    # If user didn't explicitly name output dirs, append the alignment scheme after the date tag.
    full_tag = f"{run_tag}_{scheme_tag}" if args.output_full is None else run_tag
    nir_tag = f"{run_tag}_{scheme_tag}" if args.output_nir is None else run_tag
    output_full = output_full_base.parent / f"{output_full_base.name}_{full_tag}"
    output_nir = output_nir_base.parent / f"{output_nir_base.name}_{nir_tag}"

    match_vis_dir: Optional[Path] = None
    if args.save_match_vis:
        match_vis_dir = args.match_vis_dir or (output_full / "match_vis")
    pseudo_vis_dir: Optional[Path] = None
    if args.save_pseudo_vis:
        pseudo_vis_dir = args.pseudo_vis_dir or (output_full / "pseudo_vis")
    pseudo_vis_layout = args.pseudo_vis_layout or args.match_vis_layout
    pseudo_vis_max_dim = (
        int(args.pseudo_vis_max_dim)
        if args.pseudo_vis_max_dim is not None
        else int(args.match_vis_max_dim)
    )

    align_dataset(
        images_dir=images_dir,
        spectral_dir=spectral_dir,
        output_full=output_full,
        output_nir=output_nir,
        full_format=args.full_format,
        nir_format=args.nir_format,
        alignment_reference=args.alignment_reference,
        secondary_topology=args.secondary_topology,
        chain_anchor_bands=tuple(args.chain_anchor_bands),
        chain_visible_rgb_ref=args.chain_visible_rgb_ref,
        offsets_before=offsets_before,
        offsets_after=offsets_after,
        secondary_alignment=enable_secondary,
        stage1_method=stage1_method,
        stage2_method=stage2_method,
        matchanything=matchanything,
        ecc_motion=args.ecc_motion,
        ecc_iters=args.ecc_iters,
        ecc_eps=args.ecc_eps,
        crop_black_edges=args.crop_black_edges,
        clip_percentile=args.clip_percentile,
        match_vis_dir=match_vis_dir,
        match_vis_max_matches=int(args.match_vis_max_matches),
        match_vis_layout=str(args.match_vis_layout),
        match_vis_max_dim=int(args.match_vis_max_dim),
        match_vis_inliers_only=bool(args.match_vis_inliers_only),
        match_vis_use_raw=bool(args.match_vis_use_raw),
        pseudo_vis_dir=pseudo_vis_dir,
        pseudo_vis_layout=str(pseudo_vis_layout),
        pseudo_vis_max_dim=int(pseudo_vis_max_dim),
        overwrite=args.overwrite,
        show_progress=not args.no_progress,
    )


if __name__ == "__main__":
    main()
# 最优对齐配置
# python data_align/data_align.py --rgb-dir /mnt/d/Project/master-graduation-project/data/oil/train/feedback/images --spectral-dir /mnt/d/Project/master-graduation-project/data/oil/train/feedback/spectral --alignment-reference 720 --enable-secondary-align
#  --overwrite
