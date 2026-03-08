
import argparse
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys

import cv2
import numpy as np
import tifffile

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.io.spectral_io import (
    open_hdr_img,
    save_hdr_img,
    channel as SPECTRAL_CHANNELS,
)

LOGGER = logging.getLogger(__name__)

DATE_PATTERN = re.compile(r"_(\d{8})_")
DATE_CUTOFF = datetime.strptime("20240513", "%Y%m%d")
BEFORE_CONFIG_NAME = "align_before_513.conf"
AFTER_CONFIG_NAME = "align_after_513.conf"

CHANNEL_ORDER: Tuple[object, ...] = ("jpg", *SPECTRAL_CHANNELS)
RGB_PLACEHOLDER_BANDS: Tuple[int, int, int] = (0, 1, 2)


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
        # 这里做容错，允许直接读取旧文件继续使用。
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

    parsed_values = list(parsed_values)
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
    clip_percentile: float,
    enable_secondary_alignment: bool,
    secondary_align_method: str,
    ecc_motion: str,
    ecc_iters: int,
    ecc_eps: float,
    crop_black_edges: bool,
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
    spectral_crop_offsets: Dict[int, Tuple[int, int]] = {}
    aligned_bands_raw: Dict[int, np.ndarray] = {}
    aligned_bands_u8: Dict[int, np.ndarray] = {}
    spectral_masks_preserved: Dict[int, np.ndarray] = {}
    spectral_mask_offsets: Dict[int, Tuple[int, int]] = {}
    aligned_masks: Dict[int, np.ndarray] = {}

    for idx, band_nm in enumerate(SPECTRAL_CHANNELS):
        band = hdr_cube[:, :, idx]
        preserved_raw, crop_offset = apply_pixel_offset(
            band, offsets[band_nm], fill_value=0, preserve_full=True
        )
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

    reference_key = alignment_reference
    if reference_key == "jpg":
        reference_preserved = rgb_preserved
        reference_crop_offset = rgb_crop_offset
        reference_gray = rgb_gray_preserved
    else:
        reference_preserved = spectral_preserved_raw[int(reference_key)]
        reference_crop_offset = spectral_crop_offsets[int(reference_key)]
        reference_gray = spectral_preserved_u8[int(reference_key)]

    if enable_secondary_alignment:
        ref_height, ref_width = reference_preserved.shape[:2]
        ref_gray = reference_gray

        ref_mask_preserved = (
            rgb_mask_preserved
            if reference_key == "jpg"
            else spectral_masks_preserved[int(reference_key)]
        )

        feature_targets = []
        if reference_key != "jpg":
            feature_targets.append(
                ("jpg", rgb_preserved, rgb_gray_preserved, rgb_mask_preserved)
            )
        for band_nm, preserved_raw in spectral_preserved_raw.items():
            if band_nm == reference_key:
                continue
            feature_targets.append(
                (
                    band_nm,
                    preserved_raw,
                    spectral_preserved_u8[band_nm],
                    spectral_masks_preserved[band_nm],
                )
            )

        for key, preserved_image, feature_gray, mask_preserved in feature_targets:
            try:
                target_gray = (
                    feature_gray
                    if feature_gray.ndim == 2
                    else to_grayscale(feature_gray)
                )
            except ValueError:
                LOGGER.warning(
                    "Skipping secondary alignment for %s due to unsupported shape",
                    key,
                )
                continue

            matrix: Optional[np.ndarray] = None
            if secondary_align_method == "orb":
                matrix = estimate_affine_transform(ref_gray, target_gray)
            elif secondary_align_method == "ecc":
                # ECC is much more sensitive to padding/mismatched coordinate frames than ORB.
                # Estimate ECC on the already config-aligned (cropped) images, then lift
                # the warp back to preserved coordinates for application.
                if reference_key == "jpg":
                    ref_gray_aligned = to_grayscale(rgb_aligned)
                    ref_mask_aligned = rgb_mask_aligned.astype(np.uint8) * 255
                    ref_crop_off = rgb_crop_offset
                else:
                    ref_gray_aligned = aligned_bands_u8[int(reference_key)]
                    ref_mask_aligned = aligned_masks[int(reference_key)].astype(np.uint8) * 255
                    ref_crop_off = spectral_crop_offsets[int(reference_key)]

                if key == "jpg":
                    tgt_gray_aligned = to_grayscale(rgb_aligned)
                    tgt_mask_aligned = rgb_mask_aligned.astype(np.uint8) * 255
                    tgt_crop_off = rgb_crop_offset
                else:
                    tgt_gray_aligned = aligned_bands_u8[int(key)]
                    tgt_mask_aligned = aligned_masks[int(key)].astype(np.uint8) * 255
                    tgt_crop_off = spectral_crop_offsets[int(key)]

                matrix_aligned = estimate_ecc_transform(
                    ref_gray_aligned,
                    tgt_gray_aligned,
                    motion=ecc_motion,
                    reference_mask=ref_mask_aligned,
                    target_mask=tgt_mask_aligned,
                    iters=ecc_iters,
                    eps=ecc_eps,
                )
                if matrix_aligned is not None:
                    matrix = _lift_aligned_warp_to_preserved(
                        matrix_aligned,
                        ref_crop_offset=ref_crop_off,
                        tgt_crop_offset=tgt_crop_off,
                    )
            else:
                raise ValueError(f"Unknown secondary_align_method: {secondary_align_method}")

            if matrix is None:
                LOGGER.debug(
                    "Secondary %s alignment failed for %s",
                    secondary_align_method,
                    key,
                )
                continue

            refined_preserved = warp_affine(
                preserved_image,
                matrix,
                (ref_height, ref_width),
                fill_value=0,
            )
            refined_mask = warp_affine(
                mask_preserved,
                matrix,
                (ref_height, ref_width),
                fill_value=0,
                interpolation=cv2.INTER_NEAREST,
            )
            refined_mask = (refined_mask > 0).astype(np.uint8)

            if key == "jpg":
                rgb_preserved = refined_preserved
                rgb_aligned = crop_to_shape(
                    refined_preserved, reference_crop_offset, rgb_height, rgb_width
                )
                rgb_gray_preserved = to_grayscale(refined_preserved)
                rgb_crop_offset = reference_crop_offset
                rgb_mask_preserved = refined_mask
                rgb_mask_aligned = crop_to_shape(
                    refined_mask, reference_crop_offset, rgb_height, rgb_width
                ).astype(bool)
            else:
                refined_u8 = (
                    refined_preserved
                    if refined_preserved.dtype == np.uint8
                    else normalize_to_uint8(refined_preserved, clip_percentile)
                )
                spectral_preserved_raw[int(key)] = refined_preserved
                spectral_preserved_u8[int(key)] = refined_u8
                spectral_masks_preserved[int(key)] = refined_mask
                spectral_crop_offsets[int(key)] = reference_crop_offset
                aligned_bands_raw[int(key)] = crop_to_shape(
                    refined_preserved,
                    spectral_crop_offsets[int(key)],
                    spectral_height,
                    spectral_width,
                )
                aligned_bands_u8[int(key)] = crop_to_shape(
                    refined_u8,
                    spectral_crop_offsets[int(key)],
                    spectral_height,
                    spectral_width,
                )
                aligned_masks[int(key)] = crop_to_shape(
                    refined_mask,
                    spectral_crop_offsets[int(key)],
                    spectral_height,
                    spectral_width,
                ).astype(bool)

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
        rgb_aligned, spectral_stack, full_valid_mask = crop_stack_to_mask(
            rgb_aligned, spectral_stack, full_valid_mask
        )
        spectral_valid_mask = full_valid_mask.copy()

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

    channel_labels = (
        "B",
        "G",
        "R",
        *[f"{band_nm}nm" for band_nm in SPECTRAL_CHANNELS],
    )
    return full_stack, nir_stack, channel_labels


def align_dataset(
    dataset_root: Path,
    config_dir: Path,
    output_full: Path,
    output_nir: Path,
    full_format: str,
    nir_format: str,
    alignment_reference: object,
    secondary_alignment: bool,
    secondary_align_method: str,
    ecc_motion: str,
    ecc_iters: int,
    ecc_eps: float,
    crop_black_edges: bool,
    clip_percentile: float,
    overwrite: bool,
) -> None:
    images_dir = dataset_root / "images"
    spectral_dir = dataset_root / "spectral"

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing images directory: {images_dir}")
    if not spectral_dir.is_dir():
        raise FileNotFoundError(f"Missing spectral directory: {spectral_dir}")

    if not config_dir.is_dir():
        raise FileNotFoundError(f"Missing config directory: {config_dir}")

    before_offsets = load_alignment_config(
        config_dir / BEFORE_CONFIG_NAME, alignment_reference
    )
    after_offsets = load_alignment_config(
        config_dir / AFTER_CONFIG_NAME, alignment_reference
    )

    output_full.mkdir(parents=True, exist_ok=True)
    output_nir.mkdir(parents=True, exist_ok=True)

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

    for image_path in image_files:
        stem = image_path.stem
        spectral_path = spectral_dir / f"{stem}.hdr"
        if not spectral_path.exists():
            LOGGER.warning("Skipping %s (missing spectral cube)", image_path.name)
            continue

        offsets = select_offsets(image_path.name, before_offsets, after_offsets)

        try:
            full_stack, nir_stack, channel_labels = process_sample(
                image_path,
                spectral_path,
                offsets,
                alignment_reference,
                clip_percentile,
                secondary_alignment,
                secondary_align_method,
                ecc_motion,
                ecc_iters,
                ecc_eps,
                crop_black_edges,
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
        "--dataset_root",
        type=Path,
        default=Path("/mnt/d/Project/master-graduation-project/master-graduation/data/oil/train/feedback"),
        help="Folder containing 'images' and 'spectral' subdirectories.",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("./data/config/oil/my-conf"),
        help="Directory containing alignment config files "
        "'align_before_513.conf' and 'align_after_513.conf'.",
    )
    parser.add_argument(
        "--output-full",
        type=Path,
        default=Path("/mnt/d/Project/master-graduation-project/master-graduation/data/oil/train/feedback/aligned_full_tif"),
        help="Output folder for RGB + 7-band stacks "
        "(default: dataset_root / 'aligned_full_tif').",
    )
    parser.add_argument(
        "--output-nir",
        type=Path,
        default=Path("/mnt/d/Project/master-graduation-project/master-graduation/data/oil/train/feedback/aligned_nir_tif"),
        help="Output folder for 7-band spectral stacks "
        "(default: dataset_root / 'aligned_nir_tif').",
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
        default=720,
        help="Channel used as the zero-offset alignment reference ('jpg' or wavelength).",
    )
    parser.add_argument(
        "--enable-secondary-align",
        action="store_true",
        help="Apply secondary alignment after config translation (default method: ORB).",
    )
    parser.add_argument(
        "--secondary-align-method",
        choices=["orb", "ecc"],
        default="orb",
        help="Secondary alignment method: orb or ecc (requires --enable-secondary-align).",
    )
    parser.add_argument(
        "--ecc-motion",
        choices=["translation", "affine"],
        default="translation",
        help="ECC motion model for secondary alignment (when --secondary-align-method ecc).",
    )
    parser.add_argument(
        "--ecc-iters",
        type=int,
        default=200,
        help="ECC max iterations (when --secondary-align-method ecc).",
    )
    parser.add_argument(
        "--ecc-eps",
        type=float,
        default=1e-6,
        help="ECC convergence epsilon (when --secondary-align-method ecc).",
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
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs if present.",
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

    run_tag = args.run_tag or datetime.now().strftime("%Y%m%d-%H%M")
    output_full_base = args.output_full or args.dataset_root / "aligned_full_tif"
    output_nir_base = args.output_nir or args.dataset_root / "aligned_nir_tif"
    output_full = output_full_base.parent / f"{output_full_base.name}_{run_tag}"
    output_nir = output_nir_base.parent / f"{output_nir_base.name}_{run_tag}"

    align_dataset(
        dataset_root=args.dataset_root,
        config_dir=args.config_dir,
        output_full=output_full,
        output_nir=output_nir,
        full_format=args.full_format,
        nir_format=args.nir_format,
        alignment_reference=args.alignment_reference,
        secondary_alignment=args.enable_secondary_align,
        secondary_align_method=args.secondary_align_method,
        ecc_motion=args.ecc_motion,
        ecc_iters=args.ecc_iters,
        ecc_eps=args.ecc_eps,
        crop_black_edges=args.crop_black_edges,
        clip_percentile=args.clip_percentile,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
# 最优对齐配置
# python data_align/data_align.py --dataset_root /mnt/d/Project/master-graduation-project/data/oil/train/feedback --config-dir data/config/oil/my-conf --alignment-reference 720 --enable-secondary-align
#  --overwrite
