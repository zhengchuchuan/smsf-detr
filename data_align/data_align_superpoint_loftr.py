
import argparse
import logging
import re
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys

import cv2
import numpy as np
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F

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


class SuperPointNet(nn.Module):
    """Minimal SuperPoint network (MagicLeap-style) for inference-only keypoints/descriptors."""

    def __init__(self) -> None:
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1a = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        # Detector head.
        self.conv_pa = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv_pb = nn.Conv2d(256, 65, kernel_size=1, stride=1, padding=0)

        # Descriptor head.
        self.conv_da = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv_db = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        cpa = self.relu(self.conv_pa(x))
        semi = self.conv_pb(cpa)

        cda = self.relu(self.conv_da(x))
        desc = self.conv_db(cda)
        desc = F.normalize(desc, p=2.0, dim=1, eps=1e-8)
        return semi, desc


def _load_superpoint_state_dict(weights_path: Path) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(str(weights_path), map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model", "weights"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                checkpoint = value
                break

    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported SuperPoint checkpoint format: {type(checkpoint)}")

    state_dict: Dict[str, torch.Tensor] = {}
    for key, value in checkpoint.items():
        if not isinstance(value, torch.Tensor):
            continue
        name = str(key)
        if name.startswith("module."):
            name = name[len("module.") :]
        if name.startswith("superpoint."):
            name = name[len("superpoint.") :]
        state_dict[name] = value
    if not state_dict:
        raise ValueError(f"No tensor weights found in checkpoint: {weights_path}")
    return state_dict


class SuperPointExtractor:
    def __init__(
        self,
        *,
        weights_path: Path,
        device: torch.device,
        nms_radius: int,
        keypoint_threshold: float,
        max_keypoints: int,
        remove_border: int,
    ) -> None:
        if not weights_path.is_file():
            raise FileNotFoundError(f"Missing SuperPoint weights: {weights_path}")
        self.device = device
        self.nms_radius = int(nms_radius)
        self.keypoint_threshold = float(keypoint_threshold)
        self.max_keypoints = int(max_keypoints)
        self.remove_border = int(remove_border)

        model = SuperPointNet()
        state = _load_superpoint_state_dict(weights_path)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            raise ValueError(
                f"SuperPoint weights missing keys (example: {missing[0]}). "
                f"Expected MagicLeap-style SuperPoint state_dict."
            )
        if unexpected:
            LOGGER.debug("SuperPoint weights have unexpected keys: %s", unexpected[:5])
        self.model = model.to(self.device).eval()

    @torch.no_grad()
    def extract(
        self, image: np.ndarray, *, mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        if image.ndim != 2:
            raise ValueError("SuperPoint expects 2D grayscale uint8 image.")
        if image.dtype != np.uint8:
            raise ValueError("SuperPoint expects uint8 input; call normalize_to_uint8 first.")

        height, width = image.shape[:2]
        pad_h = (-height) % 8
        pad_w = (-width) % 8

        if pad_h or pad_w:
            image_padded = np.pad(
                image, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0
            )
            mask_padded: Optional[np.ndarray]
            if mask is None:
                mask_padded = None
            else:
                if mask.shape[:2] != image.shape[:2]:
                    raise ValueError("Mask shape must match image shape.")
                mask_u8 = (mask > 0).astype(np.uint8)
                mask_padded = np.pad(
                    mask_u8,
                    ((0, pad_h), (0, pad_w)),
                    mode="constant",
                    constant_values=0,
                )
        else:
            image_padded = image
            if mask is None:
                mask_padded = None
            else:
                if mask.shape[:2] != image.shape[:2]:
                    raise ValueError("Mask shape must match image shape.")
                mask_padded = (mask > 0).astype(np.uint8)

        tensor = (
            torch.from_numpy(image_padded.astype(np.float32) / 255.0)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.device)
        )
        semi, desc = self.model(tensor)

        # Dense keypoint heatmap.
        prob = F.softmax(semi, dim=1)[:, :-1, :, :]  # (1,64,Hc,Wc)
        _, _, hc, wc = prob.shape
        prob = prob.permute(0, 2, 3, 1).reshape(1, hc, wc, 8, 8)
        heatmap = prob.permute(0, 1, 3, 2, 4).reshape(1, hc * 8, wc * 8)
        heatmap = heatmap[:, : image_padded.shape[0], : image_padded.shape[1]]

        # NMS.
        nms = int(max(self.nms_radius, 0))
        pooled = F.max_pool2d(heatmap.unsqueeze(1), kernel_size=2 * nms + 1, stride=1, padding=nms)
        keep = (heatmap.unsqueeze(1) == pooled) & (heatmap.unsqueeze(1) > float(self.keypoint_threshold))
        ys, xs = torch.where(keep[0, 0])
        if ys.numel() == 0:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 256), dtype=np.float32)

        scores = heatmap[0, ys, xs]

        # Remove padding-only region + borders.
        in_bounds = (ys < height) & (xs < width)
        border = int(max(self.remove_border, 0))
        if border > 0:
            in_bounds &= (ys >= border) & (ys < height - border) & (xs >= border) & (xs < width - border)

        if mask_padded is not None:
            mask_t = torch.from_numpy(mask_padded.astype(np.uint8)).to(self.device)
            in_bounds &= mask_t[ys, xs] > 0

        ys = ys[in_bounds]
        xs = xs[in_bounds]
        scores = scores[in_bounds]

        if ys.numel() == 0:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 256), dtype=np.float32)

        max_kpts = int(self.max_keypoints)
        if max_kpts > 0 and ys.numel() > max_kpts:
            top_scores, top_idx = torch.topk(scores, k=max_kpts, sorted=False)
            ys = ys[top_idx]
            xs = xs[top_idx]
            scores = top_scores

        # Sort by descending score for stability.
        order = torch.argsort(scores, descending=True)
        ys = ys[order]
        xs = xs[order]

        # Sample descriptors at keypoints on the coarse descriptor grid.
        _, _, hc_desc, wc_desc = desc.shape
        xs_f = xs.float() / 8.0
        ys_f = ys.float() / 8.0
        x_norm = (xs_f / max(wc_desc - 1, 1)) * 2.0 - 1.0
        y_norm = (ys_f / max(hc_desc - 1, 1)) * 2.0 - 1.0
        grid = torch.stack([x_norm, y_norm], dim=1).view(1, -1, 1, 2)
        sampled = F.grid_sample(desc, grid, mode="bilinear", align_corners=True)
        desc_out = sampled.squeeze(0).squeeze(-1).transpose(0, 1)
        desc_out = F.normalize(desc_out, p=2.0, dim=1, eps=1e-8)

        keypoints = torch.stack([xs.float(), ys.float()], dim=1).detach().cpu().numpy().astype(np.float32)
        descriptors = desc_out.detach().cpu().numpy().astype(np.float32)
        return keypoints, descriptors


@lru_cache(maxsize=4)
def get_superpoint_extractor(
    weights_path: str,
    device: str,
    nms_radius: int,
    keypoint_threshold: float,
    max_keypoints: int,
    remove_border: int,
) -> SuperPointExtractor:
    resolved_path = Path(weights_path).expanduser().resolve()
    requested = device.strip().lower()
    if requested in ("cuda", "cuda:0") and not torch.cuda.is_available():
        LOGGER.warning("SuperPoint: CUDA requested but not available; falling back to CPU.")
        requested = "cpu"
    extractor = SuperPointExtractor(
        weights_path=resolved_path,
        device=torch.device(requested),
        nms_radius=nms_radius,
        keypoint_threshold=keypoint_threshold,
        max_keypoints=max_keypoints,
        remove_border=remove_border,
    )
    return extractor


def match_descriptors_l2(
    desc_ref: np.ndarray,
    desc_tgt: np.ndarray,
    *,
    ratio: float,
    cross_check: bool,
    max_matches: int,
) -> List[cv2.DMatch]:
    if desc_ref.size == 0 or desc_tgt.size == 0:
        return []
    if desc_ref.dtype != np.float32:
        desc_ref = desc_ref.astype(np.float32, copy=False)
    if desc_tgt.dtype != np.float32:
        desc_tgt = desc_tgt.astype(np.float32, copy=False)

    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    knn = matcher.knnMatch(desc_ref, desc_tgt, k=2)
    good: List[cv2.DMatch] = []
    ratio_f = float(ratio)
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance <= ratio_f * n.distance:
            good.append(m)

    if cross_check and good:
        knn_back = matcher.knnMatch(desc_tgt, desc_ref, k=1)
        back_best: Dict[int, int] = {}
        for pair in knn_back:
            if not pair:
                continue
            back_best[pair[0].queryIdx] = pair[0].trainIdx
        good = [m for m in good if back_best.get(m.trainIdx) == m.queryIdx]

    good.sort(key=lambda m: m.distance)
    if max_matches > 0:
        good = good[: int(max_matches)]
    return good


def estimate_superpoint_affine_transform(
    reference: np.ndarray,
    target: np.ndarray,
    *,
    ref_mask: Optional[np.ndarray],
    tgt_mask: Optional[np.ndarray],
    weights_path: Path,
    device: str,
    nms_radius: int,
    keypoint_threshold: float,
    max_keypoints: int,
    remove_border: int,
    match_ratio: float,
    cross_check: bool,
    max_matches: int,
    ransac_reproj_threshold: float,
    min_inliers: int,
    min_scale: float,
    max_scale: float,
) -> Optional[np.ndarray]:
    extractor = get_superpoint_extractor(
        str(weights_path),
        device,
        int(nms_radius),
        float(keypoint_threshold),
        int(max_keypoints),
        int(remove_border),
    )
    kpts_ref, desc_ref = extractor.extract(reference, mask=ref_mask)
    kpts_tgt, desc_tgt = extractor.extract(target, mask=tgt_mask)

    if kpts_ref.shape[0] < 10 or kpts_tgt.shape[0] < 10:
        return None

    matches = match_descriptors_l2(
        desc_ref,
        desc_tgt,
        ratio=match_ratio,
        cross_check=cross_check,
        max_matches=max_matches,
    )
    if len(matches) < 8:
        return None

    ref_pts = np.float32([kpts_ref[m.queryIdx] for m in matches]).reshape(-1, 1, 2)
    tgt_pts = np.float32([kpts_tgt[m.trainIdx] for m in matches]).reshape(-1, 1, 2)
    matrix, inliers = cv2.estimateAffinePartial2D(
        tgt_pts,
        ref_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=float(ransac_reproj_threshold),
        maxIters=5000,
        confidence=0.99,
        refineIters=10,
    )
    if matrix is None or inliers is None:
        return None

    inlier_count = int(np.count_nonzero(inliers))
    if inlier_count < int(min_inliers):
        return None

    scale_x = float(np.hypot(matrix[0, 0], matrix[0, 1]))
    scale_y = float(np.hypot(matrix[1, 0], matrix[1, 1]))
    if not (float(min_scale) <= scale_x <= float(max_scale) and float(min_scale) <= scale_y <= float(max_scale)):
        LOGGER.debug(
            "Rejecting SuperPoint affine due to scale anomaly (scale_x=%.3f, scale_y=%.3f)",
            scale_x,
            scale_y,
        )
        return None

    LOGGER.debug(
        "SuperPoint: kpts_ref=%d, kpts_tgt=%d, matches=%d, inliers=%d",
        kpts_ref.shape[0],
        kpts_tgt.shape[0],
        len(matches),
        inlier_count,
    )
    return matrix


def _resize_keep_aspect(
    image: np.ndarray, *, max_dim: int, interpolation: int
) -> Tuple[np.ndarray, float, float]:
    """Resize image so that max(H,W) == max_dim (if needed).

    Returns (resized, sx, sy) where original_coord = resized_coord * (sx, sy).
    """
    if max_dim <= 0:
        return image, 1.0, 1.0

    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_dim:
        return image, 1.0, 1.0

    scale = float(max_dim) / float(longest)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    sx = float(width) / float(new_w)
    sy = float(height) / float(new_h)
    return resized, sx, sy


def _pad_to_multiple(image: np.ndarray, multiple: int, value: int = 0) -> Tuple[np.ndarray, int, int]:
    if multiple <= 1:
        return image, 0, 0
    height, width = image.shape[:2]
    pad_h = (-height) % multiple
    pad_w = (-width) % multiple
    if pad_h == 0 and pad_w == 0:
        return image, 0, 0
    padded = np.pad(
        image,
        ((0, pad_h), (0, pad_w)),
        mode="constant",
        constant_values=int(value),
    )
    return padded, pad_h, pad_w


def _load_loftr_state_dict(weights_path: Path) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(str(weights_path), map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model", "weights"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                checkpoint = value
                break

    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported LoFTR checkpoint format: {type(checkpoint)}")

    state_dict: Dict[str, torch.Tensor] = {}
    for key, value in checkpoint.items():
        if not isinstance(value, torch.Tensor):
            continue
        name = str(key)
        for prefix in ("module.", "loftr.", "matcher.", "model."):
            if name.startswith(prefix):
                name = name[len(prefix) :]
        state_dict[name] = value
    if not state_dict:
        raise ValueError(f"No tensor weights found in checkpoint: {weights_path}")
    return state_dict


@lru_cache(maxsize=2)
def get_loftr_matcher(
    pretrained: str,
    weights_path: Optional[str],
    device: str,
):
    try:
        from kornia.feature import LoFTR  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "LoFTR requires kornia. Install with: pip install kornia"
        ) from exc

    requested = device.strip().lower()
    if requested in ("cuda", "cuda:0") and not torch.cuda.is_available():
        LOGGER.warning("LoFTR: CUDA requested but not available; falling back to CPU.")
        requested = "cpu"
    torch_device = torch.device(requested)

    if weights_path:
        matcher = LoFTR(pretrained=None)
        state = _load_loftr_state_dict(Path(weights_path))
        matcher.load_state_dict(state, strict=False)
    else:
        matcher = LoFTR(pretrained=pretrained)
    matcher = matcher.to(torch_device).eval()
    return matcher


@torch.no_grad()
def estimate_loftr_affine_transform(
    reference: np.ndarray,
    target: np.ndarray,
    *,
    ref_mask: Optional[np.ndarray],
    tgt_mask: Optional[np.ndarray],
    device: str,
    pretrained: str,
    weights_path: Optional[Path],
    max_dim: int,
    min_confidence: float,
    max_matches: int,
    ransac_reproj_threshold: float,
    min_inliers: int,
    min_scale: float,
    max_scale: float,
) -> Optional[np.ndarray]:
    if reference.ndim != 2 or target.ndim != 2:
        raise ValueError("LoFTR expects 2D grayscale images.")
    if reference.dtype != np.uint8 or target.dtype != np.uint8:
        raise ValueError("LoFTR expects uint8 grayscale; call normalize_to_uint8 first.")

    ref_small, sx_ref, sy_ref = _resize_keep_aspect(
        reference, max_dim=int(max_dim), interpolation=cv2.INTER_AREA
    )
    tgt_small, sx_tgt, sy_tgt = _resize_keep_aspect(
        target, max_dim=int(max_dim), interpolation=cv2.INTER_AREA
    )

    ref_small, _, _ = _pad_to_multiple(ref_small, 8, value=0)
    tgt_small, _, _ = _pad_to_multiple(tgt_small, 8, value=0)

    matcher = get_loftr_matcher(
        str(pretrained),
        str(weights_path) if weights_path is not None else None,
        str(device),
    )
    torch_device = next(matcher.parameters()).device

    image0 = (
        torch.from_numpy(ref_small.astype(np.float32) / 255.0)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(torch_device)
    )
    image1 = (
        torch.from_numpy(tgt_small.astype(np.float32) / 255.0)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(torch_device)
    )

    input_dict: Dict[str, torch.Tensor] = {"image0": image0, "image1": image1}
    correspondences = matcher(input_dict)

    kpts0 = correspondences.get("keypoints0")
    kpts1 = correspondences.get("keypoints1")
    conf = correspondences.get("confidence")
    if kpts0 is None or kpts1 is None or conf is None:
        return None

    k0 = kpts0.detach().cpu().numpy().astype(np.float32)
    k1 = kpts1.detach().cpu().numpy().astype(np.float32)
    c = conf.detach().cpu().numpy().astype(np.float32)

    if k0.shape[0] < 8:
        return None

    keep = c >= float(min_confidence)
    k0 = k0[keep]
    k1 = k1[keep]
    c = c[keep]
    if k0.shape[0] < 8:
        return None

    if ref_mask is not None:
        ref_mask_small, _, _ = _resize_keep_aspect(
            (ref_mask > 0).astype(np.uint8),
            max_dim=int(max_dim),
            interpolation=cv2.INTER_NEAREST,
        )
        ref_mask_small, _, _ = _pad_to_multiple(ref_mask_small, 8, value=0)
    else:
        ref_mask_small = None

    if tgt_mask is not None:
        tgt_mask_small, _, _ = _resize_keep_aspect(
            (tgt_mask > 0).astype(np.uint8),
            max_dim=int(max_dim),
            interpolation=cv2.INTER_NEAREST,
        )
        tgt_mask_small, _, _ = _pad_to_multiple(tgt_mask_small, 8, value=0)
    else:
        tgt_mask_small = None

    if ref_mask_small is not None:
        x0 = np.clip(np.round(k0[:, 0]).astype(np.int32), 0, ref_mask_small.shape[1] - 1)
        y0 = np.clip(np.round(k0[:, 1]).astype(np.int32), 0, ref_mask_small.shape[0] - 1)
        keep0 = ref_mask_small[y0, x0] > 0
    else:
        keep0 = np.ones((k0.shape[0],), dtype=bool)

    if tgt_mask_small is not None:
        x1 = np.clip(np.round(k1[:, 0]).astype(np.int32), 0, tgt_mask_small.shape[1] - 1)
        y1 = np.clip(np.round(k1[:, 1]).astype(np.int32), 0, tgt_mask_small.shape[0] - 1)
        keep1 = tgt_mask_small[y1, x1] > 0
    else:
        keep1 = np.ones((k1.shape[0],), dtype=bool)

    keep = keep0 & keep1
    k0 = k0[keep]
    k1 = k1[keep]
    c = c[keep]
    if k0.shape[0] < 8:
        return None

    # Limit matches (keep strongest confidence).
    max_m = int(max_matches)
    if max_m > 0 and k0.shape[0] > max_m:
        idx = np.argsort(-c)[:max_m]
        k0 = k0[idx]
        k1 = k1[idx]

    # Scale back to original preserved coords.
    k0[:, 0] *= sx_ref
    k0[:, 1] *= sy_ref
    k1[:, 0] *= sx_tgt
    k1[:, 1] *= sy_tgt

    matrix, inliers = cv2.estimateAffinePartial2D(
        k1.reshape(-1, 1, 2),
        k0.reshape(-1, 1, 2),
        method=cv2.RANSAC,
        ransacReprojThreshold=float(ransac_reproj_threshold),
        maxIters=5000,
        confidence=0.999,
        refineIters=10,
    )
    if matrix is None or inliers is None:
        return None

    inlier_count = int(np.count_nonzero(inliers))
    if inlier_count < int(min_inliers):
        return None

    scale_x = float(np.hypot(matrix[0, 0], matrix[0, 1]))
    scale_y = float(np.hypot(matrix[1, 0], matrix[1, 1]))
    if not (float(min_scale) <= scale_x <= float(max_scale) and float(min_scale) <= scale_y <= float(max_scale)):
        LOGGER.debug(
            "Rejecting LoFTR affine due to scale anomaly (scale_x=%.3f, scale_y=%.3f)",
            scale_x,
            scale_y,
        )
        return None

    LOGGER.debug("LoFTR: matches=%d inliers=%d", k0.shape[0], inlier_count)
    return matrix.astype(np.float32)


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
    stage1_method: str,
    stage2_method: str,
    loftr_device: str,
    loftr_pretrained: str,
    loftr_weights: Optional[Path],
    loftr_max_dim: int,
    loftr_min_confidence: float,
    loftr_max_matches: int,
    loftr_ransac_reproj_threshold: float,
    loftr_min_inliers: int,
    loftr_min_scale: float,
    loftr_max_scale: float,
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

    if enable_secondary_alignment and (stage1_method != "none" or stage2_method != "none"):
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

            def apply_refine_current(matrix: np.ndarray) -> None:
                nonlocal rgb_preserved, rgb_aligned, rgb_gray_preserved, rgb_crop_offset, rgb_mask_preserved, rgb_mask_aligned

                # IMPORTANT: for multi-stage alignment we must warp the *current* images,
                # not the original `preserved_image/mask_preserved` captured before stage1.
                if key == "jpg":
                    cur_img = rgb_preserved
                    cur_mask = rgb_mask_preserved
                else:
                    cur_img = spectral_preserved_raw[int(key)]
                    cur_mask = spectral_masks_preserved[int(key)]

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

                if key == "jpg":
                    rgb_preserved = refined_raw
                    rgb_aligned = crop_to_shape(
                        refined_raw, reference_crop_offset, rgb_height, rgb_width
                    )
                    rgb_gray_preserved = to_grayscale(refined_raw)
                    rgb_crop_offset = reference_crop_offset
                    rgb_mask_preserved = refined_mask_u8
                    rgb_mask_aligned = crop_to_shape(
                        refined_mask_u8, reference_crop_offset, rgb_height, rgb_width
                    ).astype(bool)
                else:
                    spectral_preserved_raw[int(key)] = refined_raw
                    spectral_preserved_u8[int(key)] = refined_u8
                    spectral_masks_preserved[int(key)] = refined_mask_u8
                    spectral_crop_offsets[int(key)] = reference_crop_offset
                    aligned_bands_raw[int(key)] = crop_to_shape(
                        refined_raw,
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
                        refined_mask_u8,
                        spectral_crop_offsets[int(key)],
                        spectral_height,
                        spectral_width,
                    ).astype(bool)

            # Stage 1: ORB (coarse)
            if stage1_method == "orb":
                m1 = estimate_affine_transform(ref_gray, target_gray)
                if m1 is not None:
                    apply_refine_current(m1)
                else:
                    LOGGER.debug("Stage1 ORB failed for %s", key)
            elif stage1_method == "loftr":
                m1 = estimate_loftr_affine_transform(
                    ref_gray,
                    target_gray,
                    ref_mask=ref_mask_preserved,
                    tgt_mask=mask_preserved,
                    device=loftr_device,
                    pretrained=loftr_pretrained,
                    weights_path=loftr_weights,
                    max_dim=loftr_max_dim,
                    min_confidence=loftr_min_confidence,
                    max_matches=loftr_max_matches,
                    ransac_reproj_threshold=loftr_ransac_reproj_threshold,
                    min_inliers=loftr_min_inliers,
                    min_scale=loftr_min_scale,
                    max_scale=loftr_max_scale,
                )
                if m1 is not None:
                    apply_refine_current(m1)
                else:
                    LOGGER.debug("Stage1 LoFTR failed for %s", key)
            elif stage1_method != "none":
                raise ValueError(f"Unknown stage1_method: {stage1_method}")

            # Stage 2: ECC (fine)
            if stage2_method == "ecc":
                # Re-pick aligned views after Stage1 updates.
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
                        m2_aligned, ref_crop_offset=ref_crop_off, tgt_crop_offset=tgt_crop_off
                    )
                    apply_refine_current(m2)
                else:
                    LOGGER.debug("Stage2 ECC failed for %s", key)
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
    stage1_method: str,
    stage2_method: str,
    loftr_device: str,
    loftr_pretrained: str,
    loftr_weights: Optional[Path],
    loftr_max_dim: int,
    loftr_min_confidence: float,
    loftr_max_matches: int,
    loftr_ransac_reproj_threshold: float,
    loftr_min_inliers: int,
    loftr_min_scale: float,
    loftr_max_scale: float,
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
                stage1_method,
                stage2_method,
                loftr_device,
                loftr_pretrained,
                loftr_weights,
                loftr_max_dim,
                loftr_min_confidence,
                loftr_max_matches,
                loftr_ransac_reproj_threshold,
                loftr_min_inliers,
                loftr_min_scale,
                loftr_max_scale,
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
        description="Batch-align RGB and spectral HDR data into stacks (LoFTR variant; preserve original spectral dtype/bit depth)."
    )
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=Path("/mnt/d/Project/master-graduation-project/data/oil/train/feedback"),
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
        default=Path("/mnt/d/Project/master-graduation-project/data/oil/train/feedback/aligned_full_hdr"),
        help="Output folder for RGB + 7-band stacks "
        "(default: dataset_root / 'aligned_full_tif').",
    )
    parser.add_argument(
        "--output-nir",
        type=Path,
        default=Path("/mnt/d/Project/master-graduation-project/data/oil/train/feedback/aligned_nir_tif"),
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
        choices=["none", "orb", "loftr"],
        default="none",
        help="Secondary stage1 method (coarse): none/orb/loftr.",
    )
    parser.add_argument(
        "--stage2-method",
        choices=["none", "ecc"],
        default="none",
        help="Secondary stage2 method (fine): none/ecc.",
    )
    parser.add_argument(
        "--loftr-pretrained",
        type=str,
        default="outdoor",
        help="LoFTR pretrained weights name (kornia): outdoor/indoor (used when --loftr-weights is not set).",
    )
    parser.add_argument(
        "--loftr-weights",
        type=Path,
        default=None,
        help="Path to LoFTR checkpoint (.ckpt/.pth). If set, no online download is required.",
    )
    parser.add_argument(
        "--loftr-device",
        type=str,
        default="cuda",
        help="Torch device for LoFTR inference (e.g. cuda/cpu).",
    )
    parser.add_argument(
        "--loftr-max-dim",
        type=int,
        default=1024,
        help="Downscale images so max(H,W) <= this value before LoFTR (<=0 disables).",
    )
    parser.add_argument(
        "--loftr-min-confidence",
        type=float,
        default=0.5,
        help="Minimum LoFTR match confidence to keep.",
    )
    parser.add_argument(
        "--loftr-max-matches",
        type=int,
        default=2000,
        help="Keep at most this many LoFTR matches by confidence (<=0 keeps all).",
    )
    parser.add_argument(
        "--loftr-ransac-reproj-threshold",
        type=float,
        default=3.0,
        help="RANSAC reprojection threshold (pixels) for LoFTR affine estimation.",
    )
    parser.add_argument(
        "--loftr-min-inliers",
        type=int,
        default=12,
        help="Minimum inlier matches for accepting LoFTR affine.",
    )
    parser.add_argument(
        "--loftr-min-scale",
        type=float,
        default=0.7,
        help="Reject LoFTR affine if estimated scale is below this bound.",
    )
    parser.add_argument(
        "--loftr-max-scale",
        type=float,
        default=1.3,
        help="Reject LoFTR affine if estimated scale is above this bound.",
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

    # Backward compatible behavior:
    # - This script targets LoFTR as the default stage1 method.
    stage1_method = args.stage1_method
    stage2_method = args.stage2_method
    if args.enable_secondary_align and stage1_method == "none":
        stage1_method = "loftr"

    enable_secondary = args.enable_secondary_align or stage1_method != "none" or stage2_method != "none"

    if stage1_method == "loftr":
        try:
            import kornia  # type: ignore  # noqa: F401
        except Exception as exc:  # pragma: no cover
            raise SystemExit(
                "LoFTR requires kornia. Install with: pip install kornia"
            ) from exc
        if args.loftr_weights is not None and not args.loftr_weights.is_file():
            raise SystemExit(f"Missing LoFTR weights: {args.loftr_weights}")

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
        secondary_alignment=enable_secondary,
        stage1_method=stage1_method,
        stage2_method=stage2_method,
        loftr_device=args.loftr_device,
        loftr_pretrained=args.loftr_pretrained,
        loftr_weights=args.loftr_weights,
        loftr_max_dim=args.loftr_max_dim,
        loftr_min_confidence=args.loftr_min_confidence,
        loftr_max_matches=args.loftr_max_matches,
        loftr_ransac_reproj_threshold=args.loftr_ransac_reproj_threshold,
        loftr_min_inliers=args.loftr_min_inliers,
        loftr_min_scale=args.loftr_min_scale,
        loftr_max_scale=args.loftr_max_scale,
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
